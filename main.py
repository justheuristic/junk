import os
import time
from tqdm.auto import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import trange

from src.datautils import get_loaders
from src.modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
)

from src.utils import calc_avg_bits, get_mean_nbits_by_codebook  # see adjacent file (aq.py)

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def quantize_model(model, args, device):
    """main entry point to functions for model quantization"""
    tick = time.time()
    if args.wbits == 16:
        print("not quantizing the model with args.wbits=16", flush=True)
        results = None, args.wbits
    else:
        print("Loading data ...")
        dataloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
        )
        results = quantize_gptaq(model, dataloader, args, device)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results

@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.
    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    forward_arg_names = [
        "attention_mask",
    ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)
    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args


@torch.no_grad()
def perplexity_eval(model, testenc, args, dev, run):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(
        model, testenc, args, dev="cpu" if args.offload_activations else dev, nsamples=nsamples
    )
    outs = torch.zeros_like(inps)
    for k, v in forward_args.items():
        forward_args[k] = v.to(dev) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].to(dev).unsqueeze(0), **forward_args)[0]
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head(model).to(dev)
    testenc = testenc.to(dev)

    nlls = []
    for i in range(nsamples):
        lm_logits = get_lm_logits(inps[i].to(dev), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    get_model_head(model).to(torch.device("cpu"))

    if args.wandb:
        run.log({args.dataset_name: ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="none",
        help="Dataset name [c4, pajama, refinedweb, none, etc.] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to load if specified. Deprecated",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40000,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="relative threshold for     outliers; higher threshold = more outliers.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=10,
        help="#Number of codebooks.",
    )

    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="Out group size .",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=32,
        help="Input group size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16384,
        help="Batch size.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Standart beam size.",
    )
    parser.add_argument(
        "--big_beam_size",
        type=int,
        default=6,
        help="Big beam size.",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=12,
        help="Codebook size 2**nbits_per_codebook .",
    )
    parser.add_argument(
        "--beam_search_epochs",
        type=int,
        default=100,
        help="Beam search epoch .",
    )
    parser.add_argument(
        "--big_beam_search_epochs",
        type=int,
        default=1000,
        help="Do beam search with big .",
    )
    parser.add_argument(
        "--sparsity_regularizer",
        type=int,
        default=0,
        help="Sparsity regularizer.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for kmeans initialization.",
    )
    parser.add_argument(
        "--grouped_quant",
        action="store_true",
        help="Quantize grouped qkv",
    )
    parser.add_argument(
        "--kmeans_init",
        action="store_false",
        help="Init with Kmeans",
    )

    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Batch size.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")

    args = parser.parse_args()

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQ")
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_beam_search_epochs_{args.beam_search_epochs}"
            + f"_big_beam_search_epochs_{args.big_beam_search_epochs}"
        )
        args.group_size = args.in_group_size * args.out_group_size

        run = wandb.init(
            dir=os.getcwd(),
            name=args.exp_name,
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
            settings=wandb.Settings(code_dir="."),
            save_code=True,
            project="AddQuantization",
            entity="rock-and-roll",
        )

    torch.set_num_threads(16)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n============ Load model... ============")
    model = get_model(args.model_path, args.load, args.dtype).train(False)

    print("\n============ Quantizing model... ============")
    if args.wbits < 16 and args.load:
        print("\n Warning: You are quantizing quantized model!")

    quantize_model(model, args, device)

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "ptb", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
            eval_mode=True,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args, device,run)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})