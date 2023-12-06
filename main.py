import os
import time

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange

from aq_engine import AQUtil
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

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def quantize_model(model, args, device):
    """main entry point to functions for model quantization"""
    tick = time.time()

    print("Loading data ...")
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=model.seqlen,
    )
    results = quantize_aq(model, dataloader, args, device)
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
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(dev)
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
def quantize_aq(model, dataloader, args, device):
    print("\nStarting AQ quantization ...")

    inps, forward_args = get_inps(model, dataloader, args, dev="cpu" if args.offload_activations else device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    quantizers = {}
    overall_bits = 0
    model_number_of_params = 0
    layers = get_layers(model)
    for i in range(len(layers)):
        print(f"\n---------------- Layer {i} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device  # quantized layer will return there
        print(f"{layer_dev_original=}")
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}

            aq_handlers = {}
            for sublayer_name in subset:
                aq_handlers[sublayer_name] = AQUtil(subset[sublayer_name])

            def add_batch(name):
                def tmp(_, inp, out):
                    aq_handlers[name].add_batch(inp[0].data)  # noqa: F821

                return tmp

            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))
            for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                outs[j] = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
                if args.offload_activations:
                    outs[j] = outs[j].cpu()
            for h in handles:
                h.remove()

            torch.cuda.empty_cache()

            for sublayer_name in subset:
                print(f"Quantizing module {sublayer_name} of layer {i}")
                quantized = aq_handlers[sublayer_name].quantize(
                    args=args,
                    verbose=True,
                    save_quantization=False,
                )

                if save:
                    quantized.name = sublayer_name
                    full_path = save + "/" + str(i) + "/"
                    os.makedirs(full_path, exist_ok=True)
                    torch.save(quantized.state_dict(), full_path + sublayer_name)

                with torch.no_grad():
                    aq_handlers[sublayer_name].layer.weight.data = quantized().to(
                        aq_handlers[sublayer_name].layer.weight.data.dtype)
                overall_bits += torch.numel(quantized.codes) * args.nbits_per_codebook + torch.numel(
                    quantized.codebooks) * 16
                model_number_of_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits/model_number_of_params)
                quantizers["model.layers.%d.%s" % (i, sublayer_name)] = ()  # to be updated

        out_losses = []
        for j in trange(args.nsamples, desc="calc outs after quantization", leave=False):
            outs_batch = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
            if not args.skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j].to(layer_dev))
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                out_losses.append(outs_batch_loss.item())
            outs[j] = outs_batch
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        del outs_batch

        layers[i] = layer.to(layer_dev_original)
        del layer
        del aq_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
        stats_payload["Step"] = i
        if args.wandb:
            wandb.log({"out_loss": stats_payload["out_loss"]},step = i)
            wandb.log({"layer_time": stats_payload["layer_time"]},step = i)
        print(stats_payload)

    print("=====================\nFinal stats:")
    if save:
        torch.save(vars(args), save + "/args.pt")
        already_saved_weights = set()
        for name, layer in nn.ModuleList(get_layers(model)).named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                already_saved_weights.add(layer.weight)
        not_quantized_weights = {
            name: param for name, param in model.named_parameters() if param not in already_saved_weights
        }
        torch.save(not_quantized_weights, save + "/not_quantized_weights.pt")

    if args.wandb:
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        wandb.log({"Avg_bits": overall_bits/model_number_of_params})
    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers


@torch.no_grad()
def perplexity_eval(model, testenc, args, dev):
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
        shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    get_model_head(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl.item()})

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
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=False, help="Path to save quantized statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--grouping",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=500,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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

        wandb.init(
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
        perplexity_eval(model, testloader, args, device)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
