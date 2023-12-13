import os
import time

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange

from aq_engine import AQUtil
from src.datautils import get_loaders, set_seed
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
def get_inps(model, data_iterable, args, device, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching layer inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples or len(data_iterable)
    assert nsamples is not None

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)

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
                model(batch[0].to(device))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(device))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args


@torch.no_grad()
def quantize_aq(model, dataloader, args, device):
    print("\nStarting AQ quantization ...")

    inps, forward_args = get_inps(model, dataloader, args, device="cpu" if args.offload_activations else device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    quantizers = {}
    overall_bits = 0
    model_number_of_params = 0
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        layer_device_original = next(layers[layer_index].parameters()).device  # quantized layer will return there
        print(f"{layer_device_original=}")
        if layer_device_original.type != "cuda":
            layer = layers[layer_index].to(device)
        else:
            layer = layers[layer_index]
        layer_device = next(layers[layer_index].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_device) if isinstance(v, torch.Tensor) else v

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
            for j in trange(len(inps), desc="calc outs before quantization", leave=False):
                outs[j].copy_(layer(inps[j].to(layer_device).unsqueeze(0), **forward_args)[0].view_as(outs[j]),
                              non_blocking=True)
            for h in handles:
                h.remove()

            torch.cuda.empty_cache()

            for sublayer_name in subset:
                print(f"Quantizing module {sublayer_name} of layer {layer_index}")
                quantized = aq_handlers[sublayer_name].quantize(args=args, verbose=True)

                if save:
                    quantized.name = sublayer_name
                    full_path = save + "/" + str(layer_index) + "/"
                    os.makedirs(full_path, exist_ok=True)
                    print("Saved params:", quantized.init_params)
                    torch.save((quantized.state_dict(), quantized.init_params), full_path + sublayer_name)

                with torch.no_grad():
                    aq_handlers[sublayer_name].layer.weight.data = quantized().to(
                        aq_handlers[sublayer_name].layer.weight.data.dtype)

                weight_avg_bits = quantized.estimate_nbits_per_parameter()
                overall_bits += int(weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data))
                model_number_of_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits / model_number_of_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()  # to be updated

        out_losses = []
        for j in trange(len(inps), desc="calc outs after quantization", leave=False):
            outs_batch = layer(inps[j].to(layer_device).unsqueeze(0), **forward_args)[0]
            if not args.skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j].to(layer_device))
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                out_losses.append(outs_batch_loss.item())
            outs[j].copy_(outs_batch.reshape_as(outs[j]), non_blocking=True)
        del outs_batch

        layers[layer_index] = layer.to(layer_device_original)
        del layer
        del aq_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
        stats_payload["Step"] = layer_index
        if args.wandb:
            wandb.log({"out_loss": stats_payload["out_loss"]}, step=layer_index)
            wandb.log({"layer_time": stats_payload["layer_time"]}, step=layer_index)
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
        wandb.log({"Avg_bits": overall_bits / model_number_of_params})
    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers


@torch.no_grad()
def perplexity_eval(model, testenc, args, device):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(
        model, testenc, args, device="cpu" if args.offload_activations else device, nsamples=nsamples
    )
    outs = torch.zeros_like(inps)
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)

        for j in range(nsamples):
            outs[j].copy_(layer(inps[j].to(device).unsqueeze(0), **forward_args)[0])
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head(model).to(device)
    testenc = testenc.to(device)

    nlls = []
    for i in range(nsamples):
        lm_logits = get_lm_logits(inps[i].to(device), model)
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
        help="Dataset name [c4, pajama, refinedweb] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument("--nsamples", type=int, default=None,
                        help="Number of calibration data samples.If None take all calibration data.")
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=False, help="Path to save quantized statistics.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for calibration data and initialization. "
                             "Note that the main training is not strictly deterministic.")
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
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of beam search rounds before the optimization is forcibly stopped.",
    )
    parser.add_argument(
        "--relative_mse_tolerance",
        type=float,
        default=None,
        help="Stop training when when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Run (this many) Adam updates before every beam search round",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        choices=[2048, 4096],
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=16,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--scale_nbits",
        type=int,
        default=0,
        help="Number of bits dedicated to the learnable group-wise scale. 0 means do not use group-wise scales "
             "(still has row-wise scales), 1-15 means using per-group scales quantized to this many bits, "
             "16+ means use per-group scales but do not quantize them"
    )
    parser.add_argument(
        "--codebook_value_nbits",
        type=int,
        default=16,
        help="If below 16, quantize the values in each codebook with the specified number of bits"
    )
    parser.add_argument(
        "--codebook_value_num_groups",
        type=int,
        default=1,
        help="Split codebook vectors into this many groups for quantizations. Only used when quantized codebooks."
    )
    parser.add_argument(
        "--init_max_iter",
        type=int,
        default=100,
        help="Number of K-Means iterations used to initialize codebooks and codes",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Whether to use faiss.Kmeans when initializing codebooks and codes",
    )
    parser.add_argument(
        "--max_points_per_centroid",
        type=int,
        default=None,
        help="During K-means initialzation, sample (this_many * 2 ^ nbits_per_codebook) points for training K-means",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="#Number of codebooks per layer",
    )
    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="How many output units are quantized together",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="How many input features are quantized together",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Keep top-(this_many) best candidates for each codebook when finding optimal codes",
    )
    parser.add_argument(
        "--sparsity_regularizer",
        type=int,
        default=0,
        help="An (optional) regularizer that promotes sparsity. Subtracted from loss for each zero code (index)",
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Print Adam progress after each print_frequency updates",
    )
    parser.add_argument(
        '--devices',
                        metavar='N',
                        type=str,
                        nargs='+',
                        default=None,
                        help='List of devices')
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model in",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")


    torch.set_num_threads(16)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.devices is None:
        args.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    else:
        args.devices = [torch.device(device_str) for device_str in range(args.devices)]
    assert all(isinstance(device, torch.device) for device in args.devices)

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
                os.environ.get("WANDB_NAME", "AQ")
                + f"_num_codebooks_{args.num_codebooks}"
                + f"_out_group_size_{args.out_group_size}"
                + f"_in_group_size_{args.in_group_size}"
                + f"_nbits_per_codebook_{args.nbits_per_codebook}"
                + f"_codebook_value_nbits_{args.codebook_value_nbits}"
                + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
                + f"_scale_nbits_{args.scale_nbits}"
                + f"_steps_per_epoch_{args.steps_per_epoch}"
                + f"_init_max_iter{args.init_max_iter}"
                + f"_{len(args.devices)}gpus"

        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            dir=os.getcwd(),
            name=args.exp_name,
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
            settings=wandb.Settings(code_dir="."),
            project=os.environ.get("WANDB_PROJECT", f"AQ_{list(filter(len, args.model_path.split('/')))[-1]}"),
            entity=os.environ.get("WANDB_ENTITY", "rock-and-roll"),
            save_code=True,
        )

    print("\n============ Load model... ============")
    model = get_model(args.model_path, args.load, args.dtype, args.model_seqlen).train(False)

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
