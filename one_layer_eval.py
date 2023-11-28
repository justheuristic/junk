import os
import sys
import time
from tqdm.auto import trange
import ipynbname  # pip install ipynbname
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from src.aq import QuantizedWeight, _reconstruct_weight  # see adjacent file (aq.py)
from src.utils import calc_avg_bits, get_mean_nbits_by_codebook  # see adjacent file (aq.py)

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def calculating_XTX(args):
    """
    Calculating XTX and reference weight
    """
    X = torch.load(args.input_loading_dir, map_location="cpu").float().flatten(0, -2)
    XTX = torch.zeros(X.shape[-1], X.shape[-1], device=device, dtype=torch.float64)
    for i in range(0, len(X), args.batch_size):
        x_batch = X[i : i + args.batch_size].cuda().double()
        XTX.addmm_(x_batch.T, x_batch, alpha=1 / len(X))
        del x_batch
    XTX = XTX.float()
    del X
    return XTX


def compress_aq(args, reference_weigh,run):
    '''
    Compress W with AQ
    '''
    quantized_layer = QuantizedWeight(
        weight_shape=reference_weight.shape,
        num_codebooks=args.num_codebooks,
        nbits_per_codebook=args.nbits_per_codebook,
        out_group_size=args.out_group_size,
        in_group_size=args.in_group_size,
        device=device,
        init_kmeans=args.kmeans_init,
        reference_weight=reference_weight,
        alpha=args.alpha,
        verbose=True,
    )

    opt = torch.optim.Adam(quantized_layer.parameters(), lr=args.lr, betas=(0.9, 0.95))

    for epoch in range(args.num_epochs):
        start = time.perf_counter()

        reconstructed_weight = _reconstruct_weight(quantized_layer.codes, quantized_layer.codebooks)
        delta_weight = (reconstructed_weight - reference_weight).double()
        loss = (delta_weight @ XTX.double()).flatten() @ delta_weight.flatten() / len(delta_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

        run.log({"loss": loss.item()}, step=epoch)

        if epoch % args.print_frequency == 0:
            print(f"loss={loss.item():.10f}\t", f"time_on_epoch {epoch} = {time.perf_counter() - start}")
        if (epoch + 1) % args.beam_search_epochs == 0:
            if (epoch + 1) % args.big_beam_search_epochs == 0:
                print("BIG beam search")
            quantized_layer.requantize_(
                XTX,
                reference_weight,
                beam_size=args.beam_size if (epoch + 1) % args.big_beam_search_epochs != 0 else args.big_beam_size,
                sparsity_regularizer=args.sparsity_regularizer,  # tip: use const_hparam * quantized_layer.codes.numel()
                verbose=True,
            )
            if args.sparsity_regularizer != 0:
                sparsity_rate = ((quantized_layer.codes == 0).sum() / quantized_layer.codes.numel()).item()
                run.log({"sparsity rate": sparsity_rate}, step=epoch)
                if (epoch + 1) % args.big_beam_search_epochs == 0:
                    mean_code_nbits = sum(get_mean_nbits_by_codebook(quantized_layer.codes)) / args.num_codebooks
                    run.log({"Mean codebook ldngth nbits": mean_code_nbits}, step=epoch)
                    if args.in_group_size > 1 and args.out_group_size > 1:
                        curr_avg_bits = calc_avg_bits(
                            args.num_codebooks,
                            1,
                            mean_code_nbits,
                            args.nbits_per_codebook,
                            reference_weight.shape[0],
                            reference_weight.shape[1],
                        )
                        run.log({"Avg_bits": curr_avg_bits}, step=epoch)
    return quantized_layer

def load_model(args):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", low_cpu_mem_usage=True
    )

    if args.grouped_quant:
        Q = model.model.layers[args.block_number].self_attn.q_proj.weight.detach()
        K = model.model.layers[args.block_number].self_attn.k_proj.weight.detach()
        V = model.model.layers[args.block_number].self_attn.v_proj.weight.detach()

        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_repeats = model.config.num_attention_heads // model.config.num_key_value_heads
        K_repeated = K.reshape(-1, 1, head_dim, K.shape[1]).tile(1, num_repeats, 1, 1).reshape(Q.shape)
        V_repeated = V.reshape(-1, 1, head_dim, K.shape[1]).tile(1, num_repeats, 1, 1).reshape(Q.shape)

        reference_weight = torch.stack([Q, K_repeated, V_repeated], dim=1).flatten(0, 1).cuda().float()
    else:
        reference_weight = model.model.layers[args.block_number].self_attn.q_proj.weight.detach().to(device).float()
    return model, reference_weight



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "input_loading_dir",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
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
        action="store_True",
        help="Quantize grouped qkv",
    )
    parser.add_argument(
        "--kmeans_init",
        action="store_false",
        help="Init with Kmeans",
    )
    parser.add_argument(
        "--block_number",
        type=int,
        default=10,
        help="number of block of transformer .",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")

    args = parser.parse_args()


    torch.set_num_threads(16)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n============ Load model... ============")
    model, reference_weight = load_model(args)


    in_features, out_features  = reference_weight.shape[0],reference_weight.shape[1]
    estimated_bits_per_param = calc_avg_bits(args.num_codebooks, args.out_group_size, args.in_group_size,
                                                args.nbits_per_codebook, in_features,out_features)
    print("Estimated bits / param", estimated_bits_per_param)
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
        run = wandb.init(
            dir=os.getcwd(),
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
            settings=wandb.Settings(code_dir="."),
            save_code=True,
            project="AddQuantization",
            entity="rock-and-roll",
        )

        run.log({"Avg_bits": estimated_bits_per_param})

    print("============  Calculating XTX ... ============")
    XTX = calculating_XTX(args)

    print("\n============ Quantizing layer... ============")
    compress_aq(args,reference_weight,run)
