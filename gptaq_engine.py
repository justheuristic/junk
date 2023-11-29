import torch 
from src.aq import QuantizedWeight, _reconstruct_weight  # see adjacent file (aq.py)

import math
from typing import NamedTuple, Optional, Union

import torch
from tqdm.auto import tqdm


class GPTAQUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        assert self.H is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def calculating_XTX(args):
        """
        Calculating XTX and reference weight
        """
        XTX = torch.zeros(X.shape[-1], X.shape[-1], device=device, dtype=torch.float64)
        for i in range(0, len(X), args.batch_size):
            x_batch = X[i : i + args.batch_size].cuda().double()
            XTX.addmm_(x_batch.T, x_batch, alpha=1 / len(X))
            del x_batch 
        XTX = XTX.float()
        del X
        return XTX

    def quantize(
        self,
        *,
        verbose=True,
        save_quantization: bool = False,
        **kwargs,
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param blocksize: take blocks of this many input features at a time for GPTQ
        :note: blocksize affects runtime and memory, but does not affect the resulting matrix (up to machine precision)
        :param groupsize: fit quantization scaling / statistics to each group of this many input features
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :note: if groupsize_in_dim* is None, use the same quantization statistics across all input features
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :note: the un-quantized columns will be a part of the first returned result
        :param outlier_relative_threshold: threshold used for *UNSTRUCTURED* outliers, relative to
        :note: if keep_last_columns > 0, quantized_dequantized_weights[-keep_last_columns:] will be non-quantized
        :param permutation_order: re-order input features using a certain policy
        :param keep_H: if False, delete the accumulated hessian during quantize; if False, keep the accumulated hessian
        :param simplified_outliers: if True,do not perform leave-one-out evaluation when detecting outliers;
            works faster, but generally worse in perplexity
        :param verbose: if True, display a tqdm progressbar over input columns
        :param sym: if True, base weight quantization is symmetric
        :param perchannel: if True, base weight quantization will learn statistics for each output dimension separately
        :return: a QuantizationResult tuple that contains(
            weight, perm, _unused, _unused, _unused, _unused, quantization_errors, outlier_unstructured_mask
        ), see class QuantizationResult below for details
        """
       

        return 



def compress_aq_la(args, reference_weigh, run):
    """
    Compress W with AQ
    """
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
