import torch 
from src.aq import QuantizedWeight, _reconstruct_weight  # see adjacent file (aq.py)

import math
from typing import NamedTuple, Optional, Union

import torch
from tqdm.auto import trange


class GPTAQUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
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

    def quantize(
        self,
        *,
        verbose=True,
        save_quantization: bool = False,
        args,
        **kwargs,
    ) -> QuantizedWeight:
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
        quantized_layer = QuantizedWeight(
            weight_shape=self.layer.shape,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            device=self.dev,
            init_kmeans=args.kmeans_init,
            reference_weight=self.layer,
            alpha=args.alpha,
            verbose=True,
        )

        opt = torch.optim.Adam(quantized_layer.parameters(), lr=args.lr, betas=(0.9, 0.95))

        for epoch in trange(args.num_epochs):
            reconstructed_weight = _reconstruct_weight(quantized_layer.codes, quantized_layer.codebooks)
            delta_weight = (reconstructed_weight - self.layer).double()
            loss = (delta_weight @ self.H.double()).flatten() @ delta_weight.flatten() / len(delta_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % args.print_frequency == 0:
                print(f"loss={loss.item():.10f}\t")
            if (epoch + 1) % args.beam_search_epochs == 0:
                if (epoch + 1) % args.big_beam_search_epochs == 0:
                    print("BIG beam search")
                quantized_layer.requantize_(
                    self.H,
                    self.layer,
                    beam_size=args.beam_size if (epoch + 1) % args.big_beam_search_epochs != 0 else args.big_beam_size,
                    sparsity_regularizer=args.sparsity_regularizer,
                    # tip: use const_hparam * quantized_layer.codes.numel()
                    verbose=True,
                )
        return quantized_layer