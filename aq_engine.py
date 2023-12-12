import math
import random

import torch
from tqdm.auto import trange

from src.aq import QuantizedWeight


class AQUtil:
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

    @torch.enable_grad()
    def quantize(
        self,
        *,
        verbose=True,
        args,
    ) -> QuantizedWeight:
        """ create a QuantizedWeight based on the collected XTX data"""
        reference_weight = self.layer.weight.detach().cuda().float()
        quantized_weight = QuantizedWeight.create_with_init_params(
            reference_weight=reference_weight,
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.max_points_per_centroid,
            devices = args.devices,
            verbose=True,
        )

        opt = torch.optim.Adam(quantized_weight.parameters(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        for epoch in trange(args.num_epochs):
            delta_weight = (quantized_weight() - reference_weight).double()
            loss = (delta_weight @ self.H.double()).flatten() @ delta_weight.flatten() / len(delta_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % args.print_frequency == 0 and verbose:
                print(f"loss={loss.item():.10f}\t")
            if (epoch + 1) % args.beam_search_epochs == 0:
                seed_i = random.getrandbits(256)
                quantized_weight.requantize_(
                    XTX=self.H, reference_weight=reference_weight, beam_size=args.beam_size,
                    sparsity_regularizer=args.sparsity_regularizer, dim_rng=random.Random(seed_i), verbose=True)
        return quantized_weight
