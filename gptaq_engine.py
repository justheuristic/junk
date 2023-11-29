import math

import torch
from tqdm.auto import trange

from src.aq import QuantizedWeight, _reconstruct_weight  # see adjacent file (aq.py)


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
        args,
        **kwargs,
    ) -> QuantizedWeight:
        """
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
            if epoch % args.print_frequency == 0 and verbose:
                print(f"loss={loss.item():.10f}\t")
            if (epoch + 1) % args.beam_search_epochs == 0:
                if (epoch + 1) % args.big_beam_search_epochs == 0 and verbose:
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
