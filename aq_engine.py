from __future__ import annotations
import math
import random
from argparse import Namespace
from typing import Union, Sequence, Optional

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.utils import ellipsis


class AQUtil(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in QuantizedWeight """

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.XTX = torch.zeros((self.columns, self.columns), device=self.device)
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """Accumulate a minibatch of layer inputs and update the hessian (aka X.T @ X)"""
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.XTX *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.XTX += inp.matmul(inp.t())

    @torch.enable_grad()
    def quantize(
        self,
        *,
        verbose=True,
        args: Namespace,
    ) -> QuantizedWeight:
        """ create a QuantizedWeight based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)
        self.reference_weight = self.layer.weight.detach().to(self.device).float()
        self.quantized_weight = QuantizedWeight.create_with_init_params(
            reference_weight=self.reference_weight,
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.max_points_per_centroid,
            devices=args.devices,
            verbose=True,
        )

        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        replicas = None
        if len(args.devices) > 1:
            replicas = torch.nn.parallel.replicate(self, args.devices)
            replicas[0] = self

        for epoch in range(args.num_epochs):
            if len(args.devices) == 1:
                loss = self._compute_mse()
            else:
                loss = self._compute_mse_parallel(args.devices, replicas, differentiable_parameters)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % args.print_frequency == 0 and verbose:
                print(f"loss={loss.item():.10f}\t")

            if (epoch + 1) % args.beam_search_epochs == 0:
                self.beam_search_update_codes_(
                    args.devices, replicas, differentiable_parameters, beam_size=args.beam_size,
                    sparsity_regularizer=args.sparsity_regularizer, verbose=True)
        return self.quantized_weight

    def _compute_mse(self, selection: Union[slice, ellipsis, torch.Tensor] = ...) -> torch.Tensor:
        """
        Compute the activation MSE error = ||X @ quantized_weight - X @ reference_weight||^2
        Use the square-of-difference formula to avoid materializing per-batch predictions
        :param selection:  By default, compute MSE normally. If selection is specified, this method will instead
            compute MSE over a portion of output channels that align with the selected out_groups (for parallelism)
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        """
        assert self.quantized_weight is not None, "must be called inside / after AQUtil.quantize"
        XTX = self.XTX.double()
        delta_weight = (self.quantized_weight(selection) - self.reference_weight[selection]).to(XTX.dtype)
        return (delta_weight @ XTX).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

    def _substitute_and_compute_mse(self, overrides: nn.ParameterDict, selection: slice) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then compute MSE"""
        for param_name, param_value in overrides.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self._compute_mse(selection)

    def _compute_mse_parallel(self,
                              devices: Sequence[torch.device],
                              replicas: Sequence[AQUtil],
                              parameters_to_replicate: nn.ParameterDict) -> torch.Tensor:
        """Compute MSE in parallel over output channels"""
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        shard_size = (num_output_groups - 1) // len(devices) + 1
        active_slices_by_replica = [
            slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))]
        funcs_by_replica = [replica._substitute_and_compute_mse for replica in replicas]
        inputs_by_replica = [(dict(), active_slices_by_replica[0])]  # no overrides needed for 0-th replica
        for i in range(1, len(devices)):
            inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
        mse_components = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)
        return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).sum()

    def _substitute_and_beam_search(self, overrides: nn.ParameterDict, selection: slice, **kwargs) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then run beam search"""
        for param_name, param_value in overrides.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self.quantized_weight.beam_search_update_codes_(
            self.XTX, self.reference_weight, selection=selection, **kwargs)

    @torch.no_grad()
    def beam_search_update_codes_(self, devices: Sequence[torch.device], replicas: Sequence[AQUtil],
                                  parameters_to_replicate: nn.ParameterDict, **kwargs):
        """Update self.quantized_weight.codes in-place via beam search"""
        seed = random.getrandbits(256)
        if len(devices) == 1:  # single device
            assert replicas is None
            self.quantized_weight.beam_search_update_codes_(
                self.XTX, self.reference_weight, dim_rng=random.Random(seed), **kwargs)
        else:
            assert replicas[0] is self
            replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
            num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
            shard_size = (num_output_groups - 1) // len(devices) + 1
            active_slices_by_replica = [
                slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))]

            funcs_by_replica = [replica._substitute_and_beam_search for replica in replicas]
            inputs_by_replica = [(dict(), active_slices_by_replica[i])]
            for i in range(1, len(devices)):
                inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
            kwargs_by_replica = [dict(kwargs) for _ in range(len(devices))]
            new_code_parts_by_replica = torch.nn.parallel.parallel_apply(
                funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices)
            # gather all code parts and assign them to each replica
            for device, replica in zip(devices, replicas):
                replica.codes[...] = Gather.apply(device, 0, new_code_parts_by_replica)


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    assert name in module._parameters
    module._parameters[name] = new_value
