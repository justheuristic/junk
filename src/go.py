from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
from typing import Sequence, Iterator, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Gather

from aq_engine import replace_parameter_
from src.aq import QuantizedWeight
from src.utils import iterate_minibatches


@torch.enable_grad()
def finetune_groupwise(
    *,
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    args: Namespace,
    verbose: bool = True,
    **kwargs,
) -> QuantizedWeight:
    """
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param layer: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param inps: a list of tensors of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs: a list of tensors of previous output activations, [nsamples_per_device, seq_len, hidden_size]
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
    assert isinstance(inps, (list, tuple)) and isinstance(inps, (list, tuple))
    assert len(inps) == len(outs) == len(args.devices)
    for i in range(len(args.devices)):
        assert isinstance(inps[i], torch.Tensor) and isinstance(outs[i], torch.Tensor)
        if not args.offload_activations:
            assert inps[i].device == outs[i].device == args.devices[i], (inps[i].device, outs[i].device, args.devices)
        else:
            assert inps[i].device == outs[i].device == torch.device('cpu')
            assert inps[i].is_pinned() and outs[i].is_pinned()

    # replicate non-trainable parameters to each GPU
    replicas = kwargs_by_device = None
    if len(args.devices) > 1:
        replicas = torch.nn.parallel.replicate(layer, args.devices)
        replicas[0] = layer
        kwargs_by_device = []
        for device in args.devices:
            kwargs_by_device.append({k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                                     for k, v in kwargs.items()})

    # initialize trainable parameters on main device
    #TODO this code is a mess; unfuck it
    # TODO -- vvvvvvvv CRAPPY CODE THAT SHOULD BE REFACTORED vvvvvvvv
    # intent: for each replica, store a pair (submodule, name) where to put each trainable param
    differentiable_parameters_by_name = {name: param for name, param in layer.named_parameters() if param.requires_grad}
    param_names, differentiable_parameters = zip(*differentiable_parameters_by_name.items())
    differentiable_parameters = nn.ParameterList(differentiable_parameters)
    substitution_tables = []
    if replicas:
        param_to_name = {param: name for name, param in differentiable_parameters_by_name.items()}
        param_occurences = defaultdict(list)   # param_name -> List [ Tuple [submodule name, attr name] ]
        for submodule_name, submodule in layer.named_modules():
            for attr_name, param in submodule.named_parameters(recurse=False):  # immediate params (excluding children)
                if param in param_to_name:
                    param_name = param_to_name[param]
                    param_occurences[param_name].append((submodule_name, attr_name))
        assert len(param_occurences) == len(differentiable_parameters), "internal error: not all trainable parameters were found"

        for replica in replicas:
            substitution_table = list()  # for each master param -> List[ Tuple[replica submodule, attr name] ]
            replica_modules_by_name: Dict[str, nn.Module] = dict(replica.named_modules())

            for param_name, master_param in zip(param_names, differentiable_parameters):
                param_substitutions = list()
                for submodule_name, attr_name in param_occurences[param_name]:
                    param_substitutions.append((replica_modules_by_name[submodule_name], attr_name))
                substitution_table.append(param_substitutions)
            substitution_tables.append(substitution_table)

    # TODO -- ^^^^^^^^ CRAPPY CODE THAT SHOULD BE REFACTORED ^^^^^^^^
    # end of crappy code


    print(f"Fine-tuning {sum(param.numel() for param in differentiable_parameters)} parameters")
    opt = torch.optim.Adam(differentiable_parameters, lr=args.lr, betas=(0.9, 0.95), amsgrad=True)


    assert args.batch_size // len(args.devices) == 0, "batch_size must be divisible by the number of GPUs"
    per_device_batch_size = args.batch_size // len(args.devices)
    batch_iterators = [
        iterate_minibatches(inps[i], outs[i], batch_size=per_device_batch_size)
        for i in range(len(args.devices))
    ]  # TODO maybe add asynchronous host-to-device copy here


    previous_best_loss = float('inf')  # for early stopping
    for epoch in range(args.max_epochs):
        for step in range(args.steps_per_epoch):
            if len(args.devices) == 1:
                loss = _compute_mse_on_batch(args.devices[0], layer, batch_iterators[0], **kwargs)
            else:
                loss = _compute_mse_parallel(args.devices, replicas, differentiable_parameters, substitution_tables, batch_iterators, kwargs_by_device)

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")
            if step == 0 and args.relative_mse_tolerance is not None:
                if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                    return layer  # early stopping; no updates after last epoch's beam search
                previous_best_loss = min(previous_best_loss, loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

        # TODO MAYBE RUN EVAL HERE?!
    return layer


def _compute_mse_on_batch(
        device: torch.device, layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], **kwargs) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    TODO docs
    """
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(device, non_blocking=True)  # TODO this should be prefetched
    inps_batch = inps_batch.to(device, non_blocking=True)  # TODO this should be prefetched; when prefetched, remove device arg frokm this function

    # TODO un-hardcode this
    if 'attention_mask' in kwargs:
        assert kwargs['attention_mask'].ndim == 4
        assert kwargs['attention_mask'].shape[0] == 1
        kwargs = dict(kwargs, attention_mask=kwargs['attention_mask'].tile(len(inps_batch), 1, 1, 1))

    outs_prediction, *_unused = layer(inps_batch, **kwargs)
    assert outs_prediction.shape == outs_batch.shape
    return F.mse_loss(outs_prediction, outs_batch)



def _compute_mse_parallel(devices: Sequence[torch.device],
                          replicas: Sequence[nn.Module],
                          parameters_to_replicate: nn.ParameterList,
                          substitution_tables: Sequence[List[Sequence[Tuple[nn.Module, str]]]],
                          batch_iterators: Sequence[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
                          kwargs_by_device: Sequence[Dict[str, Any]]
                          ) -> torch.Tensor:
    """Compute MSE in parallel over multiple GPUs, each GPU processes a portion of samples"""
    replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
    funcs_by_replica = [_compute_mse_on_batch for _ in replicas]
    inputs_by_replica = []
    for i in range(len(devices)):
        if i != 0:  # no overrides needed for master module
            for replacement_param, param_substitutions in zip(replicated_parameters[i], substitution_tables[i]):
                for (replica_submodule, attr_name) in param_substitutions:
                    replace_parameter_(replica_submodule, attr_name, replacement_param)
        inputs_by_replica.append((devices[i], replicas[i], batch_iterators[i]))
    mse_components = torch.nn.parallel.parallel_apply(
        funcs_by_replica, inputs_by_replica, kwargs_by_device, devices=devices)
    return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).mean()
