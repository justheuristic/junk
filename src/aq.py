import functools
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange


class QuantizedWeight(nn.Module):
    def __init__(self, *,
                 weight_shape: Tuple[int, int],
                 in_group_size: int,
                 out_group_size: int,
                 num_codebooks: int,
                 nbits_per_codebook: int = 8,
                 init_kmeans: bool = False,
                 init_mean: float = 0.0,
                 init_std: float = 1.0,
                 device: torch.device = torch.device('cpu'),
                 **init_kwargs
                 ):
        super().__init__()
        out_features, in_features = weight_shape
        assert in_features % in_group_size == 0
        assert out_features % out_group_size == 0

        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = codebook_size = 2 ** nbits_per_codebook
        
        if init_kmeans:
            assert 'reference_weight' in init_kwargs, "please provide reference_weight"
            codes, codebooks = init_aq_kmeans(
                num_codebooks=num_codebooks, out_group_size=out_group_size, in_group_size=in_group_size,
                codebook_size=self.codebook_size, **init_kwargs)
        else:
            codebooks = torch.randn(
                num_codebooks, codebook_size, out_group_size, in_group_size, device=device
                ).mul_(init_std).add_(init_mean)
            codes = torch.randint(low=0, high=codebook_size,
                                  size=(out_features // out_group_size, in_features // in_group_size, num_codebooks),
                                  device=device)

        self.codebooks = nn.Parameter(codebooks, requires_grad=True)
        self.codes = nn.Parameter(codes, requires_grad=False)

    def forward(self, input):
        """Multiply :input: by the quantized weight matrix"""
        return F.linear(input, _reconstruct_weight(self.codes, self.codebooks))

    @torch.no_grad()
    def requantize_(self, XTX: torch.Tensor, reference_weight: torch.Tensor, *,
                    beam_size: int, sparsity_regularizer: float, verbose: bool):
        """
        Update self.codes in-place via beam search so as to minimize squared errors
        :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
        :note: if XTX is divided by dataset size, this function will return *mean* squared error
        :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
        :param beam_size: consider up to this many best encoding combinations
        :param sparsity_regularizer: subtract this value from beam search objective each time you have a zero code somewhere
        :param verbose: if True, draw a progressbar and periodically print best loss

        """
        self.codes[...] = beam_search_optimal_codes(
            XTX=XTX, reference_weight=reference_weight, codebooks=self.codebooks, prev_codes=self.codes,
            beam_size=beam_size, sparsity_regularizer=sparsity_regularizer, verbose=verbose
        )


@torch.inference_mode()
def beam_search_optimal_codes(
        XTX: torch.Tensor,
        reference_weight: torch.Tensor,
        codebooks: torch.Tensor,
        prev_codes: torch.IntTensor,
        *,
        beam_size: int,
        sparsity_regularizer: float = 0,
        verbose: bool,
):
    """
    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if XTX is divided by dataset size, this function will return *mean* squared error
    :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_siz, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_out_groups, num_in_groups, num_codebooks]
    :param beam_size: consider up to this many best encoding combinations
    :param sparsity_regularizer: subtract this value from beam search objective each time you have a zero code somewhere
    :param verbose: if True, draw a progressbar and periodically print best loss
    :return: best quantization codes found, same shape as prev_codes

    :intuition: the beam search needs to produce weight codes that minimize MSE error
    - the codes are of shape [out_features / out_group_size, in_features / in_group_size, num_codebooks]

    Out of those three dimensions, out_features is "independent", i.e. changing code in
    one output feature does not increase the MSE error for another feature. Therefore,
    beam search for different output features can run in independently in parallel.

    Neither (in_features / in_group_size) nor (num_codebooks) dimension are independent:
    - changing the encoding for one feature can compensate the error from encoding another, OBC-style
    - for a single weight group, changing code in one codebook can affect the optimal choice in another codebook
    Therefore, beam search must go in a double loop over (in_features/in_group_size) and (num_codebooks) dimensions

    This leaves one choice: which dimension used for outer loop, and which one goes is in the inner loop?
    Due to the nature of beam search, interactions between dimensions of inner loop will be explored better.
    We chose to use (in_features/in_group_size) in the outer loop and (num_codebooks) for the inner loop.
    This is based on an intuition from GPTQ: you can get decent performance by quantizing each input unit ...
    ... greedily --- GPTQ does not change quantizations for previously quantized features and works fine.
    Therefore, we believe that we can also use a greedy approach to compensate error between input features.
    In turn, we believe that the codes used to encode the same weights (additively) are more inter-dependent.
    This should be treated as an educated guess with no proof and no ablation (as of the time of writing).

    """
    num_out_groups, num_in_groups, num_codebooks = prev_codes.shape
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = num_in_groups * in_group_size
    out_features = num_out_groups * out_group_size
    assert reference_weight.shape == (out_features, in_features)
    prev_weight = _reconstruct_weight(prev_codes, codebooks)

    # initialize all beam codes as previous codes - so they can be updated during beam search
    beam_codes = prev_codes.unsqueeze(0)
    # beam_codes shape: [current beam_size, num_out_groups, num_in_groups, num_codebooks], initial beam_size = 1
    beam_weights = prev_weight.unsqueeze(0)
    # beam_weights shape: [current beam_size, out_features, in_features], initial beam size = 1

    beam_losses = _channelwise_squared_error(XTX, prev_weight, reference_weight
                                             ).reshape(1, num_out_groups, out_group_size).sum(-1)
    # beam_losses shape: [current beam_size, num_out_groups], initial beam_size = 1
    if sparsity_regularizer != 0:
        beam_losses = beam_losses - sparsity_regularizer * (prev_codes == 0).sum(dim=(-1, -2))[None, :]

    if verbose:
        progressbar = trange(num_in_groups * num_codebooks)
    for input_group_index in range(num_in_groups):
        for codebook_index in range(num_codebooks):
            ### part 1: compute losses for every possible candidate for one given codebook and input group.
            # Currently, we compute errors for all output features in parallel in a vectorized fashion.
            candidate_losses = _beam_search_squared_errors(
                XTX=XTX, reference_weight=reference_weight, codebooks=codebooks,
                beam_losses=beam_losses, beam_codes=beam_codes, beam_weights=beam_weights,
                input_group_index=input_group_index, codebook_index=codebook_index,
                sparsity_regularizer=sparsity_regularizer
            )  # [current beam_size, codebook_size, num_out_groups]

            # part 2: select beam_size new best codes and re-arrange beam to account for the fact that ...
            # ... sometimes two or more top candidates originate from the same source in previous beam
            beam_codes, beam_weights, beam_losses = _beam_search_select_best(
                beam_codes, beam_weights, codebooks,
                input_group_index, codebook_index, candidate_losses,
                beam_size
            )

            if verbose:
                progressbar.update()
                if (input_group_index * num_codebooks + codebook_index) % verbose != 0:
                    continue  # if update is an integer, compute metrics every (this many) beam search steps
                best_loss = beam_losses.min(0).values.sum().item() / out_features
                info = f"in_group {input_group_index} / {num_in_groups} "
                info += f"| codebook {codebook_index} / {num_codebooks} "
                if sparsity_regularizer == 0:
                    info += f"| loss {best_loss:.10f}"
                else:  # un-regularize to restore MSE loss, report sparsity rate
                    num_zero_codes = (beam_codes[0] == 0).sum().item()
                    best_loss = best_loss + sparsity_regularizer / out_features * num_zero_codes
                    sparsity = num_zero_codes / prev_codes.numel()
                    info += f"| loss {best_loss:.5f} | sparse {sparsity * 100:.1f}% |"

                progressbar.desc = info
    return beam_codes[0]


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def _reconstruct_weight(codes: torch.IntTensor, codebooks: torch.Tensor) -> torch.FloatTensor:
    """
    Decode float weights from quantization codes. Differentiable.
    :param codes: tensor of integer quantization codes, shape [*dims, num_out_groups, num_in_groups, num_codebooks]
    :param codebooks: tensor of vectors for each quantization code, [num_codebooks, codebook_size, out_group_size, in_group_size]
    :return: reconstructed weight tensor of shape [*dims, num_in_groups*group_size]
    """
    num_out_groups, num_in_groups, num_codebooks = codes.shape[-3:]
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=codes.device
    )  # shape: [num_codebooks]
    reconstructed_weight_flat = F.embedding_bag(
        codes.flatten(0, -2) + codebook_offsets,
        codebooks.flatten(0, 1).flatten(-2, -1),
        mode='sum'
    )  # [prod(dims) * num_out_groups * num_in_groups, out_group_size * in_group_size]

    reconstructed_weight_groupwise = reconstructed_weight_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    return reconstructed_weight_groupwise.swapaxes(-3, -2).reshape(list(codes.shape[:-3]) + [out_features, in_features])


@maybe_script
def _beam_search_squared_errors(
        XTX: torch.Tensor, reference_weight: torch.Tensor, codebooks: torch.Tensor,
        beam_losses: torch.Tensor, beam_codes: torch.Tensor, beam_weights: torch.Tensor,
        input_group_index: int, codebook_index: int, sparsity_regularizer: float,
) -> torch.Tensor:
    """
    Compute MSE or sum-of-squared-error losses for all possible ways to replace quantization codes for one input group
     and one codebook. Works in parallel for all output-dimension groups.

    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if both XTX *and* beam_loses are divided by dataset size, this function will return mean squared error
    :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param beam_losses: sum-of-squared-error for each hypothesis in beam and for each output channel;
        shape: [beam_size, num_out_groups]
    :param beam_codes: a tensor with best weight codes, shape: [beam_size, num_out_groups, num_in_groups, num_codebooks]
    :param beam_weights: a tensor with de-quantized beam_codes, shape: [beam_size, out_features, in_features]
    :param input_group_index: an index of one group of in_features that is being re-encoded
    :param codebook_index: an index of one codebook for that group of features that is being re-encoded
    :return: 3d tensor of losses, shape = [beam_size, codebook_size, num_out_groups]
    """
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    out_features = num_out_groups * out_group_size

    input_group_slice = slice(input_group_index * in_group_size, (input_group_index + 1) * in_group_size)

    prev_codes_part = beam_codes[:, :, input_group_index, codebook_index]  # [beam_size, num_out_groups]

    prev_weight_part = F.embedding(
        prev_codes_part, codebooks[codebook_index].flatten(-2, -1)
    ).view(beam_size, out_features, in_group_size)  # previous codes de-quantized

    cand_weights = codebooks[codebook_index]  # [codebook_size, out_group_size, in_group_size], all replacement codes

    # Step 1: compute flat dot products between X @ (W_original - W_quantized_without_replaced_codes) and
    # ... and (X @ Ci) where Ci are all possible replacement codes
    delta_weight_without_part = reference_weight - beam_weights
    delta_weight_without_part[:, :, input_group_slice] += prev_weight_part

    delta_weights = torch.sub(cand_weights[None, :, None, :, :],
                              prev_weight_part.view(beam_size, 1, num_out_groups, out_group_size, in_group_size)
                              ).reshape(beam_size, codebook_size, out_features, in_group_size)
    # dWTXTX is equivalent to < X @ (W - \sum BiCi except current codebook), X @ SOMETHING >
    dWTXTXg = delta_weight_without_part @ XTX[..., input_group_slice]  # [beam_size, out_features, in_group_size]
    # below: use torch.matmul to compute broadcasted batch matrix multiplication; see matmul docs

    dot_products = torch.matmul(
        dWTXTXg.view(beam_size, 1, num_out_groups, 1, out_group_size * in_group_size),
        delta_weights.view(beam_size, codebook_size, num_out_groups, out_group_size * in_group_size, 1),
    ).reshape(beam_size, codebook_size, num_out_groups)

    # step 2: compute
    XoldBkC_norms_sq = torch.bmm(
        (prev_weight_part @ XTX[input_group_slice, input_group_slice]).view(
            beam_size * num_out_groups, 1, out_group_size * in_group_size),
        prev_weight_part.view(beam_size * num_out_groups, out_group_size * in_group_size, 1)
    ).reshape(beam_size, 1, num_out_groups)

    XnewBkC_norms_sq = torch.bmm(
        (cand_weights.flatten(0, 1) @ XTX[input_group_slice, input_group_slice]).view(
            codebook_size, 1, out_group_size * in_group_size),
        cand_weights.view(codebook_size, out_group_size * in_group_size, 1)
    ).reshape(1, codebook_size, 1)

    candidate_squared_errors = (
            beam_losses[:, None, :] - 2 * dot_products + XnewBkC_norms_sq - XoldBkC_norms_sq
    )  # shape: [beam_size, codebook_size, num_out_groups]

    if sparsity_regularizer != 0:
        candidate_squared_errors += sparsity_regularizer * (prev_codes_part == 0).to(XTX.dtype)[:, None, :]
        candidate_squared_errors[:, 0, :] -= sparsity_regularizer

    return candidate_squared_errors


@maybe_script
def _beam_search_select_best(
        beam_codes: torch.Tensor, beam_weights: torch.Tensor, codebooks: torch.Tensor,
        input_group_index: int, codebook_index: int, candidate_losses: torch.Tensor,
        beam_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select top-:beam_size: and reorder beam accordingly, return new beam
    :param beam_codes: a tensor with best weight codes, shape: [beam_size, num_out_groups, num_in_groups, num_codebooks]
    :param beam_weights: a tensor with de-quantized beam_codes, shape: [beam_size, out_features, in_features]
    :param codebooks: a tensor with look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param input_group_index: an index of one group of in_features that is being re-encoded
    :param codebook_index: an index of one codebook for that group of features that is being re-encoded
    :param candidate_losses: a 3d tensor of losses, shape = [beam_size, codebook_size, nun_out_groups]
    :param beam_size: how many top hypotheses should be selected

    :returns: new (beam_codes, beam_weights, beam_losses)
    """
    dtype = candidate_losses.dtype
    device = candidate_losses.device
    _prev_beam_size, codebook_size, num_out_groups = candidate_losses.shape
    _prev_beam_size, out_features, in_features = beam_weights.shape
    _prev_beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    flat_best = candidate_losses.flatten(0, 1).topk(dim=0, k=beam_size, largest=False)
    best_hypo_source_ids = flat_best.indices // codebook_size
    best_hypo_codes = flat_best.indices % codebook_size
    # ^-- shape: [beam_size, out_features]

    # reorder beam codes and weights
    new_beam_codes = torch.full(
        size=(len(best_hypo_codes), num_out_groups, num_in_groups, num_codebooks), fill_value=-1, dtype=beam_codes.dtype,
        device=device
    )  # [beam_size, num_out_groups, num_in_groups, num_codebooks]
    new_beam_weights = torch.empty(len(best_hypo_codes), out_features, in_features, dtype=dtype, device=device)
    arange_out_groups = torch.arange(num_out_groups, device=device)

    for beam_index in range(len(best_hypo_codes)):
        new_beam_codes[beam_index, :, ...] = \
            beam_codes[best_hypo_source_ids[beam_index, :], arange_out_groups, ...]
        new_beam_codes[beam_index, :, input_group_index, codebook_index] = \
            best_hypo_codes[beam_index, :]
        new_beam_weights[beam_index, :, :] = _reconstruct_weight(new_beam_codes[beam_index, ...], codebooks)

    # Note: the code above can be further accelerated by 1) vectorzing loop and ...
    # ... 2) updating new_beam_weights only for the chosen input group
    return new_beam_codes, new_beam_weights, flat_best.values


@maybe_script
def _channelwise_squared_error(XTX: torch.Tensor, weight: torch.Tensor, reference_weight: torch.Tensor):
    """
    Compute per-channel squared error between X @ weight_or_weights and X @ reference_weight
    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if XTX is divided by dataset size, this function will return *mean* squared error
    :param weight: predicted/reconstructed weights of shape [*dims, out_features, in_features]
    :param reference_weight: reference weight of shape [out_features, in_features]
    :return: per-channel squared errors of shape [*dims, out_features]
    """
    XW_norm_square = torch.matmul(weight[..., :, None, :], (weight @ XTX)[..., :, :, None]).flatten(-3)
    XWreference_norm_square = torch.bmm(reference_weight[:, None, :], (reference_weight @ XTX)[:, :, None]).flatten(-3)
    dot_product = torch.matmul((reference_weight @ XTX)[:, None, :], weight[..., :, :, None]).flatten(-3)
    return XW_norm_square - 2 * dot_product + XWreference_norm_square


@torch.no_grad()
def init_aq_kmeans(reference_weight: torch.Tensor, *,
                   num_codebooks: int, out_group_size: int, in_group_size: int, codebook_size: int, alpha: float = 1.0,
                   verbose: bool = False, **kwargs):
    """
    Create initial codes and codebooks using residual K-means clustering of weights
    :params reference_weight, num_codebooks, out_group_size, in_group_size, nbits, verbose: same as in QuantizedWeght
    :param alpha: shrinkage for residual quantization, typically in (0, 1] semi-interval
    :param kwargs: any additional params are forwarded to fit_kmeans
    """
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size
    weight_residue = reference_weight.reshape(
        num_out_groups, out_group_size, num_in_groups, in_group_size
    ).clone().swapaxes(-3, -2).reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    codebooks = []
    codes = []

    for _ in trange(num_codebooks, desc='initializing with kmeans') if verbose else range(num_codebooks):
        codebook_i, codes_i, reconstructed_weight_i = fit_kmeans(weight_residue, k=codebook_size, **kwargs)
        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = alpha * codebook_i.reshape(1, codebook_size, out_group_size, in_group_size)
        weight_residue -= reconstructed_weight_i * alpha
        codes.append(codes_i)
        codebooks.append(codebook_i)
        del reconstructed_weight_i
    codebooks = torch.cat(codebooks, dim=0)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks


@maybe_script
def kmeans_greedy_init(data: torch.Tensor, k: int) -> torch.Tensor:
    """Get initial clusters by iteratively choosing a vector that is the farthest from already selected clusters"""
    clusters = torch.zeros(k, data.shape[1], device=data.device)
    running_min_distances = torch.full((data.shape[0],), torch.inf, device=data.device, dtype=data.dtype)
    data_norm_squared = data.norm(p=2, dim=1).square()

    for i in range(k):
        clusters[i] = data[running_min_distances.argmax()]
        distances_to_cluster_i = data_norm_squared - 2 * data @ clusters[i] + clusters[i].norm().square()
        running_min_distances = torch.minimum(
            running_min_distances, distances_to_cluster_i, out=running_min_distances)
    return clusters


@maybe_script
def fit_kmeans(data: torch.Tensor, k: int, max_iter: int = 1000, check_every: int = 10,
               rtol: float = 1e-06, atol: float = 1e-08, random_init: bool = False):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param check_every: check for convergence (allclose(new_centroids, old_centroids)) once in this many steps
    :param rtol: early stopping relative tolerance for centroids
    :param atol: early stopping absolute tolerance for centroids
    :param random_init: if True, initialize with random points using pytorch global RNG
        if False (default), init by greedily selecting the point that is farthest from any cluster
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    if random_init:
        clusters = data[torch.randperm(data.shape[0])[:k], :]  # [k, dim]
    else:
        clusters = kmeans_greedy_init(data, k)

    for i in range(max_iter):
        nearest_indices = torch.addmm(
            torch.bmm(clusters[:, None, :], clusters[:, :, None]).flatten(),
            data, clusters.T, beta=-0.5
        ).argmax(1)
        # note: the above formula equals to - 0.5 || data[:, None, :] - clusters[None, :, :] || ^ 2 + const

        new_clusters = torch.zeros_like(clusters).index_reduce_(
            dim=0, index=nearest_indices, source=data, reduce='mean', include_self=False)

        if i % check_every == 0:
            if torch.allclose(new_clusters, clusters, rtol=rtol, atol=atol):
                break
        clusters = new_clusters
    nearest_indices = torch.addmm(
        torch.bmm(clusters[:, None, :], clusters[:, :, None]).flatten(),
        data, clusters.T, beta=-0.5
    ).argmax(1)
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data


@maybe_script
def find_nearest_cluster(data, clusters):
    """Find nearest clusters and return their indices"""
    return torch.addmm(
        torch.bmm(clusters[:, None, :], clusters[:, :, None]).flatten(),  data, clusters.T, beta=-0.5
    ).argmax(1)
