import itertools
from typing import Optional

import torch


def fit_kmeans_1d_to_weights(
        data: torch.Tensor, *, k: int, groupsize_dim0: Optional[int] = None, groupsize_dim1: Optional[int] = None,
        **kwargs
):
    """
    :param data: weight matrix or matrices to be compressed, shape: [dim0, dim1, item_dim]
    :param k: number of clusters, typically 2 ** bitrate
    :param groupsize_dim0: this many elements across dim 0 are compressed together, default = all
    :param groupsize_dim1: this many elements across dim 1 are compressed together, default = all
    :param max_iter: run for at most this many kmeans iterations (-1 = run until convergence)
    :param offset_rate: if greater than 0, skip this percentage of smallest/largest elements for initialization
    :param verbose: print mse and early stopping info
    :param rtol: stop if next clusters are allclose to previous iteration's clusters within this relative tolerance
    :param atol: same as rtol, but absolute tolerance
    :returns: (clusters, indices, restored_data)
        - clusters are centroids of shape [dim0 // groupsize_dim0, dim1 // groupsize_dim1, item_dim, num_clusters]
        - indices are integers [0, k) in the same shape as data; they denote the index of the nearest centroid
        - restored_data is a floating point tensor in the same shape as data; they are dequantized(quantized(data))
    """
    if data.ndim == 2:
        clusters, indices, restored_data = fit_kmeans_1d_to_weights(
            data.unsqueeze(-1), k=k, groupsize_dim0=groupsize_dim0, groupsize_dim1=groupsize_dim1, **kwargs)
        return clusters.squeeze(-2), indices.squeeze(-1), restored_data.squeeze(-1)
    assert data.ndim == 3
    if groupsize_dim0 is None:
        groupsize_dim0 = data.shape[0]
    if groupsize_dim1 is None:
        groupsize_dim1 = data.shape[1]
    assert k <= (groupsize_dim1 * groupsize_dim0), "there are more clusters than datapoints; no point in clustering"

    dim0, dim1, item_dim = data.shape

    # reshape data as [num_groups * item_dim, group_size]
    groupwise_data = data.reshape(
        dim0 // groupsize_dim0, groupsize_dim0, dim1 // groupsize_dim1, groupsize_dim1, item_dim
    ).permute(0, 2, 4, 1, 3).flatten(3, 4).flatten(0, 2)
    # groupwise_data shape: [num_groups_dim0 * num_groups_dim1 * item_dim, groupsize_dim0 * groupsize_dim1]

    # apply the group-wise kmeans
    clusters, groupwise_cluster_indices, groupwise_restored_data = fit_kmeans_1d(groupwise_data, k=k, **kwargs)

    # undo reshapes from step1
    group_shape = ((dim0 // groupsize_dim0), (dim1 // groupsize_dim1), item_dim, groupsize_dim0, groupsize_dim1)
    restored_data = groupwise_restored_data.reshape(*group_shape).permute(0, 3, 1, 4, 2).reshape_as(data)
    reshaped_indices = groupwise_cluster_indices.reshape(*group_shape).permute(0, 3, 1, 4, 2).reshape_as(data)
    reshaped_clusters = clusters.reshape((dim0 // groupsize_dim0), (dim1 // groupsize_dim1), item_dim, k)
    return reshaped_clusters, reshaped_indices, restored_data


def get_leave_one_out_kmeans_1d(
        data: torch.Tensor, *, k: int, groupsize_dim0: Optional[int] = None, groupsize_dim1: Optional[int] = None,
        **kwargs
):
    """Fit 1-dimensional KMEANS excluding each one datapoint at a time, return the cluster that is nearest to the original datapoint"""
    assert data.ndim == 3
    if groupsize_dim0 is None:
        groupsize_dim0 = data.shape[0]
    if groupsize_dim1 is None:
        groupsize_dim1 = data.shape[1]
    assert k <= (groupsize_dim1 * groupsize_dim0), "there are more clusters than datapoints; no point in clustering"

    dim0, dim1, item_dim = data.shape

    # reshape data as [num_groups * item_dim, group_size]
    groupwise_data = data.reshape(
        dim0 // groupsize_dim0, groupsize_dim0, dim1 // groupsize_dim1, groupsize_dim1, item_dim
    ).permute(0, 2, 4, 1, 3).flatten(3, 4).flatten(0, 2)
    # groupwise_data shape: [num_groups_dim0 * num_groups_dim1 * item_dim, groupsize_dim0 * groupsize_dim1]

    loo_indices = torch.arange(groupwise_data.shape[1], device=groupwise_data.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = groupwise_data[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]

    # apply the group-wise kmeans
    sorted_clusters, _, _ = fit_kmeans_1d(groupwise_loo_data.flatten(0, 1), k=k, **kwargs)
    borders = (sorted_clusters[:, 1:] + sorted_clusters[:, :-1]) / 2
    groupwise_restored_data = sorted_clusters.gather(
        1, torch.searchsorted(borders, groupwise_data.flatten().unsqueeze(-1), side='left'
                              )).reshape_as(groupwise_data)

    group_shape = ((dim0 // groupsize_dim0), (dim1 // groupsize_dim1), item_dim, groupsize_dim0, groupsize_dim1)
    restored_data = groupwise_restored_data.reshape(*group_shape).permute(0, 3, 1, 4, 2).reshape_as(data)
    return restored_data


def fit_kmeans_1d(
        groupwise_data: torch.Tensor, k: int, max_iter: int = -1, offset_rate: float = 0, verbose: bool = False,
        initial_clusters: Optional[torch.Tensor] = None,
        **kwargs
):
    """
    :param groupwise_data: stuff to be compressed, shape: [num_groups, group_size]
    :param k: the number of centroids to find
    :param max_iter: run for at most this many kmeans iterations (-1 = run until convergence)
    :param offset_rate: if greater than 0, skip this percentage of smallest/largest elements for initialization
    :param verbose: print mse and early stopping info
    :param kwargs: optionally provide rtol=... and atol=... for early stopping;
    :note: if rtol/atol is speficied, these tolerances are measured between cluster centroids from subsequent steps
    :returns: (clusters, indices, restored_data)
        - clusters are centroids of shape
        - indices are integers [0, k) in the same shape as data; they denote the index of the nearest centroid
        - restored_data is a floating point tensor in the same shape as data; they are dequantized(quantized(data))

    :TODO: torch.jit.script / torch.compile
    """
    assert groupwise_data.ndim == 2
    assert 0 <= offset_rate < 0.5

    # step 2: pre-sort data and initialize kmeans with uniform percentiles
    sorted_data, groupwise_sort_indices = groupwise_data.sort(dim=1)
    groupwise_ranks_1based = groupwise_sort_indices.argsort(-1).add_(1)
    del groupwise_sort_indices

    # ^-- [num_groups, group_size]; sorted by group_size
    sorted_cumsum = torch.cat([
        torch.zeros_like(sorted_data[:, :1]), sorted_data.cumsum(dim=1)], dim=1)
    # ^-- [num_groups, group_size + 1]; sorted by group_size + 1
    if initial_clusters is not None:
        clusters = initial_clusters
    else:
        offset = int((sorted_data.shape[1] - 1) * offset_rate)
        init_indices = torch.linspace(offset, sorted_data.shape[1] - 1 - offset, k, dtype=torch.int64)
        clusters = sorted_data[:, init_indices]  # shape: [num_groups, k]

    # step 3: run kmeans
    def _groupwise_find_border_indices(clusters, sorted_data):
        borders = (clusters[:, 1:] + clusters[:, :-1]) / 2
        column = clusters[:, :1]
        borders = torch.cat([
            torch.full_like(column, float('-inf')), borders, torch.full_like(column, float('inf'))
        ], dim=1)
        border_indices = torch.searchsorted(sorted_data, borders, side='left')
        return border_indices

    for i in itertools.count():
        border_indices = _groupwise_find_border_indices(clusters, sorted_data)
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1)
        new_cluster_centers = torch.where(
            count_by_cluster > 0,
            sum_by_cluster / count_by_cluster,
            sorted_data.gather(1, border_indices[:, :-1].clamp_max(sorted_data.shape[1] - 1))
        )
        if torch.allclose(new_cluster_centers, clusters, **kwargs):
            if verbose:
                print(f"Early stopping after {i} iterations")
            break
        clusters = new_cluster_centers
        if max_iter > 0 and i >= max_iter:
            break

    # step 4: determine the final clustering
    border_indices = _groupwise_find_border_indices(clusters, sorted_data)
    groupwise_cluster_indices = torch.searchsorted(border_indices[:, 1:], groupwise_ranks_1based, side='left')
    groupwise_restored_data = clusters.gather(1, groupwise_cluster_indices)
    # [num_groups, k]

    if verbose:
        sorted_cumsum_squares = torch.cat([
            torch.zeros_like(sorted_data[:, :1]), sorted_data.square().cumsum(dim=1)], dim=1)
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        sum_squares_by_cluster = torch.diff(sorted_cumsum_squares.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1).clamp_min(1)
        mse_l2 = (groupwise_restored_data - groupwise_data).square().mean()
        mse_approx = (sum_squares_by_cluster - 2 * sum_by_cluster * clusters + count_by_cluster * clusters.square())
        mse_approx = mse_approx.sum(0) / count_by_cluster.sum(0)
        print(f'mse: {mse_l2.mean().item()} , dot-based estimate: {mse_approx.mean().item()}')

    return clusters, groupwise_cluster_indices, groupwise_restored_data


def quantize_weight_kmeans_1d(weight: torch.Tensor, sorted_clusters: torch.Tensor) -> torch.IntTensor:
    """Find the nearest cluster to a given datapoint and return its index"""
    assert weight.ndim == 2 and sorted_clusters.ndim == 3, (weight.shape, sorted_clusters.shape)
    assert weight.shape[0] % sorted_clusters.shape[0] == 0
    assert weight.shape[1] % sorted_clusters.shape[1] == 0
    num_groups_dim0, num_groups_dim1 = sorted_clusters.shape[:2]
    groupsize_dim0 = weight.shape[0] // num_groups_dim0
    groupsize_dim1 = weight.shape[1] // num_groups_dim1
    groupwise_weight = weight.reshape(
        num_groups_dim0, groupsize_dim0, num_groups_dim1, groupsize_dim1
    ).permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1)

    sorted_clusters_flat = sorted_clusters.flatten(0, 1)
    borders = (sorted_clusters_flat[:, 1:] + sorted_clusters_flat[:, :-1]) / 2
    groupwise_indices = torch.searchsorted(borders, groupwise_weight, side='left')
    indices = groupwise_indices.reshape(
        num_groups_dim0, num_groups_dim1, groupsize_dim0, groupsize_dim1
    ).permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1)
    return indices


def dequantize_weight_kmeans_1d(indices: torch.IntTensor, sorted_clusters: torch.FloatTensor) -> torch.FloatTensor:
    """Reconstruct floating point weights from quantization indices and sorted 1d clusters"""
    assert indices.ndim == 2 and sorted_clusters.ndim == 3, (indices.shape, sorted_clusters.shape)
    assert indices.shape[0] % sorted_clusters.shape[0] == 0
    assert indices.shape[1] % sorted_clusters.shape[1] == 0
    num_groups_dim0, num_groups_dim1 = sorted_clusters.shape[:2]
    groupsize_dim0 = indices.shape[0] // num_groups_dim0
    groupsize_dim1 = indices.shape[1] // num_groups_dim1
    groupwise_indices = indices.reshape(
        num_groups_dim0, groupsize_dim0, num_groups_dim1, groupsize_dim1
    ).permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1)

    groupwise_weights = sorted_clusters.gather(2, groupwise_indices.unsqueeze(1).long())

    reconstructed_weights = groupwise_weights.reshape(
        num_groups_dim0, num_groups_dim1, groupsize_dim0, groupsize_dim1
    ).permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1)
    return reconstructed_weights

