from typing import Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


def pairwise_cross_entropy(left: torch.Tensor, right: torch.Tensor, block_size: int = 4096):
    """
    Compute clip loss from a pair of equal-shape tensors, use block-wise optimization to save vRAM

    :param left: tensor[batch_size, vector_dim]
    :param right: tensor[batch_size, vector_dim]
    :param block_size: accumulate crossentropy in square [block_size, block_size] blocks
    :returns: loss = mean(xent_left_values / 2 + xent_right_values / 2)
    """
    return _PairwiseCrossEntropyFunction.apply(left, right, block_size)


class _PairwiseCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, left: torch.Tensor, right: torch.Tensor, block_size: int):
        loss, _, _, left_logsumexp, right_logsumexp = _pairwise_crossentropy_forward(left, right, block_size)
        ctx.save_for_backward(left, right, left_logsumexp, right_logsumexp)
        ctx._block_size = block_size
        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_loss: torch.Tensor):
        block_size = ctx._block_size
        left, right, left_logsumexp, right_logsumexp = ctx.saved_tensors
        grad_left, grad_right = _pairwise_crossentropy_backward(
            grad_loss, left, right, left_logsumexp, right_logsumexp, block_size)
        return grad_left, grad_right, None


def _pairwise_crossentropy_forward(
        left: torch.Tensor,
        right: torch.Tensor,
        block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute pairwise crossentropy one block at a time.

    :param left: tensor[batch_size, vector_dim]
    :param right: tensor[batch_size, vector_dim]
    :param block_size: accumulate crossentropy in square [block_size, block_size] blocks
    :returns: 5 tensors, each of shape [batch_size]:
      - loss = mean(xent_left_values / 2 + xent_right_values / 2)
      - xent_left_values - equivalent to cross_entropy(left@right.t(), torch.arange(batch_size), reduction=None)
      - xent_right_values - equivalent to cross_entropy(right@left.t(), torch.arange(batch_size), reduction=None)
      - left_logsumexp - equivalent to (left @ right.t()).logsumexp(1)
      - right_logsumexp - equivalent to (right @ left.t()).logsumexp(1)
    """
    assert left.ndim == right.ndim == 2 and left.shape == right.shape
    diag_dot_products = torch.bmm(left[..., None, :], right[..., :, None]).view(len(left))

    # left logsumexp[i] is defined as [ log(sum_j(  exp(left[i] @ right[j])  )) ]
    left_logsumexp = torch.full((len(left),), -float('inf'), dtype=torch.float32, device=left.device)
    right_logsumexp = torch.full((len(right),), -float('inf'), dtype=torch.float32, device=right.device)
    
    for block_start_left in range(0, len(left), block_size):
        left_block_ix = slice(block_start_left, block_start_left + block_size)

        for block_start_right in range(0, len(right), block_size):
            right_block_ix = slice(block_start_right, block_start_right + block_size)

            block_products = left[left_block_ix] @ right[right_block_ix].t()  # [block_size @ block_size]
            torch.logaddexp(
                left_logsumexp[left_block_ix],
                block_products.logsumexp(dim=1),
                out=left_logsumexp[left_block_ix]
            )

            torch.logaddexp(
                right_logsumexp[right_block_ix],
                block_products.logsumexp(dim=0),
                out=right_logsumexp[right_block_ix]
            )

    xent_left_values = left_logsumexp - diag_dot_products
    xent_right_values = right_logsumexp - diag_dot_products
    loss = (xent_left_values.mean() + xent_right_values.mean()) / 2
    return loss, xent_left_values, xent_right_values, left_logsumexp, right_logsumexp


def _pairwise_crossentropy_backward(
        grad_output: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
        left_logsumexp: torch.Tensor, right_logsumexp: torch.Tensor,
        block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backprop gradients w.r.t. mean pair-wise cross-entropy to left and right vectors

    :param grad_output: grad w.r.t. loss, where loss = (xent(left @ right.t(), diag) + xent(right @ left.t(), diag) / 2
    :param left: tensor[batch_size, vector_dim] of left-hand-side embeddings
    :param right: tensor[batch_size, vector_dim] of right-hand-side embeddings
    :param left_logsumexp: tensor[batch_size] containing (left @ right.t()).logsumexp(1)
    :param right_logsumexp: tensor[batch_size] containing (right @ left.t()).logsumexp(1)
    :returns: grad w.r.t. left and right, respectively
    """
    assert grad_output.ndim == 0, "grad w.r.t. loss must be scalar"

    # grad crossentropy wrt left = -right + [softmax(left@right.t(), dim=1)] @ right

    # step 1: initialize gradients with right/left tensors (first summand from above)
    grad_right = left.mul(-1 / len(left))
    grad_left = right.mul(-1 / len(left))

    for block_start_left in range(0, len(left), block_size):
        left_block_ix = slice(block_start_left, block_start_left + block_size)

        for block_start_right in range(0, len(right), block_size):
            right_block_ix = slice(block_start_right, block_start_right + block_size)

            block_products = left[left_block_ix] @ right[right_block_ix].t()  # [block_size @ block_size]
            block_probs_left = (block_products - left_logsumexp[left_block_ix, None]).exp_()

            # step 2A: accumulate one block of [softmax(left @ right.t(), dim=1) @ right]
            grad_left[left_block_ix].addmm_(block_probs_left, right[right_block_ix], alpha=0.5 / len(right))
            # ... and vice versa for grad w.r.t. right = -left + [softmax(left @ right.t()).t() @ left]
            grad_right[right_block_ix].addmm_(block_probs_left.t(), left[left_block_ix], alpha=0.5 / len(left))

            # step 2B: same as 2A, but for inverse right-to-left cross-entropy
            # WARNING: block_probs_left is no longer valid after this line
            block_probs_right = torch.sub(
                block_products, right_logsumexp[None, right_block_ix], out=block_probs_left
            ).exp_()

            grad_left[left_block_ix].addmm_(block_probs_right, right[right_block_ix], alpha=0.5 / len(right))
            grad_right[right_block_ix].addmm_(block_probs_right.t(), left[left_block_ix], alpha=0.5 / len(left))
    return grad_left, grad_right


def test_forward(block_size: int = 2):
    left = torch.randn(6, 30).div_(10)
    right = torch.randn(6, 30).div_(10)

    loss, xent_left_values, xent_right_values, left_logsumexp, right_logsumexp = \
        _pairwise_crossentropy_forward(left, right, block_size)

    assert torch.allclose((left @ right.t()).logsumexp(1), left_logsumexp)
    assert torch.allclose((left @ right.t()).logsumexp(0), right_logsumexp)

    assert torch.allclose(
        xent_left_values, F.cross_entropy(left @ right.t(), torch.arange(len(left)), reduction='none'))
    assert torch.allclose(
        xent_right_values, F.cross_entropy((left @ right.t()).t(), torch.arange(len(left)), reduction='none'))


def test_loss_backward(block_size: int = 3):
    left = torch.randn(6, 30).div_(10)
    right = torch.randn(6, 30).div_(10)

    loss, xent_left_values, xent_right_values, left_logsumexp, right_logsumexp = \
        _pairwise_crossentropy_forward(left, right, block_size)

    grad_left, grad_right = _pairwise_crossentropy_backward(
        torch.tensor(1.0), left, right, left_logsumexp, right_logsumexp, block_size
    )

    left_ = left.detach().requires_grad_(True)
    right_ = right.detach().requires_grad_(True)
    dots = (left_ @ right_.t())
    targets = torch.arange(len(left))

    loss_l2r = F.cross_entropy(dots, targets, reduction='none')
    loss_r2l = F.cross_entropy(dots.t(), targets)
    loss_ = (loss_l2r / 2 + loss_r2l / 2).mean()
    loss_.backward()

    assert torch.allclose(loss, loss_)
    assert torch.allclose(grad_left, left_.grad, atol=1e-6, rtol=0)
    assert torch.allclose(grad_right, right_.grad, atol=1e-6, rtol=0)
