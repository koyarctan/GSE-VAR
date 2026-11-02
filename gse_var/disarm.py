"""Antithetic Bernoulli sampling and the unbiased DisARM logit estimator.

Dong, Mnih and Tucker (2020), equations (6)--(7).
The estimator operates on loss differences and has no model/optimizer dependency.
"""
import torch

@torch.no_grad()
def _antithetic_masks(logits: torch.Tensor, generator: torch.Generator):
    probabilities = logits.sigmoid()
    uniform = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device,
                         generator=generator)
    return uniform < probabilities, (1 - uniform) < probabilities


@torch.no_grad()
def _disarm_gradient(logits, first_mask, second_mask, first_loss, second_loss):
    """Equation (7), with scalar or broadcastable target-local loss differences."""
    sign = 1 - 2 * second_mask.to(logits.dtype)
    different = (first_mask != second_mask).to(logits.dtype)
    return 0.5 * (first_loss - second_loss) * sign * different * logits.abs().sigmoid()
