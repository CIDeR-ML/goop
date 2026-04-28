"""ADC digitization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DigitizationConfig:
    """ADC digitization parameters (post-convolution).

    Applied after convolution: adds a pedestal offset, rounds to integers,
    and clamps to the ADC bit range [0, 2^n_bits - 1].
    """

    n_bits: int = 14          # 14-bit ADC → [0, 16383]
    pedestal: float = 1500.0  # baseline offset in ADC counts


def digitize(
    data: torch.Tensor, pedestal: float, n_bits: int, *, ste: bool = False,
) -> torch.Tensor:
    """Add pedestal, round to integer, clamp to ADC range.

    Params
    ------
    data: torch.Tensor
        The input data to digitize.
    pedestal: float
        The pedestal offset in ADC counts.
    n_bits: int
        The number of bits for the ADC.
    ste: bool
        Whether to use the straight-through estimator (to keep gradient flow)

    Returns float32 tensor with integer-valued entries in ``[0, 2^n_bits - 1]``.
    """
    adc_max = (1 << n_bits) - 1
    x = data + pedestal
    x_q = x.round().clamp(0, adc_max)
    if ste:
        return x_q + (x - x.detach())
    return x_q
