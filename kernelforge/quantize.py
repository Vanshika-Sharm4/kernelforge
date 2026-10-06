"""Symmetric per-output-channel int8 weight quantization (W8A16).

Weights have shape (in_features, out_features). One scale is stored per output
column, so a quantized matmul can apply the scale once, after accumulation:

    y[:, n] = (x @ q[:, n]) * scale[n]
"""

from __future__ import annotations

import torch


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (in, out) weight to int8 with one float32 scale per output column."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=0).clamp(min=1e-8) / 127.0  # (out,)
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct an approximate floating point weight."""
    return (q.float() * scale[None, :]).to(dtype)
