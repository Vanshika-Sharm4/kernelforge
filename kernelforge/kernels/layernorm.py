"""Fused LayerNorm Triton kernel (forward only).

One program per row: load the row once, compute mean and variance in float32,
normalize, apply the affine transform, and store. Mean/variance accumulate in
float32 even for float16 inputs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    row_stride,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * w + b

    tl.store(y_ptr + row * row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    n_rows, n_cols = x2d.shape
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError(f"row length {n_cols} too large for single-block layernorm")
    num_warps = 4 if block <= 1024 else 8 if block <= 4096 else 16
    y = torch.empty_like(x2d)
    _layernorm_kernel[(n_rows,)](
        x2d, weight, bias, y, x2d.stride(0), n_cols, eps, BLOCK_N=block, num_warps=num_warps
    )
    return y.view(shape)
