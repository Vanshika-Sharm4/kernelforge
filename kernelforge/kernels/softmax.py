"""Fused (scale + causal mask + softmax) Triton kernel.

One program handles one row. The whole row is loaded once, scaled, masked,
reduced (max and sum) and normalized in registers, then written once. The
unfused PyTorch version launches separate kernels for scale, mask and softmax
and round-trips the full attention-score matrix through DRAM between them.

Limitation: a full row must fit in one block (n_cols <= 65536).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    T_q,
    offset,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    in_bounds = cols < n_cols

    x = tl.load(in_ptr + row * in_row_stride + cols, mask=in_bounds, other=-float("inf"))
    x = x.to(tl.float32) * scale

    if CAUSAL:
        # Row `row` is query position (row % T_q) of its (batch, head). With a KV
        # cache the queries are the LAST T_q positions, so the last visible key
        # column for this query is (q_idx + offset), where offset = T_k - T_q.
        q_idx = row % T_q
        x = tl.where(cols <= q_idx + offset, x, -float("inf"))

    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(
        out_ptr + row * out_row_stride + cols,
        y.to(out_ptr.dtype.element_ty),
        mask=in_bounds,
    )


def _launch(x2d: torch.Tensor, scale: float, causal: bool, t_q: int) -> torch.Tensor:
    n_rows, n_cols = x2d.shape
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError(f"row length {n_cols} too large for single-block softmax")
    num_warps = 4 if block <= 1024 else 8 if block <= 4096 else 16
    out = torch.empty_like(x2d)
    _softmax_kernel[(n_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        n_cols,
        t_q,
        n_cols - t_q,  # offset = T_k - T_q
        scale,
        CAUSAL=causal,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
    return out


def softmax(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Row-wise softmax(x * scale) over the last dimension."""
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    return _launch(x2d, scale, causal=False, t_q=1).view(shape)


def causal_softmax(scores: torch.Tensor, scale: float) -> torch.Tensor:
    """softmax(scores * scale) with a causal mask. scores: (..., T_q, T_k), T_k >= T_q."""
    *_, t_q, t_k = scores.shape
    if t_k < t_q:
        raise ValueError("expected T_k >= T_q")
    x2d = scores.reshape(-1, t_k).contiguous()
    return _launch(x2d, scale, causal=True, t_q=t_q).view(scores.shape)
