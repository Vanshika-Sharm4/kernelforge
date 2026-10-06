"""Tiled Triton matmul with fused epilogue (bias, GELU) and optional int8 weights.

Computes  C = act(A @ B + bias)  for A (M, K) and B (K, N).

Techniques
----------
* Tiling: each program computes a BLOCK_M x BLOCK_N tile of C, looping over K in
  BLOCK_K steps. Tiles are staged through shared memory/registers by Triton and
  multiplied with tensor cores via ``tl.dot`` (float32 accumulation).
* Grouped program ordering: programs are launched in groups of GROUP_M row-tiles
  so neighbouring programs reuse the same B tiles, improving L2 cache hit rate.
* Autotuning: several (block shape, num_warps, num_stages) configs are
  benchmarked once per (M, N, K) and the fastest is cached.
* Fused epilogue: bias add and tanh-GELU are applied to the float32 accumulator
  before the single store, saving one or two full read/write passes over C.
* W8A16: with ``QUANT`` set, B is int8 with one float scale per output column.
  Weights are converted to the activation dtype in registers, so DRAM traffic for
  weights is halved versus fp16 (this is what speeds up memory-bound decode).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ACTIVATIONS = {None: 0, "gelu": 1}


def _configs() -> list[triton.Config]:
    if os.environ.get("TRITON_INTERPRET") == "1":  # keep CPU interpreter tests fast
        return [triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=2, num_stages=2)]
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
        # Small-M tiles for decode (M = batch size, often 1..16).
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    QUANT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- map the 1D program id to a (pid_m, pid_n) output tile, grouped for L2 reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ---- main loop over K, accumulating in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        if QUANT:
            b = b.to(a.dtype)  # int8 -> activation dtype (exact), dequantized in registers
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- fused epilogue: per-column scale, bias, activation
    if QUANT:
        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
        acc = acc * scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    if ACTIVATION == 1:
        # tanh-approximated GELU ("gelu_new", used by GPT-2); tanh(z) = 2*sigmoid(2z) - 1
        inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _launch(a, b, bias, activation, scale):
    if activation not in _ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    lead, k_dim = a.shape[:-1], a.shape[-1]
    a2 = a.reshape(-1, k_dim)
    m, k = a2.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"shape mismatch: A is (*, {k}) but B is ({k2}, {n})")
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _matmul_kernel[grid](
        a2,
        b,
        c,
        bias if bias is not None else c,  # dummy pointers when unused
        scale if scale is not None else c,
        m,
        n,
        k,
        a2.stride(0),
        a2.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        ACTIVATION=_ACTIVATIONS[activation],
        QUANT=scale is not None,
    )
    return c.view(*lead, n)


def matmul(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None, activation: str | None = None) -> torch.Tensor:
    """act(a @ b + bias). a: (..., K), b: (K, N) any strides, bias: (N,)."""
    return _launch(a, b, bias, activation, scale=None)


def matmul_w8(
    a: torch.Tensor,
    qb: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """act((a @ dequant(qb)) + bias) with int8 weights qb (K, N) and per-column scale (N,)."""
    if qb.dtype != torch.int8:
        raise ValueError("qb must be int8")
    return _launch(a, qb, bias, activation, scale=scale)
