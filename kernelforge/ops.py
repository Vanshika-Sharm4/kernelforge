"""Backend dispatch. Every compute op the model uses goes through this module.

Two backends are available:
  * "torch"  : plain PyTorch ops (the baseline)
  * "triton" : the custom kernels in kernelforge.kernels (requires a GPU, or
               TRITON_INTERPRET=1 to emulate on CPU for testing)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kernelforge.quantize import dequantize

VALID_BACKENDS = ("torch", "triton")
_BACKEND = "torch"


def set_backend(name: str) -> None:
    global _BACKEND
    if name not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {name!r}")
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


def _use_triton() -> bool:
    return _BACKEND == "triton"


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if _use_triton():
        from kernelforge.kernels.layernorm import layernorm as k

        return k(x, weight, bias, eps)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _apply_activation(y: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return y
    if activation == "gelu":
        return F.gelu(y, approximate="tanh")
    raise ValueError(f"unsupported activation {activation!r}")


def linear(x, weight, bias=None, activation=None):
    """act(x @ weight + bias) with weight of shape (in, out)."""
    if _use_triton():
        from kernelforge.kernels.matmul import matmul

        return matmul(x, weight, bias=bias, activation=activation)
    y = x @ weight
    if bias is not None:
        y = y + bias
    return _apply_activation(y, activation)


def linear_w8(x, qweight, scale, bias=None, activation=None):
    """Like linear(), but with int8 weights (in, out) and a per-output-column scale."""
    if _use_triton():
        from kernelforge.kernels.matmul import matmul_w8

        return matmul_w8(x, qweight, scale, bias=bias, activation=activation)
    y = x @ dequantize(qweight, scale, x.dtype)
    if bias is not None:
        y = y + bias
    return _apply_activation(y, activation)


def scaled_causal_softmax(scores: torch.Tensor, scale: float) -> torch.Tensor:
    """softmax(scores * scale) with a causal mask; scores is (B, H, T_q, T_k), T_k >= T_q."""
    if _use_triton():
        from kernelforge.kernels.softmax import causal_softmax

        return causal_softmax(scores, scale)
    t_q, t_k = scores.shape[-2:]
    mask = torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device).tril(diagonal=t_k - t_q)
    probs = (scores.float() * scale).masked_fill(~mask, float("-inf"))
    return torch.softmax(probs, dim=-1).to(scores.dtype)
