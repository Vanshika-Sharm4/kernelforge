"""Each Triton kernel is checked against a plain PyTorch reference."""

import pytest
import torch
import torch.nn.functional as F

from conftest import tol


@pytest.mark.parametrize("shape", [(4, 33), (7, 128), (3, 1000)])
def test_softmax(triton_device, dtypes, shape):
    from kernelforge.kernels.softmax import softmax

    for dtype in dtypes:
        x = torch.randn(shape, device=triton_device, dtype=dtype) * 3
        ref = torch.softmax(x.float() * 0.5, dim=-1).to(dtype)
        torch.testing.assert_close(softmax(x, scale=0.5), ref, **tol(dtype))


@pytest.mark.parametrize("t_q,t_k", [(8, 8), (5, 17), (1, 40), (16, 16)])
def test_causal_softmax(triton_device, dtypes, t_q, t_k):
    """Includes T_q < T_k, which is the KV-cache decode/prefill-continuation case."""
    from kernelforge.kernels.softmax import causal_softmax

    for dtype in dtypes:
        scores = torch.randn(2, 3, t_q, t_k, device=triton_device, dtype=dtype)
        mask = torch.ones(t_q, t_k, dtype=torch.bool, device=triton_device).tril(diagonal=t_k - t_q)
        ref = torch.softmax((scores.float() * 0.125).masked_fill(~mask, float("-inf")), dim=-1).to(dtype)
        out = causal_softmax(scores, 0.125)
        torch.testing.assert_close(out, ref, **tol(dtype))
        assert torch.all(out[..., ~mask] == 0)  # masked positions get exactly zero probability


@pytest.mark.parametrize("shape", [(4, 64), (5, 768), (3, 100)])
def test_layernorm(triton_device, dtypes, shape):
    from kernelforge.kernels.layernorm import layernorm

    for dtype in dtypes:
        x = torch.randn(shape, device=triton_device, dtype=dtype) * 2 + 1
        w = torch.randn(shape[-1], device=triton_device, dtype=dtype)
        b = torch.randn(shape[-1], device=triton_device, dtype=dtype)
        ref = F.layer_norm(x.float(), (shape[-1],), w.float(), b.float(), 1e-5).to(dtype)
        torch.testing.assert_close(layernorm(x, w, b), ref, **tol(dtype))


@pytest.mark.parametrize("m,k,n", [(1, 64, 96), (17, 100, 70), (64, 128, 128), (33, 257, 65)])
@pytest.mark.parametrize("bias,act", [(False, None), (True, None), (True, "gelu")])
def test_matmul(triton_device, dtypes, m, k, n, bias, act):
    from kernelforge.kernels.matmul import matmul

    for dtype in dtypes:
        a = torch.randn(m, k, device=triton_device, dtype=dtype)
        b = torch.randn(k, n, device=triton_device, dtype=dtype) / k**0.5
        bv = torch.randn(n, device=triton_device, dtype=dtype) if bias else None
        ref = a.float() @ b.float()
        if bv is not None:
            ref = ref + bv.float()
        if act == "gelu":
            ref = F.gelu(ref, approximate="tanh")
        torch.testing.assert_close(matmul(a, b, bias=bv, activation=act), ref.to(dtype), **tol(dtype))


def test_matmul_transposed_weight(triton_device):
    """The tied LM head passes wte.weight.t(), a non-contiguous (K, N) view."""
    from kernelforge.kernels.matmul import matmul

    a = torch.randn(5, 48, device=triton_device)
    w = torch.randn(70, 48, device=triton_device)  # (vocab, embd)
    torch.testing.assert_close(matmul(a, w.t()), a @ w.t(), **tol(torch.float32))


def test_matmul_batched_leading_dims(triton_device):
    from kernelforge.kernels.matmul import matmul

    a = torch.randn(2, 3, 40, device=triton_device)
    b = torch.randn(40, 50, device=triton_device)
    out = matmul(a, b)
    assert out.shape == (2, 3, 50)
    torch.testing.assert_close(out, a @ b, **tol(torch.float32))


@pytest.mark.parametrize("m,k,n", [(1, 64, 96), (9, 130, 70)])
def test_matmul_int8(triton_device, dtypes, m, k, n):
    from kernelforge.kernels.matmul import matmul_w8
    from kernelforge.quantize import dequantize, quantize_per_channel

    for dtype in dtypes:
        a = torch.randn(m, k, device=triton_device, dtype=dtype)
        w = torch.randn(k, n, device=triton_device) / k**0.5
        q, scale = quantize_per_channel(w)
        ref = (a.float() @ dequantize(q, scale, torch.float32)).to(dtype)  # same quantized weights
        torch.testing.assert_close(matmul_w8(a, q, scale), ref, **tol(dtype))


def test_quantization_error_is_small():
    from kernelforge.quantize import dequantize, quantize_per_channel

    w = torch.randn(256, 128)
    q, scale = quantize_per_channel(w)
    assert q.dtype == torch.int8 and scale.shape == (128,)
    err = (dequantize(q, scale, torch.float32) - w).abs().max()
    assert err <= scale.max() / 2 + 1e-6  # rounding error is at most half a quantization step
