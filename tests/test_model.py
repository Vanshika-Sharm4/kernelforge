"""End-to-end model checks on a tiny random GPT (no downloads needed)."""

import pytest
import torch

from kernelforge import ops
from kernelforge.generate import generate
from kernelforge.model import GPT, GPTConfig


def tiny_model(device="cpu", seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=97, block_size=64, n_layer=2, n_head=2, n_embd=32)
    return GPT(cfg).to(device).eval()


@torch.no_grad()
def test_kv_cache_matches_full_forward():
    """Prefill + one-token decode steps must reproduce the no-cache logits exactly."""
    model = tiny_model()
    idx = torch.randint(0, 97, (2, 12))
    full = model(idx)  # (2, 12, V)

    cache = model.new_cache(batch_size=2, max_len=32)
    out = [model(idx[:, :7], cache=cache)]
    for t in range(7, 12):
        out.append(model(idx[:, t : t + 1], cache=cache))
    torch.testing.assert_close(torch.cat(out, dim=1), full, atol=1e-5, rtol=1e-5)
    assert cache.length == 12


@torch.no_grad()
def test_generate_with_and_without_cache_agree():
    model = tiny_model()
    prompt = torch.randint(0, 97, (2, 5))
    a = generate(model, prompt, max_new_tokens=10, use_cache=True)
    b = generate(model, prompt, max_new_tokens=10, use_cache=False)
    assert torch.equal(a, b)


@torch.no_grad()
def test_causality():
    """Changing a future token must not change earlier logits."""
    model = tiny_model()
    idx = torch.randint(0, 97, (1, 10))
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 97
    torch.testing.assert_close(model(idx)[:, :-1], model(idx2)[:, :-1])


@torch.no_grad()
def test_triton_backend_matches_torch_backend(triton_device):
    model = tiny_model(triton_device)
    idx = torch.randint(0, 97, (2, 11), device=triton_device)

    ops.set_backend("torch")
    ref = model(idx)
    ops.set_backend("triton")
    out = model(idx)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    # also with the KV cache (exercises T_q < T_k in the fused softmax)
    cache = model.new_cache(2, 32)
    pieces = [model(idx[:, :6], cache=cache)] + [model(idx[:, t : t + 1], cache=cache) for t in range(6, 11)]
    torch.testing.assert_close(torch.cat(pieces, dim=1), ref, atol=2e-3, rtol=2e-3)


@torch.no_grad()
def test_int8_model_stays_close(triton_device):
    ref_model = tiny_model(triton_device)
    q_model = tiny_model(triton_device).quantize_int8()
    idx = torch.randint(0, 97, (2, 9), device=triton_device)
    assert all(m.is_quantized for m in q_model.modules() if hasattr(m, "is_quantized"))
    ref = ref_model(idx)
    for backend in ("torch", "triton"):
        ops.set_backend(backend)
        out = q_model(idx)
        assert (out - ref).abs().max() < 0.1  # tiny random model: int8 error stays small
