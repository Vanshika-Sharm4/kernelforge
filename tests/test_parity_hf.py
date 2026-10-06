"""Parity against the Hugging Face GPT-2 reference (downloads weights on first run).

Run with:  pytest -m hf
"""

import pytest
import torch

pytestmark = pytest.mark.hf

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def reference():
    try:
        hf = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
        tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as exc:  # no network / cache
        pytest.skip(f"could not load gpt2: {exc}")
    return hf, tok


@torch.no_grad()
def test_fp32_logits_match_huggingface(reference):
    from kernelforge import ops
    from kernelforge.model import from_pretrained_gpt2

    hf, tok = reference
    ids = torch.tensor([tok.encode("KernelForge brings up GPT-2 from scratch and checks it against the reference.")])
    ref = hf(ids).logits
    ops.set_backend("torch")
    ours = from_pretrained_gpt2("gpt2")(ids)
    assert (ours - ref).abs().max() < 1e-3
    assert torch.equal(ours.argmax(-1), ref.argmax(-1))


@torch.no_grad()
def test_triton_fp16_logits_close_to_huggingface(reference, triton_device):
    from kernelforge import ops
    from kernelforge.model import from_pretrained_gpt2

    hf, tok = reference
    ids = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog.")])
    ref = hf(ids).logits
    ops.set_backend("triton")
    dtype = torch.float16 if triton_device == "cuda" else torch.float32
    ours = from_pretrained_gpt2("gpt2", device=triton_device, dtype=dtype)(ids.to(triton_device)).float().cpu()
    # fp16 drifts a little; require top-1 agreement on nearly every position.
    agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean()
    assert agree >= 0.9
