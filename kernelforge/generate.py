"""Greedy text generation with or without a KV cache, plus a small CLI."""

from __future__ import annotations

import argparse
import time

import torch

from kernelforge import ops
from kernelforge.model import GPT, from_pretrained_gpt2


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
    """Greedy decoding. idx: (B, T) prompt ids. Returns (B, T + max_new_tokens)."""
    batch, prompt_len = idx.shape
    if prompt_len + max_new_tokens > model.cfg.block_size:
        raise ValueError("prompt + new tokens exceed block_size")

    if use_cache:
        cache = model.new_cache(batch, prompt_len + max_new_tokens)
        logits = model(idx, cache=cache, last_only=True)  # prefill: process the whole prompt once
        for _ in range(max_new_tokens):
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            logits = model(nxt, cache=cache, last_only=True)  # decode: one token per step
        return idx

    for _ in range(max_new_tokens):  # no cache: recompute the full sequence every step
        logits = model(idx, last_only=True)
        idx = torch.cat([idx, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with KernelForge GPT-2")
    parser.add_argument("--prompt", default="The key to fast LLM inference is")
    parser.add_argument("--model", default="gpt2", help="gpt2, gpt2-medium, ...")
    parser.add_argument("--backend", choices=ops.VALID_BACKENDS, default="triton")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--int8", action="store_true", help="int8 weight-only quantization")
    args = parser.parse_args()

    from transformers import GPT2TokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ops.set_backend(args.backend)
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = from_pretrained_gpt2(args.model, device=device, dtype=getattr(torch, args.dtype))
    if args.int8:
        model.quantize_int8()
    idx = torch.tensor([tok.encode(args.prompt)], device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = generate(model, idx, args.max_new_tokens, use_cache=not args.no_cache)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(tok.decode(out[0].tolist()))
    print(f"\n[{args.backend}/{args.dtype}{'/int8' if args.int8 else ''}] "
          f"{args.max_new_tokens / elapsed:.1f} tokens/s (includes prefill and any autotune warmup)")


if __name__ == "__main__":
    main()
