"""End-to-end GPT-2 benchmark: backend x batch size x KV cache x int8.

    python -m benchmarks.bench_model [--tag colab-t4] [--random-init]

--random-init skips the Hugging Face download (throughput does not depend on the
weight values). Writes results/<tag>/model.csv.
"""

from __future__ import annotations

import argparse
import csv
import json

import torch

from benchmarks.common import env_info, require_cuda, results_dir
from kernelforge import ops
from kernelforge.model import GPT, GPTConfig, from_pretrained_gpt2

FIELDS = ["backend", "int8", "kv_cache", "batch", "prompt_len", "new_tokens", "prefill_ms",
          "decode_ms_per_token", "decode_tokens_per_s", "weight_bw_gbps", "peak_mem_gb"]


def build(args, backend, int8):
    ops.set_backend(backend)
    if args.random_init:
        torch.manual_seed(0)
        model = GPT(GPTConfig()).cuda().half().eval()
    else:
        model = from_pretrained_gpt2("gpt2", device="cuda", dtype=torch.float16)
    return model.quantize_int8() if int8 else model


@torch.no_grad()
def measure(model, batch, prompt_len, new_tokens, use_cache, reps=3):
    idx = torch.randint(0, model.cfg.vocab_size, (batch, prompt_len), device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def run():
        torch.cuda.synchronize()
        cur = idx
        if use_cache:
            cache = model.new_cache(batch, prompt_len + new_tokens)
            start.record()
            logits = model(cur, cache=cache, last_only=True)
            end.record(); torch.cuda.synchronize()
            prefill = start.elapsed_time(end)
            start.record()
            for _ in range(new_tokens):
                nxt = logits[:, -1].argmax(-1, keepdim=True)
                logits = model(nxt, cache=cache, last_only=True)
            end.record(); torch.cuda.synchronize()
            return prefill, start.elapsed_time(end)
        # no cache: each step re-runs the whole growing sequence
        start.record()
        for _ in range(new_tokens):
            logits = model(cur, last_only=True)
            cur = torch.cat([cur, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        end.record(); torch.cuda.synchronize()
        return 0.0, start.elapsed_time(end)

    run()  # warmup (also triggers Triton autotuning/compilation)
    torch.cuda.reset_peak_memory_stats()
    results = [run() for _ in range(reps)]
    prefill = sorted(r[0] for r in results)[len(results) // 2]
    decode_total = sorted(r[1] for r in results)[len(results) // 2]
    return prefill, decode_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    require_cuda()

    out = results_dir(args.tag)
    rows = []
    for backend, int8 in [("torch", False), ("triton", False), ("triton", True)]:
        model = build(args, backend, int8)
        for use_cache in (True, False):
            for batch in args.batches:
                if not use_cache and batch > 4:
                    continue  # the no-cache baseline is slow; small batches are enough to show the effect
                prefill, decode_total = measure(model, batch, args.prompt_len, args.new_tokens, use_cache)
                per_tok = decode_total / args.new_tokens
                row = dict(
                    backend=backend, int8=int8, kv_cache=use_cache, batch=batch,
                    prompt_len=args.prompt_len, new_tokens=args.new_tokens,
                    prefill_ms=round(prefill, 3), decode_ms_per_token=round(per_tok, 4),
                    decode_tokens_per_s=round(batch * 1000.0 / per_tok, 1),
                    weight_bw_gbps=round(model.num_bytes() / (per_tok / 1000.0) / 1e9, 1),
                    peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                )
                rows.append(row)
                print(row)
        del model
        torch.cuda.empty_cache()

    with open(out / "model.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "env.json").write_text(json.dumps(env_info(), indent=2))
    print(f"wrote {out}/model.csv")


if __name__ == "__main__":
    main()
