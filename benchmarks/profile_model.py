"""Profile one prefill + a few decode steps with torch.profiler.

    python -m benchmarks.profile_model --backend triton [--random-init]

Prints the top GPU kernels by total time and writes a Chrome trace to
results/traces/ (open it in chrome://tracing or https://ui.perfetto.dev).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmarks.common import require_cuda
from kernelforge import ops
from kernelforge.model import GPT, GPTConfig, from_pretrained_gpt2


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=ops.VALID_BACKENDS, default="triton")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    require_cuda()

    ops.set_backend(args.backend)
    if args.random_init:
        torch.manual_seed(0)
        model = GPT(GPTConfig()).cuda().half().eval()
    else:
        model = from_pretrained_gpt2("gpt2", device="cuda", dtype=torch.float16)

    idx = torch.randint(0, model.cfg.vocab_size, (args.batch, args.prompt_len), device="cuda")

    def run():
        cache = model.new_cache(args.batch, args.prompt_len + args.steps)
        with record_function("prefill"):
            logits = model(idx, cache=cache, last_only=True)
        with record_function("decode"):
            for _ in range(args.steps):
                logits = model(logits[:, -1].argmax(-1, keepdim=True), cache=cache, last_only=True)

    run()  # warmup / autotune
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        run()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=60))
    trace_dir = Path("results/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"trace_{args.backend}_b{args.batch}.json"
    prof.export_chrome_trace(str(path))
    print(f"trace written to {path}")


if __name__ == "__main__":
    main()
