"""Kernel micro-benchmarks: custom Triton kernels vs PyTorch/cuBLAS.

    python -m benchmarks.bench_kernels [--tag colab-t4] [--dtype float16]

Writes results/<tag>/kernels.csv and PNG plots.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmarks.common import env_info, record, require_cuda, results_dir, time_ms, write_csv
from kernelforge.kernels.layernorm import layernorm as tl_layernorm
from kernelforge.kernels.matmul import matmul as tl_matmul
from kernelforge.kernels.matmul import matmul_w8 as tl_matmul_w8
from kernelforge.kernels.softmax import causal_softmax as tl_causal_softmax
from kernelforge.kernels.softmax import softmax as tl_softmax
from kernelforge.quantize import quantize_per_channel


def torch_scale_mask_softmax(scores, scale):
    t_q, t_k = scores.shape[-2:]
    mask = torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device).tril(diagonal=t_k - t_q)
    return torch.softmax((scores * scale).masked_fill(~mask, float("-inf")), dim=-1)


def bench_softmax(rows, dtype):
    eb = torch.empty((), dtype=dtype).element_size()
    for n in [128, 256, 512, 1024, 2048, 4096, 8192]:
        m = 8192
        x = torch.randn(m, n, device="cuda", dtype=dtype)
        nbytes = 2 * m * n * eb  # read + write
        record(rows, "softmax", "torch", f"N={n}", m, n, 0, dtype, time_ms(lambda: torch.softmax(x, dim=-1)), 0, nbytes)
        record(rows, "softmax", "triton", f"N={n}", m, n, 0, dtype, time_ms(lambda: tl_softmax(x)), 0, nbytes)

    # The realistic attention case: scale + causal mask + softmax (unfused in eager PyTorch).
    for t in [128, 256, 512, 1024]:
        b, h = 8, 12
        scores = torch.randn(b, h, t, t, device="cuda", dtype=dtype)
        nbytes = 2 * scores.numel() * eb  # ideal: read once, write once
        record(rows, "causal_softmax", "torch", f"T={t}", b * h * t, t, 0, dtype,
               time_ms(lambda: torch_scale_mask_softmax(scores, 0.125)), 0, nbytes)
        record(rows, "causal_softmax", "triton", f"T={t}", b * h * t, t, 0, dtype,
               time_ms(lambda: tl_causal_softmax(scores, 0.125)), 0, nbytes)


def bench_layernorm(rows, dtype):
    eb = torch.empty((), dtype=dtype).element_size()
    for n in [768, 1024, 2048, 4096, 8192]:
        m = 8192
        x = torch.randn(m, n, device="cuda", dtype=dtype)
        w = torch.randn(n, device="cuda", dtype=dtype)
        b = torch.randn(n, device="cuda", dtype=dtype)
        nbytes = 2 * m * n * eb
        record(rows, "layernorm", "torch", f"N={n}", m, n, 0, dtype,
               time_ms(lambda: F.layer_norm(x, (n,), w, b, 1e-5)), 0, nbytes)
        record(rows, "layernorm", "triton", f"N={n}", m, n, 0, dtype,
               time_ms(lambda: tl_layernorm(x, w, b, 1e-5)), 0, nbytes)


def bench_matmul(rows, dtype):
    eb = torch.empty((), dtype=dtype).element_size()

    def add(kernel, label, m, n, k, torch_fn, triton_fn):
        flops = 2 * m * n * k
        nbytes = (m * k + k * n + m * n) * eb
        record(rows, kernel, "torch", label, m, n, k, dtype, time_ms(torch_fn), flops, nbytes)
        record(rows, kernel, "triton", label, m, n, k, dtype, time_ms(triton_fn), flops, nbytes)

    for s in [512, 1024, 2048, 4096]:  # large square: compute-bound
        a = torch.randn(s, s, device="cuda", dtype=dtype)
        b = torch.randn(s, s, device="cuda", dtype=dtype) / s**0.5
        add("matmul", f"square {s}", s, s, s, lambda: a @ b, lambda: tl_matmul(a, b))

    # GPT-2 small shapes: tokens = batch * seq; decode has M = batch (tiny -> memory-bound)
    for m in [1, 8, 64, 512, 4096]:
        for name, k, n in [("qkv", 768, 2304), ("mlp_up", 768, 3072), ("mlp_down", 3072, 768)]:
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            b = torch.randn(k, n, device="cuda", dtype=dtype) / k**0.5
            add("matmul", f"gpt2 {name} M={m}", m, n, k, lambda: a @ b, lambda: tl_matmul(a, b))

    # Fused bias + GELU epilogue vs eager matmul + bias + gelu (3 kernels)
    for m in [64, 512, 4096]:
        k, n = 768, 3072
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype) / k**0.5
        bias = torch.randn(n, device="cuda", dtype=dtype)
        add("matmul_bias_gelu", f"mlp_up M={m}", m, n, k,
            lambda: F.gelu(a @ b + bias, approximate="tanh"),
            lambda: tl_matmul(a, b, bias=bias, activation="gelu"))

    # int8 weights vs fp16 weights in the memory-bound decode regime
    for m in [1, 4, 16]:
        for name, k, n in [("mlp_up", 768, 3072), ("big 4096x4096", 4096, 4096), ("big 4096x11008", 4096, 11008)]:
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            w = torch.randn(k, n, device="cuda", dtype=dtype) / k**0.5
            q, scale = quantize_per_channel(w)
            flops = 2 * m * n * k
            nb_fp, nb_q = (m * k + k * n + m * n) * eb, m * k * eb + k * n + m * n * eb
            record(rows, "matmul_w8", "triton-fp16-weights", f"{name} M={m}", m, n, k, dtype,
                   time_ms(lambda: tl_matmul(a, w)), flops, nb_fp)
            record(rows, "matmul_w8", "triton-int8-weights", f"{name} M={m}", m, n, k, dtype,
                   time_ms(lambda: tl_matmul_w8(a, q, scale)), flops, nb_q)


def plot(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)

    def bars(kernel, metric, ylabel, fname, variants=("torch", "triton")):
        d = df[df.kernel == kernel]
        if d.empty:
            return
        labels = list(dict.fromkeys(d.label))
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels)), 4))
        width = 0.8 / len(variants)
        for i, v in enumerate(variants):
            vals = [d[(d.label == lab) & (d.variant == v)][metric].mean() for lab in labels]
            ax.bar([x + i * width for x in range(len(labels))], vals, width, label=v)
        ax.set_xticks([x + width * (len(variants) - 1) / 2 for x in range(len(labels))])
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{kernel} ({d.dtype.iloc[0]})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    bars("softmax", "gbps", "GB/s (higher is better)", "softmax_gbps.png")
    bars("causal_softmax", "gbps", "effective GB/s (higher is better)", "causal_softmax_gbps.png")
    bars("layernorm", "gbps", "GB/s (higher is better)", "layernorm_gbps.png")
    bars("matmul", "tflops", "TFLOPS (higher is better)", "matmul_tflops.png")
    bars("matmul_bias_gelu", "tflops", "TFLOPS (higher is better)", "matmul_bias_gelu_tflops.png")
    bars("matmul_w8", "gbps", "effective GB/s (higher is better)", "matmul_int8_gbps.png",
         variants=("triton-fp16-weights", "triton-int8-weights"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="results subfolder (default: GPU name)")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()
    require_cuda()

    dtype = getattr(torch, args.dtype)
    out = results_dir(args.tag)
    rows: list[dict] = []
    for name, fn in [("softmax", bench_softmax), ("layernorm", bench_layernorm), ("matmul", bench_matmul)]:
        print(f"== {name} ==")
        fn(rows, dtype)
    write_csv(rows, out / "kernels.csv")
    (out / "env.json").write_text(json.dumps(env_info(), indent=2))
    plot(rows, out)
    print(f"wrote {out}/kernels.csv and plots")


if __name__ == "__main__":
    main()
