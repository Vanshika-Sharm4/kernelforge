"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import torch

# Theoretical peak specs: (dense FP16 tensor-core TFLOPS, DRAM bandwidth GB/s).
# These are vendor datasheet numbers and are only used to draw roofline ceilings.
# Double-check them for your GPU, or override with --peak-tflops / --peak-gbps.
GPU_SPECS = {
    "T4": (65.0, 320.0),
    "L4": (121.0, 300.0),
    "A100": (312.0, 1555.0),
    "H100": (989.0, 3350.0),
}

CSV_FIELDS = ["kernel", "variant", "label", "M", "N", "K", "dtype", "ms", "flops", "bytes", "tflops", "gbps"]


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def gpu_slug() -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", gpu_name()).strip("-").lower()


def guess_peaks(name: str | None = None) -> tuple[float, float] | None:
    name = name or gpu_name()
    for key, spec in GPU_SPECS.items():
        if key in name:
            return spec
    return None


def results_dir(tag: str | None) -> Path:
    path = Path("results") / (tag or gpu_slug())
    path.mkdir(parents=True, exist_ok=True)
    return path


def time_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    """Median runtime in ms using Triton's CUDA-event based timer (flushes L2 between runs)."""
    import triton

    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def record(rows: list[dict], kernel, variant, label, m, n, k, dtype, ms, flops, nbytes) -> None:
    rows.append(
        dict(
            kernel=kernel,
            variant=variant,
            label=label,
            M=m,
            N=n,
            K=k,
            dtype=str(dtype).replace("torch.", ""),
            ms=ms,
            flops=flops,
            bytes=nbytes,
            tflops=flops / ms / 1e9 if flops else 0.0,
            gbps=nbytes / ms / 1e6,
        )
    )


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("These benchmarks need a CUDA GPU (try Google Colab or Kaggle with a GPU runtime).")


def env_info() -> dict:
    import triton

    return {
        "gpu": gpu_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        "os_env_note": os.environ.get("KERNELFORGE_NOTE", ""),
    }
