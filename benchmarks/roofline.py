"""Roofline plot from results/<tag>/kernels.csv.

    python -m benchmarks.roofline --tag colab-t4 [--peak-tflops 65 --peak-gbps 320]

Achieved TFLOPS is plotted against arithmetic intensity (FLOPs per byte of DRAM
traffic). Points under the sloped part of the roof are memory-bound; points under
the flat part are compute-bound.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.common import guess_peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--peak-gbps", type=float, default=None)
    args = parser.parse_args()

    out = Path("results") / args.tag
    df = pd.read_csv(out / "kernels.csv")
    df = df[df.flops > 0].copy()
    df["ai"] = df.flops / df.bytes
    df["tflops_achieved"] = df.flops / df.ms / 1e9

    peaks = (args.peak_tflops, args.peak_gbps)
    if None in peaks:
        guessed = guess_peaks()
        if guessed is None:
            raise SystemExit("Unknown GPU: pass --peak-tflops and --peak-gbps from your GPU datasheet.")
        peaks = (args.peak_tflops or guessed[0], args.peak_gbps or guessed[1])
    peak_tflops, peak_gbps = peaks

    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.logspace(-1, 4, 200)
    ax.plot(xs, np.minimum(peak_gbps * xs / 1e3, peak_tflops), "k-", lw=2, label="roofline")
    ridge = peak_tflops * 1e3 / peak_gbps
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge * 1.05, peak_tflops * 0.02, f"ridge ≈ {ridge:.0f} FLOP/B", color="gray", fontsize=8)

    styles = {"torch": ("o", "tab:blue"), "triton": ("^", "tab:red")}
    for variant, (marker, color) in styles.items():
        d = df[(df.variant == variant) & (df.kernel.isin(["matmul", "matmul_bias_gelu"]))]
        ax.scatter(d.ai, d.tflops_achieved, marker=marker, color=color, alpha=0.75, label=f"{variant} matmul")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title("Matmul roofline (decode shapes sit far left; large GEMMs sit near the flat roof)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roofline.png", dpi=150)
    print(f"wrote {out}/roofline.png (peaks: {peak_tflops} TFLOPS, {peak_gbps} GB/s)")


if __name__ == "__main__":
    main()
