"""Run every benchmark in order and build the report.

    python -m benchmarks.run_all --tag colab-t4 [--random-init]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(*cmd: str) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m", *cmd], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="e.g. colab-t4")
    parser.add_argument("--random-init", action="store_true", help="skip the Hugging Face download")
    args = parser.parse_args()

    run("benchmarks.bench_kernels", "--tag", args.tag)
    model_args = ["--random-init"] if args.random_init else []
    run("benchmarks.bench_model", "--tag", args.tag, *model_args)
    run("benchmarks.roofline", "--tag", args.tag)
    run("benchmarks.make_report", "--tag", args.tag)
    run("benchmarks.profile_model", "--backend", "torch", *model_args)
    run("benchmarks.profile_model", "--backend", "triton", *model_args)


if __name__ == "__main__":
    main()
