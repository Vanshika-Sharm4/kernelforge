"""Shared fixtures.

Triton kernels run on a CUDA GPU. For CI / laptops without a GPU, set
TRITON_INTERPRET=1 to emulate kernels on the CPU (slow, small shapes only).
"""

import os

import pytest
import torch


def _triton_device():
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    return None


@pytest.fixture(scope="session")
def triton_device():
    dev = _triton_device()
    if dev is None:
        pytest.skip("needs a CUDA GPU, or TRITON_INTERPRET=1 for CPU emulation")
    return dev


@pytest.fixture(scope="session")
def dtypes(triton_device):
    # fp16 on the CPU interpreter is slow and not what we want to validate there.
    return [torch.float32, torch.float16] if triton_device == "cuda" else [torch.float32]


@pytest.fixture(autouse=True)
def _reset_backend():
    from kernelforge import ops

    ops.set_backend("torch")
    yield
    ops.set_backend("torch")


def tol(dtype):
    return dict(atol=2e-2, rtol=2e-2) if dtype == torch.float16 else dict(atol=2e-3, rtol=2e-3)
