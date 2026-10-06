# KernelForge: design notes

This document explains *why* the project is built the way it is, and gives the performance reasoning behind each
optimization. Read it before you run the benchmarks so you know what the results should look like and why.

## 1. Scope

* **Model:** GPT-2 124M (12 layers, 12 heads, 768 hidden, vocab 50257, context 1024), inference only.
* **Hardware:** any NVIDIA GPU supported by Triton (developed for a free Colab/Kaggle T4).
* **Goal:** a small, fully understood system that exercises the full optimization loop (bring-up, verification, profiling,
  kernels, benchmarking, roofline explanation), not a production inference server.

## 2. Model bring-up

`model.py` re-implements GPT-2 rather than wrapping `transformers`, so every operation can be redirected through
`ops.py`. Linear weights are stored as `(in, out)` (Hugging Face's `Conv1D` layout), so checkpoints load without
transposes and `y = x @ W + b`. The output head is tied to the token embedding and uses `wte.weight.t()`, a
non-contiguous `(K, N)` view; the matmul kernel takes explicit strides so it handles this without a copy.

**Verification.** Bring-up is only trustworthy if numerics are checked:
1. Our fp32 logits must match Hugging Face GPT-2 (`pytest -m hf`, max abs diff < 1e-3).
2. KV-cached incremental decoding must equal full recomputation.
3. The Triton backend must equal the PyTorch backend.

## 3. Where the time goes (memory-bound vs compute-bound)

A kernel's attainable performance is bounded by `min(peak_FLOPS, arithmetic_intensity × peak_bandwidth)`, where
arithmetic intensity (AI) is FLOPs per byte moved to/from DRAM.

For a GEMM `(M×K) @ (K×N)` in fp16: `AI = 2MNK / (2(MK + KN + MN))`.

| Regime | M | AI (approx) | Bound by |
|---|---|---|---|
| Large GEMM / prefill | 1000s | 100s FLOP/B | compute (tensor cores) |
| Decode, batch 1 | 1 | ≈ 1 FLOP/B | memory bandwidth |
| Decode, batch 16 | 16 | ≈ 16 FLOP/B | memory bandwidth |

On a T4 (≈65 TFLOPS fp16 tensor, ≈320 GB/s) the ridge point is ≈ 200 FLOP/B, so *all* decode GEMMs are far into the
memory-bound region. That is why:

* **Decode is limited by streaming weights.** GPT-2 small is ≈ 249 MB in fp16, so at 320 GB/s a decode step can't take
  less than ≈ 0.78 ms (≈ 1,280 tokens/s at batch 1). `bench_model.py` reports `weight_bw_gbps` so you can see what
  fraction of peak bandwidth you actually reach.
* **int8 weights help decode, not prefill.** The transformer blocks hold ≈ 170 MB of those bytes in fp16 (≈ 85 MB in
  int8); the tied embedding/LM head (≈ 77 MB) stays fp16. Total traffic drops ≈ 34 %, so the best case for a purely
  bandwidth-bound decode step is ≈ 1.5×. In compute-bound prefill the extra dequantization work gives no benefit.
* **Batching is nearly free in decode.** Going from batch 1 to 16 re-uses the same weight bytes for 16× the FLOPs, so
  tokens/s scales almost linearly until AI approaches the ridge.

If your measured batch-1 decode is *much* slower than the bandwidth floor, the likely culprit is not bandwidth but
**kernel launch overhead** (GPT-2 small launches several hundred tiny kernels per step). Check the profiler trace for
gaps between kernels. CUDA Graphs are the standard fix (see roadmap).

## 4. Kernels

### 4.1 Fused scale + causal mask + softmax (`kernels/softmax.py`)
Eager PyTorch computes `softmax(mask(scores * scale))` as separate kernels (scale, masked_fill, softmax), each reading
and writing the full `(B, H, T_q, T_k)` score tensor: ≈ 6 passes over DRAM. The fused kernel loads each row once, does
everything in registers, and stores once: 2 passes, so up to ≈ 3× less traffic. Because softmax is memory-bound, traffic
reduction translates almost directly to speedup.

* One program per row; the row is reduced with `tl.max` and `tl.sum` after subtracting the max for numerical stability.
* Computation is in float32 even for fp16 inputs.
* **KV-cache aware masking.** With a cache the queries are the *last* `T_q` of `T_k` positions. Row `r` is query
  `r % T_q`, whose last visible key is `(r % T_q) + (T_k - T_q)`. Tests cover `T_q = 1`, `T_q < T_k` and `T_q = T_k`.

### 4.2 Fused LayerNorm (`kernels/layernorm.py`)
One program per row computes mean and variance (float32) and writes the normalized, scaled output in one pass.
Eager PyTorch's `layer_norm` is already a fused CUDA kernel, so expect parity or a modest win, not a large one. This
is an honest data point worth discussing.

### 4.3 Tiled matmul with fused epilogue (`kernels/matmul.py`)
* **Tiling.** Each program owns a `BLOCK_M × BLOCK_N` output tile and loops over K in `BLOCK_K` chunks. Data reuse
  within the tile turns `O(MNK)` global loads into `O(MNK / tile)`; `tl.dot` maps to tensor cores.
* **Grouped ordering (`GROUP_M`).** Launching programs in groups of `GROUP_M` row-tiles makes consecutive programs share
  the same B columns, raising L2 hit rate.
* **Autotuning.** Block shapes, `num_warps` and `num_stages` (software pipelining depth) are searched per `(M, N, K)`.
  Configs that exceed shared memory are skipped automatically; small-`M` tiles are included for decode.
* **Fused epilogue.** Bias and tanh-GELU run on the float32 accumulator before the single store. Eager PyTorch needs
  separate kernels for bias and GELU, each re-reading `C`. The fusion matters most for the MLP up-projection
  (`M × 3072`).
* **Honest expectation.** cuBLAS is heavily tuned; on large square GEMMs the Triton kernel typically reaches a
  fraction to roughly parity. The wins are fusion and small-M decode shapes.

### 4.4 int8 weight-only matmul (W8A16)
Weights are quantized symmetrically per output column: `q = round(w / s)`, `s = max|w| / 127`. Because the scale is
per column, it factors out of the dot product, so the kernel accumulates `a @ q` in float32 and multiplies by
`s[n]` once in the epilogue. int8 → fp16 conversion is exact, so the only error is the rounding in `q`
(≤ ½ quantization step per weight; tested). Activations stay fp16, which avoids activation-outlier problems at the
cost of not using int8 tensor cores.

## 5. KV cache

Without a cache each decode step re-runs the entire growing sequence: step `t` costs `O(t)` tokens of compute, so
generating `n` tokens is `O(n²)` token-passes. With a cache, step `t` processes one token and only *reads* the stored
keys/values. The cache holds `2 × n_layer × n_embd × bytes` per token per sequence = 36.9 KB for GPT-2 small in fp16
(≈ 38 MB at 1,024 tokens), preallocated statically so no allocation happens in the decode loop.

## 6. Numerics

* Accumulate matmuls in float32; do softmax/LayerNorm statistics in float32.
* On Ampere+ GPUs `tl.dot` on float32 inputs may use TF32 (≈ 1e-3 relative error). Test tolerances account for this.
* fp16 results are compared with loose tolerances (2e-2) and, end-to-end, by top-1 token agreement.

## 7. Limitations and what I would do next

* **Baseline strength.** The PyTorch attention path is a manual implementation, not
  `torch.nn.functional.scaled_dot_product_attention` (which can dispatch to FlashAttention). Any speedup claimed over
  the manual path is *not* a claim about beating SDPA. Adding an SDPA baseline is the first thing to do.
* **Partially fused attention.** Scores are materialised; a FlashAttention-style kernel (tile K/V, online softmax,
  never write the `T×T` matrix) removes that traffic entirely.
* **Single-block rows** in softmax/LayerNorm (≤ 65,536 elements).
* **No CUDA Graphs**, so batch-1 decode pays launch overhead.
* **Static batch**, no paged KV cache or continuous batching.

## 8. Mapping to other accelerators

The concepts transfer directly to other ML accelerators and kernel languages: tiling to on-chip memory, keeping
matmul accumulation in higher precision, fusing elementwise epilogues into the producer, deciding bound-ness with a
roofline, and validating every kernel against a reference implementation.
