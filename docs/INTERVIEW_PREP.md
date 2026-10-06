# Explaining KernelForge (interview prep)

You should be able to answer every question below *from memory, in your own words*, and point at the code. If you
can't yet, re-read the relevant file until you can; the project is only worth listing if you understand it.

## 60-second pitch

"I implemented GPT-2 from scratch in PyTorch with a KV cache and checked its logits against the Hugging Face
reference. Then I profiled it, found that decode is memory-bound and that the attention softmax and the MLP's
bias+GELU were doing extra passes over memory, and wrote Triton kernels to fuse them: a fused scale-mask-softmax, a
fused LayerNorm, a tiled tensor-core matmul with a bias+GELU epilogue, and an int8 weight-only matmul. I benchmarked
each against PyTorch/cuBLAS, and used a roofline model to explain which kernels are bandwidth-bound versus
compute-bound and why the speedups land where they do."

## Questions to be ready for

**Why is decode memory-bound?** One token per step means the GEMMs have `M = batch`. Every weight is read once and used
for only `2·batch` FLOPs per element, so arithmetic intensity ≈ `batch` FLOP/byte, far below the GPU's ridge point
(~200 FLOP/B on a T4). Time ≈ bytes / bandwidth.

**Why does fusing softmax help?** Softmax is memory-bound: its cost is the number of passes over the score matrix.
Eager scale + mask + softmax is ~6 passes; fused is 2.

**Why did your matmul not beat cuBLAS on big squares (if it didn't)?** cuBLAS is hand-tuned per architecture
(instruction scheduling, tile shapes, split-K, etc.). Triton gets close with autotuning, but the real wins are
fusion (epilogues cuBLAS can't do) and shapes cuBLAS handles less well.

**What does `GROUP_M` do?** Re-orders which output tiles run consecutively so they share B tiles in L2.

**What does `num_stages` do?** Software pipelining: overlaps loading the next K-tile with computing on the current
one. More stages hide more latency but use more shared memory.

**How do you handle the causal mask with a KV cache?** Queries are the last `T_q` positions of `T_k`. Query `i`
sees keys `0 .. i + (T_k - T_q)`. See `softmax.py`.

**How does int8 quantization work here, and what's the error?** Per-output-channel symmetric scales; scale factors out
of the dot product so it is applied once per output column. Max error per weight is half a quantization step.
Weight-only: activations stay fp16.

**Why accumulate in float32?** Summing thousands of fp16 products loses precision and can overflow; tensor cores
accumulate fp16×fp16 into fp32 natively.

**How did you know your kernels were correct?** Reference comparison on awkward shapes, KV-cache-vs-full equivalence,
backend-vs-backend equivalence, and HF logits parity.

**What would you do next?** FlashAttention-style fusion, CUDA Graphs for launch overhead, SDPA baseline, paged KV
cache, fp8/int4.

**What are the weaknesses of your benchmarks?** Single GPU type; baseline attention is manual, not SDPA; synthetic
prompts; fixed sequence lengths. (Say this before they ask.)

## Turning your measurements into resume bullets

Only use numbers from `docs/RESULTS.md` / the CSVs you generated. Template (replace every `[…]`):

* Brought up an open-weight transformer (GPT-2 124M) for inference in PyTorch with a static KV cache, validating logits
  against the Hugging Face reference (max abs diff `[value from pytest -m hf]`).
* Wrote `[N]` Triton kernels (fused scale-mask-softmax, fused LayerNorm, tiled matmul with bias+GELU epilogue, int8
  weight-only matmul), achieving `[X]`× on fused softmax and `[Y]`× on bias+GELU matmul vs PyTorch on `[GPU]`.
* Used roofline analysis and `torch.profiler` to show decode is memory-bound (`[Z]`% of peak bandwidth), then cut
  weight traffic ≈ 34 % with int8 weights, improving batch-1 decode from `[A]` to `[B]` tokens/s.
* Built a reproducible benchmark suite (batch size, sequence length, KV cache on/off) with CSV/plot output and CI-run
  CPU correctness tests.

If a result is *not* a win (e.g., matmul vs cuBLAS), say so in the README. Hiring managers trust candidates who can
explain why.
