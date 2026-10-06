"""A from-scratch GPT-2 implementation with a static KV cache and pluggable kernels.

Written explicitly (instead of wrapping Hugging Face) so every op routes through
kernelforge.ops and can be swapped between PyTorch and custom Triton kernels.
Weights can be loaded from the Hugging Face GPT-2 checkpoints for parity tests.

Linear weights are stored as (in_features, out_features) (the Hugging Face
"Conv1D" layout), so y = x @ W + b and checkpoints load without transposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelforge import ops
from kernelforge.quantize import quantize_per_channel


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    layer_norm_eps: float = 1e-5


class KVCache:
    """Static, preallocated key/value cache: (n_layer, batch, n_head, max_len, head_dim)."""

    def __init__(self, cfg: GPTConfig, batch_size: int, max_len: int, device, dtype):
        head_dim = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch_size, cfg.n_head, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0  # number of positions already filled

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        t = k_new.shape[2]
        start, end = self.length, self.length + t
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        self.k[layer, :, :, start:end] = k_new
        self.v[layer, :, :, start:end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, t: int) -> None:
        self.length += t


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return ops.layernorm(x, self.weight, self.bias, self.eps)


class Linear(nn.Module):
    """y = act(x @ W + b), W: (in, out). Can be converted to int8 weights in place."""

    def __init__(self, in_features: int, out_features: int, activation: str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.activation = activation
        self.register_buffer("qweight", None)
        self.register_buffer("qscale", None)

    @property
    def is_quantized(self) -> bool:
        return self.qweight is not None

    def quantize(self) -> None:
        q, scale = quantize_per_channel(self.weight)
        self.qweight = q.to(self.weight.device)
        self.qscale = scale.to(self.weight.device)
        self.register_parameter("weight", None)  # free the floating point copy

    def forward(self, x):
        if self.is_quantized:
            return ops.linear_w8(x, self.qweight, self.qscale, self.bias, self.activation)
        return ops.linear(x, self.weight, self.bias, self.activation)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, layer: int, cache: KVCache | None = None):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(layer, k, v)  # (B, H, T_total, D)

        scores = q @ k.transpose(-2, -1)  # (B, H, T_q, T_k)
        probs = ops.scaled_causal_softmax(scores, 1.0 / math.sqrt(self.head_dim))
        y = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = Linear(cfg.n_embd, 4 * cfg.n_embd, activation="gelu")  # GELU fused in the epilogue
        self.c_proj = Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(self.c_fc(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, layer: int, cache: KVCache | None = None):
        x = x + self.attn(self.ln_1(x), layer, cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = LayerNorm(cfg.n_embd, cfg.layer_norm_eps)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.01)

    def new_cache(self, batch_size: int, max_len: int | None = None) -> KVCache:
        p = self.wte.weight
        return KVCache(self.cfg, batch_size, max_len or self.cfg.block_size, p.device, p.dtype)

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None, last_only: bool = False):
        """idx: (B, T) token ids -> logits (B, T, V), or (B, 1, V) if last_only."""
        _, t = idx.shape
        start = cache.length if cache is not None else 0
        if start + t > self.cfg.block_size:
            raise ValueError("sequence longer than block_size")
        pos = torch.arange(start, start + t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for i, block in enumerate(self.blocks):
            x = block(x, i, cache)
        x = self.ln_f(x)
        if cache is not None:
            cache.advance(t)
        if last_only:
            x = x[:, -1:, :]
        return ops.linear(x, self.wte.weight.t())  # tied output head

    def quantize_int8(self) -> "GPT":
        """Convert every transformer-block Linear to int8 weights (embeddings stay fp)."""
        for m in self.modules():
            if isinstance(m, Linear) and not m.is_quantized:
                m.quantize()
        return self

    def num_bytes(self) -> int:
        """Bytes of all parameters and buffers (what a decode step must stream from DRAM)."""
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        total += sum(b.numel() * b.element_size() for b in self.buffers() if b is not None)
        return total


def from_pretrained_gpt2(name: str = "gpt2", device="cpu", dtype=torch.float32) -> GPT:
    """Load Hugging Face GPT-2 weights (needs `transformers` and network on first use)."""
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(name)
    hc = hf.config
    cfg = GPTConfig(
        vocab_size=hc.vocab_size,
        block_size=hc.n_positions,
        n_layer=hc.n_layer,
        n_head=hc.n_head,
        n_embd=hc.n_embd,
        layer_norm_eps=hc.layer_norm_epsilon,
    )
    model = GPT(cfg)
    sd = hf.state_dict()

    def copy(dst: torch.Tensor, key: str) -> None:
        dst.copy_(sd[key])

    with torch.no_grad():
        copy(model.wte.weight, "transformer.wte.weight")
        copy(model.wpe.weight, "transformer.wpe.weight")
        copy(model.ln_f.weight, "transformer.ln_f.weight")
        copy(model.ln_f.bias, "transformer.ln_f.bias")
        for i, blk in enumerate(model.blocks):
            p = f"transformer.h.{i}."
            copy(blk.ln_1.weight, p + "ln_1.weight")
            copy(blk.ln_1.bias, p + "ln_1.bias")
            copy(blk.ln_2.weight, p + "ln_2.weight")
            copy(blk.ln_2.bias, p + "ln_2.bias")
            copy(blk.attn.c_attn.weight, p + "attn.c_attn.weight")
            copy(blk.attn.c_attn.bias, p + "attn.c_attn.bias")
            copy(blk.attn.c_proj.weight, p + "attn.c_proj.weight")
            copy(blk.attn.c_proj.bias, p + "attn.c_proj.bias")
            copy(blk.mlp.c_fc.weight, p + "mlp.c_fc.weight")
            copy(blk.mlp.c_fc.bias, p + "mlp.c_fc.bias")
            copy(blk.mlp.c_proj.weight, p + "mlp.c_proj.weight")
            copy(blk.mlp.c_proj.bias, p + "mlp.c_proj.bias")
    return model.to(device=device, dtype=dtype).eval()
