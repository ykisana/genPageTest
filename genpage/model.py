"""Decoder-only transformer (GPT-style) with a KV cache that supports multi-token append,
truncation (for speculative rollback) and batch expansion (for skeleton parallel fill).

Input and output embeddings are untied, as in GenPage, because NTP pretraining (softmax)
and WBC post-training (per-token sigmoid) want different logit scales.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from genpage.config import ModelConfig


class KVCache:
    def __init__(self, n_layers, batch, n_heads, max_len, head_dim, device, dtype=torch.float32):
        shape = (n_layers, batch, n_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.len = 0

    def truncate(self, n: int):
        assert 0 <= n <= self.len
        self.len = n

    def expand(self, batch: int):
        """Repeat a batch-1 prefix for `batch` independent continuations (copies only the live prefix)."""
        assert self.k.shape[1] == 1
        n = self.len
        k = self.k.new_zeros((self.k.shape[0], batch, *self.k.shape[2:]))
        v = torch.zeros_like(k)
        k[:, :, :, :n] = self.k[:, :, :, :n]
        v[:, :, :, :n] = self.v[:, :, :, :n]
        self.k, self.v = k, v


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_tables(max_len, head_dim, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    f = torch.outer(torch.arange(max_len).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [B, 1, T, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.h = cfg.d_model, cfg.n_heads
        self.hd = d // cfg.n_heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        hidden = int(8 * d / 3 / 64 + 1) * 64
        self.w12 = nn.Linear(d, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
        self.drop = cfg.dropout

    def forward(self, x, cos, sin, layer_cache, attn_mask, is_causal):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if layer_cache is not None:
            ck, cv, start = layer_cache
            ck[:, :, start:start + T] = k
            cv[:, :, start:start + T] = v
            k, v = ck[:, :, :start + T], cv[:, :, :start + T]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        a, b = self.w12(self.n2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(a) * b)


class MTPHead(nn.Module):
    """Medusa-style residual block + its own vocab projection."""

    def __init__(self, d, vocab):
        super().__init__()
        self.lin = nn.Linear(d, d)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.out = nn.Linear(d, vocab, bias=False)

    def forward(self, h):
        return self.out(h + F.silu(self.lin(h)))


class GenPageLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_idx_emb = nn.Embedding(cfg.n_row_slots, cfg.d_model) if cfg.n_row_slots else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)   # untied from self.emb
        self.mtp = nn.ModuleList(MTPHead(cfg.d_model, cfg.vocab_size) for _ in range(cfg.n_mtp_heads))
        cos, sin = rope_tables(cfg.max_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(b.w3.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)) and m.weight.requires_grad:
            if not (isinstance(m, nn.Linear) and m.weight.abs().sum() == 0):
                nn.init.normal_(m.weight, std=0.02)

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.weight.numel() + self.head.weight.numel()
            n -= sum(h.out.weight.numel() for h in self.mtp)
        return n

    def new_cache(self, batch=1, max_len=None, device=None):
        c = self.cfg
        return KVCache(c.n_layers, batch, c.n_heads, max_len or c.max_len, c.d_model // c.n_heads,
                       device or self.head.weight.device, self.head.weight.dtype)

    def hidden(self, idx, cache: KVCache = None, pos=None, attn_mask=None, row_idx=None):
        """idx [B,T] -> final hidden states [B,T,D]. Appends to `cache` if given.

        pos: optional [B,T] or [T] position ids (default: contiguous after the cache).
        attn_mask: optional bool mask broadcastable to [B,1,T,S] (True = may attend).
        row_idx: optional [B,T] row-slot ids (skeleton model), -1 = none.
        """
        B, T = idx.shape
        start = cache.len if cache is not None else 0
        if pos is None:
            pos = torch.arange(start, start + T, device=idx.device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(B, T)
        cos = self.rope_cos[pos].unsqueeze(1)
        sin = self.rope_sin[pos].unsqueeze(1)
        x = self.emb(idx)
        if row_idx is not None and self.row_idx_emb is not None:
            x = x + self.row_idx_emb(row_idx.clamp(min=0)) * (row_idx >= 0).unsqueeze(-1)

        is_causal = False
        if attn_mask is None:
            if cache is None or start == 0:
                is_causal = T > 1
            elif T > 1:
                q = torch.arange(start, start + T, device=idx.device).unsqueeze(1)
                attn_mask = torch.arange(start + T, device=idx.device).unsqueeze(0) <= q
        for i, b in enumerate(self.blocks):
            lc = (cache.k[i], cache.v[i], start) if cache is not None else None
            x = b(x, cos, sin, lc, attn_mask, is_causal)
        if cache is not None:
            cache.len = start + T
        return self.norm(x)

    def forward(self, idx, cache=None, **kw):
        return self.head(self.hidden(idx, cache, **kw))
