"""Prefix-LM transformer: [view tokens (bidirectional)] + [history tokens (causal)].

The view block is the *current* observation (27 sticker colors for the U/F/R corner view,
or 54 for the full-view oracle ablation). It is recomputed after every action. The history
holds the model's own past action tokens (moves, rotations, scratch). The model predicts the
next action token from the hidden state at the last position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as T


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ff_mult: float = 8 / 3
    view_len: int = 27
    max_hist: int = 640
    vocab: int = T.VOCAB
    dropout: float = 0.0
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


def rope_cache(seq_len, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: [B,H,S,D]
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = RMSNorm(d)
        ff = int(cfg.ff_mult * d)
        ff = (ff + 31) // 32 * 32
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)
        self.drop = cfg.dropout

    def forward(self, x, cos, sin, mask):
        B, S, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, D))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class CubeLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab, d)
        self.slot = nn.Embedding(cfg.view_len, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, cfg.vocab, bias=False)
        self.apply(self._init)
        for b in self.blocks:  # scaled residual init
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(b.w2.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
        self._masks: dict = {}
        self._rope: dict = {}

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def _mask(self, S, device):
        key = (S, str(device))
        if key not in self._masks:
            V = self.cfg.view_len
            i = torch.arange(S, device=device)[:, None]
            j = torch.arange(S, device=device)[None, :]
            self._masks[key] = (j < V) | (j <= i)  # bool [S,S], True = attend
        return self._masks[key]

    def _rope_for(self, S, device):
        key = (S, str(device))
        if key not in self._rope:
            self._rope[key] = rope_cache(S, self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_base, device)
        return self._rope[key]

    def forward(self, view: torch.Tensor, hist: torch.Tensor, hist_len: torch.Tensor) -> torch.Tensor:
        """view [B,V] color tokens, hist [B,L] tokens (right-padded), hist_len [B].
        Returns logits [B,vocab] for the next action token."""
        B, V = view.shape
        assert V == self.cfg.view_len
        # pad total length to a multiple of 8: keeps SDPA on the fast (aligned-mask) kernel and
        # limits the number of distinct shapes seen by the allocator.
        pad = (-(V + hist.shape[1])) % 8
        if pad:
            hist = F.pad(hist, (0, pad))
        x = torch.cat([self.tok(view) + self.slot.weight[None, :V], self.tok(hist)], dim=1)
        S = x.shape[1]
        cos, sin = self._rope_for(S, x.device)
        mask = self._mask(S, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        idx = (V - 1 + hist_len).clamp(max=S - 1)
        h = x[torch.arange(B, device=x.device), idx]
        return self.head(self.norm(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def save_ckpt(path, model: CubeLM, extra: dict | None = None):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), "extra": extra or {}}, path)


def load_ckpt(path, device="cuda") -> CubeLM:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = CubeLM(ModelConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.extra = ck.get("extra", {})
    return m
