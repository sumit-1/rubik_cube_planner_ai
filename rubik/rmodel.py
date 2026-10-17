"""Recurrent duplex transformer.

Per step the model reads  [K memory slots | view tokens | last-H action tokens | act query |
K write queries]  with full (bidirectional) attention and emits two things:
  * action logits (move / rotation / done), from the act-query position
  * K new memory vectors (gated update of the old memory), from the write-query positions
The memory is a continuous feedback loop (LSTM-like): it is trained end-to-end by the action
loss through truncated BPTT, so no memory content is ever supervised or engineered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as T
from .model import RMSNorm


@dataclass
class RecConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ff_mult: float = 8 / 3
    mem_slots: int = 16
    window: int = 16
    view_len: int = 27
    vocab: int = T.VOCAB
    use_view_head: bool = True


class RBlock(nn.Module):
    def __init__(self, cfg: RecConfig):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = RMSNorm(d)
        ff = (int(cfg.ff_mult * d) + 31) // 32 * 32
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, D))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class RecurrentCubeLM(nn.Module):
    is_recurrent = True

    def __init__(self, cfg: RecConfig):
        super().__init__()
        self.cfg = cfg
        d, K, V, H = cfg.d_model, cfg.mem_slots, cfg.view_len, cfg.window
        self.tok = nn.Embedding(cfg.vocab, d)
        self.view_pos = nn.Parameter(torch.randn(V, d) * 0.02)
        self.recent_pos = nn.Parameter(torch.randn(H, d) * 0.02)
        self.mem_pos = nn.Parameter(torch.randn(K, d) * 0.02)
        self.act_q = nn.Parameter(torch.randn(1, d) * 0.02)
        self.write_q = nn.Parameter(torch.randn(K, d) * 0.02)
        self.mem0 = nn.Parameter(torch.randn(K, d) * 0.02)
        self.mem_norm = RMSNorm(d)
        self.blocks = nn.ModuleList([RBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, cfg.vocab, bias=False)
        self.w_norm = RMSNorm(d)
        self.w_cand = nn.Linear(d, d, bias=False)
        self.w_gate = nn.Linear(d, d)
        # Next-view prediction head: per-slot cross-attention over the K memory vectors,
        # conditioned on the candidate action token embedding.
        self.vp_norm = RMSNorm(d)
        self.vp_slot = nn.Parameter(torch.randn(V, d) * 0.02)
        self.vp_act = nn.Linear(d, d, bias=False)
        self.vp_q = nn.Linear(d, d, bias=False)
        self.vp_kv = nn.Linear(d, 2 * d, bias=False)
        self.vp_out = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 6),
        )
        self.apply(self._init)
        nn.init.constant_(self.w_gate.bias, 0.0)
        for b in self.blocks:
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(b.w2.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def init_mem(self, B: int) -> torch.Tensor:
        return self.mem0[None].expand(B, -1, -1).contiguous()

    def step(self, view: torch.Tensor, recent: torch.Tensor, mem: torch.Tensor):
        """view [B,V] color tokens, recent [B,H] last actions (PAD-left-padded),
        mem [B,K,d]. Returns (logits [B,vocab], new_mem [B,K,d])."""
        B = view.shape[0]
        K = self.cfg.mem_slots
        x = torch.cat([
            self.mem_norm(mem).to(self.tok.weight.dtype) + self.mem_pos,
            self.tok(view) + self.view_pos,
            self.tok(recent) + self.recent_pos,
            self.act_q.expand(B, -1, -1),
            self.write_q.expand(B, -1, -1),
        ], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x[:, -K - 1])
        hw = self.w_norm(x[:, -K:])
        gate = torch.sigmoid(self.w_gate(hw).float())
        new_mem = gate * mem.float() + (1 - gate) * self.w_cand(hw).float()
        return logits, new_mem

    def predict_next_view(self, new_mem: torch.Tensor, action_tok: torch.Tensor) -> torch.Tensor:
        """Given updated memory [B,K,d] and action token [B], predict next 3-face view logits [B,V,6]."""
        B, K, D = new_mem.shape
        V = self.cfg.view_len
        H = self.cfg.n_heads
        m = self.vp_norm(new_mem.to(self.tok.weight.dtype))  # [B,K,D]
        a = self.vp_act(self.tok(action_tok))[:, None, :]    # [B,1,D]
        q = self.vp_q(self.vp_slot[None, :, :] + a).view(B, V, H, D // H).transpose(1, 2)
        k, v = self.vp_kv(m).view(B, K, 2, H, D // H).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, V, D)
        return self.vp_out(attn + a)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def save_rckpt(path, model: RecurrentCubeLM, extra: dict | None = None):
    ex = dict(extra or {})
    ex["kind"] = "rnn"
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), "extra": ex, "kind": "rnn"}, path)


def load_rckpt(path, device="cuda") -> RecurrentCubeLM:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = RecurrentCubeLM(RecConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"], strict=False)
    m.extra = ck.get("extra", {})
    return m


def load_any(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") == "rnn":
        return load_rckpt(path, device)
    from .model import load_ckpt
    return load_ckpt(path, device)
