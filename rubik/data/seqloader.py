"""GPU-resident trajectory loader for the recurrent model (whole trajectories, padded)."""
from __future__ import annotations

import numpy as np
import torch

from .. import tokenizer as T
from . import gen


class TrajData:
    def __init__(self, path: str, device="cuda", holdout_frac=0.01, max_len=200):
        d = gen.load(path)
        self.device = torch.device(device)
        self.states = torch.as_tensor(d["states"], device=self.device)
        self.tokens = torch.as_tensor(d["tokens"].astype(np.int64), device=self.device)
        self.wcode = torch.as_tensor(d["weights"], device=self.device)
        self.stage = torch.as_tensor(d["stages"], device=self.device)
        ts = d["traj_start"].astype(np.int64)
        starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
        lens = np.diff(np.r_[starts, len(ts)])
        keep = lens <= max_len
        starts, lens = starts[keep], lens[keep]
        n_val = max(1, int(len(starts) * holdout_frac))
        self.train_start = torch.as_tensor(starts[:-n_val], device=self.device)
        self.train_len = torch.as_tensor(lens[:-n_val], device=self.device)
        self.val_start = torch.as_tensor(starts[-n_val:], device=self.device)
        self.val_len = torch.as_tensor(lens[-n_val:], device=self.device)
        self.M = len(ts)
        self.name = path

    def sample(self, B: int, split="train"):
        st, ln = (self.train_start, self.train_len) if split == "train" else (self.val_start, self.val_len)
        idx = torch.randint(0, st.numel(), (B,), device=self.device)
        s, l = st[idx], ln[idx]
        Tm = int(l.max().item())
        ar = torch.arange(Tm, device=self.device)[None]
        valid = ar < l[:, None]
        g = (s[:, None] + ar).clamp(max=self.M - 1)
        states = self.states[g]  # [B,T,54] uint8
        tokens = torch.where(valid, self.tokens[g], torch.zeros_like(g))
        w = torch.where(valid & (self.wcode[g] == gen.W_FULL), 1.0, 0.0)
        stage = self.stage[g]
        return states, tokens, w, stage, valid
