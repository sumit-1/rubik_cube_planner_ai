"""GPU-resident step dataset with on-the-fly (view, history) batch construction."""
from __future__ import annotations

import numpy as np
import torch

from .. import tokenizer as T
from . import gen


class StepData:
    def __init__(self, path: str, device="cuda", holdout_frac: float = 0.01, view_mode="ufr"):
        d = gen.load(path)
        self.device = torch.device(device)
        self.view_mode = view_mode
        self.V = T.view_len(view_mode)
        self.states = torch.as_tensor(d["states"], device=self.device)  # uint8 [M,54]
        self.tokens = torch.as_tensor(d["tokens"].astype(np.int64), device=self.device)
        self.wcode = torch.as_tensor(d["weights"], device=self.device)
        self.stage = torch.as_tensor(d["stages"], device=self.device)
        self.start = torch.as_tensor(d["traj_start"].astype(np.int64), device=self.device)
        self.M = self.tokens.shape[0]
        cut = int(self.M * (1 - holdout_frac))
        # align cut to trajectory boundary
        while cut < self.M and self.start[cut].item() != cut:
            cut += 1
        self.cut = cut
        w = self.wcode > 0
        ar = torch.arange(self.M, device=self.device)
        self.train_idx = ar[w & (ar < cut)]
        self.val_idx = ar[w & (ar >= cut)]
        self.name = path

    def batch(self, idx: torch.Tensor, w_scratch: float = 0.25):
        t = idx - self.start[idx]
        L = int(t.max().item()) if idx.numel() else 0
        ar = torch.arange(L, device=self.device)[None, :]
        gather = (self.start[idx][:, None] + ar).clamp(max=self.M - 1)
        hist = self.tokens[gather]
        hist = torch.where(ar < t[:, None], hist, torch.zeros_like(hist))
        view = self.states[idx][:, : self.V].long() + T.COLOR_BASE
        target = self.tokens[idx]
        wc = self.wcode[idx]
        w = torch.where(wc == gen.W_FULL, 1.0, torch.where(wc == gen.W_SCRATCH, w_scratch, 0.0))
        return view, hist, t, target, w.float(), self.stage[idx]

    def sample(self, B: int, split: str = "train", w_scratch: float = 0.25):
        pool = self.train_idx if split == "train" else self.val_idx
        idx = pool[torch.randint(0, pool.numel(), (B,), device=self.device)]
        return self.batch(idx, w_scratch)
