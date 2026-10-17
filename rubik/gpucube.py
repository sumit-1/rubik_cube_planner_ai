"""Batched torch cube environment (runs on GPU). State: [B,54] int64 colors."""
from __future__ import annotations

import numpy as np
import torch

from . import cube as C
from . import tokenizer as T


class BatchCube:
    def __init__(self, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.perm = torch.as_tensor(T.PERM_TABLE, device=self.device)  # [V,54]
        self.solved = torch.as_tensor(C.SOLVED.astype(np.int64), device=self.device)

    def solved_batch(self, b: int) -> torch.Tensor:
        return self.solved.unsqueeze(0).repeat(b, 1)

    def step(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Apply one action token per cube. Non-cube tokens leave the cube untouched."""
        return state.gather(1, self.perm[tokens])

    def run(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Apply a [B,L] token sequence, return all intermediate states [B,L+1,54]."""
        outs = [state]
        for i in range(tokens.shape[1]):
            state = self.step(state, tokens[:, i])
            outs.append(state)
        return torch.stack(outs, 1)

    def oriented_solved(self, state: torch.Tensor) -> torch.Tensor:
        return (state == self.solved).all(1)

    def solved_any_orientation(self, state: torch.Tensor) -> torch.Tensor:
        s = state.view(-1, 6, 9)
        return (s == s[:, :, :1]).all(2).all(1)

    def view(self, state: torch.Tensor, mode: str = "ufr") -> torch.Tensor:
        n = T.view_len(mode)
        return state[:, :n] + T.COLOR_BASE
