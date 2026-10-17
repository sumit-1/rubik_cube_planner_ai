"""Vocabulary. Action tokens drive the cube (moves, rotations) or do nothing (scratch)."""
from __future__ import annotations

import numpy as np

from . import cube as C

N_SCRATCH = 24

PAD = 0
DONE = 1
MOVE_BASE = 2
ROT_BASE = MOVE_BASE + len(C.MOVES)  # 20
SCRATCH_BASE = ROT_BASE + len(C.ROTS)  # 29
COLOR_BASE = SCRATCH_BASE + N_SCRATCH  # 53
N_ACTIONS = COLOR_BASE  # ids 0..52 are valid action/output tokens
VOCAB = 64  # padded

NAMES: list[str] = (
    ["<pad>", "<done>"]
    + list(C.MOVES)
    + list(C.ROTS)
    + [f"<s{i}>" for i in range(N_SCRATCH)]
    + [f"c{i}" for i in range(6)]
)
NAMES += [f"<unused{i}>" for i in range(VOCAB - len(NAMES))]
TOK = {n: i for i, n in enumerate(NAMES)}

MOVE_IDS = np.arange(MOVE_BASE, MOVE_BASE + len(C.MOVES))
ROT_IDS = np.arange(ROT_BASE, ROT_BASE + len(C.ROTS))
SCRATCH_IDS = np.arange(SCRATCH_BASE, SCRATCH_BASE + N_SCRATCH)
CUBE_IDS = np.concatenate([MOVE_IDS, ROT_IDS])

# Token id -> sticker permutation (identity for non-cube tokens).
PERM_TABLE = np.tile(np.arange(54), (VOCAB, 1))
for _n in C.CUBE_ACTIONS:
    PERM_TABLE[TOK[_n]] = C.PERM[_n]


def is_cube_token(t: int) -> bool:
    return MOVE_BASE <= t < SCRATCH_BASE


def is_scratch(t: int) -> bool:
    return SCRATCH_BASE <= t < COLOR_BASE


def tok(name: str) -> int:
    return TOK[name]


def inv_token(t: int) -> int:
    return TOK[C.invert_move(NAMES[t])]


def encode_alg(alg) -> list[int]:
    return [TOK[m] for m in C.parse_alg(alg)]


def decode(ids) -> list[str]:
    return [NAMES[int(i)] for i in ids]


def view_tokens(state: np.ndarray, mode: str = "ufr") -> np.ndarray:
    """Color tokens for the observation (27 for 'ufr', 54 for 'full')."""
    n = 27 if mode == "ufr" else 54
    return state[:n].astype(np.int64) + COLOR_BASE


def view_len(mode: str) -> int:
    return 27 if mode == "ufr" else 54
