"""F2L teacher: solves pairs in slot order FR, FL, BL, BR (canonical frame) keeping cross
and previously solved pairs. Each pair is solved with PDB-guided IDA* (cross+pair PDB)."""
from __future__ import annotations

import numpy as np

from .. import cube as C
from . import pdb

SLOT_NAMES = ["FR", "FL", "BL", "BR"]


def pairs_solved(state: np.ndarray) -> list[bool]:
    pos = pdb.track_positions(state)
    return [
        bool(pos[4 + 2 * a] == pdb.HOMES24[4 + 2 * a] and pos[5 + 2 * a] == pdb.HOMES24[5 + 2 * a])
        for a in range(4)
    ]


def next_unsolved_pair(state: np.ndarray) -> int | None:
    for a, ok in enumerate(pairs_solved(state)):
        if not ok:
            return a
    return None


def solve_pair(state: np.ndarray, a: int) -> list[str]:
    """Solve pair `a` preserving cross and pairs 0..a-1 (assumed already solved)."""
    pos = pdb.track_positions(state)
    active = tuple(i <= a for i in range(4))
    return pdb.ida_solve(pos, active, need_cross=True, maxd=16)


def solve_f2l(state: np.ndarray) -> list[tuple[int, list[str]]]:
    out = []
    s = state
    for a in range(4):
        mv = solve_pair(s, a)
        s = C.apply_alg(s, mv)
        out.append((a, mv))
    return out
