"""Cross teacher: optimal D-cross solve (canonical frame) via PDB-guided IDA*."""
from __future__ import annotations

import numpy as np

from . import pdb


def solve_cross(state: np.ndarray) -> list[str]:
    """`state` must be in canonical orientation. Returns face turns solving the D cross."""
    pos = pdb.track_positions(state)
    return pdb.ida_solve(pos, (False, False, False, False), need_cross=True, maxd=10)


def cross_solved(state: np.ndarray) -> bool:
    pos = pdb.track_positions(state)
    return bool(np.array_equal(pos[:4], pdb.HOMES24[:4]))
