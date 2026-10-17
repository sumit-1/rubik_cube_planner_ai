"""CFOP-staged teacher. Plans are expressed in the *canonical* frame (centers U R F D L B =
0..5). `to_current_token` converts a canonical-frame move into the face turn that is correct
in whatever orientation the (rotated) cube currently has."""
from __future__ import annotations

import numpy as np

from .. import cube as C
from . import cross, f2l, oll, pll

STAGES = ["cross", "f2l0", "f2l1", "f2l2", "f2l3", "oll", "pll"]
STAGE_GROUP = {"cross": 1, "f2l0": 2, "f2l1": 2, "f2l2": 2, "f2l3": 2, "oll": 3, "pll": 4}


def solve_stage(state: np.ndarray, stage: str) -> list[str]:
    """Moves (canonical frame) completing `stage` assuming earlier stages are done."""
    if stage == "cross":
        return cross.solve_cross(state)
    if stage.startswith("f2l"):
        return f2l.solve_pair(state, int(stage[3]))
    if stage == "oll":
        return oll.solve_oll(state)
    if stage == "pll":
        return pll.solve_pll(state)
    raise ValueError(stage)


def advance(state: np.ndarray, stages: list[str]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Run teacher over the stages; return final state and [(stage, move), ...]."""
    plan = []
    for st in stages:
        mv = solve_stage(state, st)
        state = C.apply_alg(state, mv)
        plan += [(st, m) for m in mv]
    return state, plan


def full_plan(state: np.ndarray) -> list[tuple[str, str]]:
    return advance(state, STAGES)[1]


def to_current_token(canon_move: str, current_state: np.ndarray) -> str:
    """Face turn in the current (possibly rotated) frame that equals `canon_move`."""
    fi = C.FACE_IDX[canon_move[0]]
    centers = current_state[C.CENTER_IDX]
    loc = int(np.flatnonzero(centers == fi)[0])
    return C.FACES[loc] + canon_move[1:]
