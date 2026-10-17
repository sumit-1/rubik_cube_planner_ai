"""OLL teacher: orient the last layer in one look (canonical frame, F2L solved).

For each of the (at most 216) orientation patterns we find a short <R,U,F> algorithm by
meet-in-the-middle search on a *pattern* (label) state: stickers of F2L pieces keep their
exact color, LL stickers are collapsed to {U-colored, don't-care}. Results are cached.
"""
from __future__ import annotations

import json
import operator
import os

import numpy as np

from .. import cube as C
from .pdb import CACHE_DIR

LL_PIECES = [("UF", 2), ("UR", 2), ("UB", 2), ("UL", 2), ("UFR", 3), ("UFL", 3),
             ("UBL", 3), ("UBR", 3)]
LL_STICKERS = sorted({s for name, _ in LL_PIECES for s in C.piece_of(name)})
_LL_MASK = np.zeros(54, dtype=bool)
_LL_MASK[LL_STICKERS] = True

GEN_IDX = [C.MOVES.index(m) for m in ("U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2")]
GEN = [C.MOVES[i] for i in GEN_IDX]
_IG = [operator.itemgetter(*C.PERM[m].tolist()) for m in GEN]
_FACE_OF = [m[0] for m in GEN]
_INV = {i: GEN.index(C.invert_move(m)) for i, m in enumerate(GEN)}


def ll_labels(state: np.ndarray) -> bytes:
    lab = state.astype(np.uint8).copy()
    ll = _LL_MASK
    lab[ll] = np.where(state[ll] == 0, 0, 6)
    return lab.tobytes()


_GOAL = ll_labels(C.SOLVED)
_BWD: dict | None = None
BWD_DEPTH = 6


def _bfs_layers(start: bytes, depth: int):
    """Yield (level, dict state -> (parent_state, move_idx)) for the frontier at each level."""
    seen = {start: None}
    frontier = [(start, -1)]
    yield 0, {start: None}, seen
    for lvl in range(1, depth + 1):
        new = {}
        for st, lastm in frontier:
            lf = _FACE_OF[lastm] if lastm >= 0 else None
            for mi, ig in enumerate(_IG):
                if _FACE_OF[mi] == lf:
                    continue
                ns = bytes(ig(st))
                if ns in seen:
                    continue
                seen[ns] = (st, mi)
                new[ns] = (st, mi)
        frontier = [(s, v[1]) for s, v in new.items()]
        yield lvl, new, seen


def _backward() -> dict:
    global _BWD
    if _BWD is None:
        seen = None
        for _, _, seen in _bfs_layers(_GOAL, BWD_DEPTH):
            pass
        _BWD = seen
    return _BWD


def _path(seen: dict, st: bytes) -> list[int]:
    out = []
    while seen[st] is not None:
        st, mi = seen[st]
        out.append(mi)
    return out[::-1]


def mitm(start: bytes, max_fwd: int = 7) -> list[str]:
    bwd = _backward()
    if start in bwd:
        p = _path(bwd, start)  # goal --p--> start ; so start --inv(rev p)--> goal
        return [GEN[_INV[m]] for m in reversed(p)]
    best = None
    for lvl, new, seen in _bfs_layers(start, max_fwd):
        if lvl == 0:
            continue
        for st in new:
            if st in bwd:
                f = _path(seen, st)
                b = _path(bwd, st)
                alg = [GEN[m] for m in f] + [GEN[_INV[m]] for m in reversed(b)]
                if best is None or len(alg) < len(best):
                    best = alg
        if best is not None:
            return best
    raise RuntimeError("no OLL alg found within depth limit")


_CACHE: dict[str, list[str]] | None = None
_CACHE_PATH = os.path.join(CACHE_DIR, "oll.json")


def _cache() -> dict:
    global _CACHE
    if _CACHE is None:
        if os.path.exists(_CACHE_PATH):
            _CACHE = json.load(open(_CACHE_PATH))
        else:
            _CACHE = {}
    return _CACHE


def oll_case_key(state: np.ndarray) -> str:
    return "".join("1" if state[s] == 0 else "0" for s in LL_STICKERS)


def oll_oriented(state: np.ndarray) -> bool:
    return bool(np.all(state[:9] == 0))


def solve_oll(state: np.ndarray) -> list[str]:
    """Algorithm (face turns) orienting the LL; state canonical with F2L solved."""
    if oll_oriented(state):
        return []
    c = _cache()
    key = oll_case_key(state)
    if key not in c:
        c[key] = mitm(ll_labels(state))
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(c, open(_CACHE_PATH, "w"))
    return list(c[key])
