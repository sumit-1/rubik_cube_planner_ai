"""PLL teacher: permute the (already oriented) last layer.

A library of standard PLL algorithms is verified in the simulator (rotation-free
conversion; must leave F2L and LL orientation intact). The teacher then runs Dijkstra over
the <= 288 LL permutation states with edges = {U turns, library algs, library inverses}
so every state gets a shortest solution in this macro space (including the final AUF).
"""
from __future__ import annotations

import heapq
import json
import os

import numpy as np

from .. import cube as C
from .pdb import CACHE_DIR

PLL_ALGS = {
    "Aa": "x R' U R' D2 R U' R' D2 R2",
    "Ab": "x R2 D2 R U R' D2 R U' R",
    "E": "x' R U' R' D R U R' D' R U R' D R U' R' D'",
    "F": "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "Ga": "R2 U R' U R' U' R U' R2 D U' R' U R D'",
    "Gb": "R' U' R U D' R2 U R' U R U' R U' R2 D",
    "Gc": "R2 U' R U' R U R' U R2 D' U R U' R' D",
    "Gd": "R U R' U' D R2 U' R U' R' U R' U R2 D'",
    "H": "R2 U2 R U2 R2 U2 R2 U2 R U2 R2",
    "Ja": "R' U L' U2 R U' R' U2 R L",
    "Jb": "R U R' F' R U R' U' R' F R2 U' R'",
    "Na": "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "Nb": "R' U R U' R' F' U' F R U R' F R' F' R U' R",
    "Ra": "R U' R' U' R U R D R' U' R D' R' U2 R'",
    "Rb": "R' U2 R U2 R' F R U R' U' R' F' R2",
    "T": "R U R' U' R' F R2 U' R' U' R U R' F'",
    "Ua": "R U' R U R U R U' R' U' R2",
    "Ub": "R2 U R U R' U' R' U' R' U R'",
    "V": "R' U R' U' y R' F' R2 U' R' U R' F R F",
    "Y": "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "Z": "R' U' R U' R U R U' R' U R U R2 U' R'",
}


def to_face_turns(alg: str) -> list[str]:
    """Convert an alg containing x/y/z rotations into an equivalent rotation-free sequence
    (expressed in the original frame, trailing rotation dropped)."""
    cur2orig = {f: f for f in C.FACES}
    out = []
    for t in C.parse_alg(alg):
        if t[0] in "xyz":
            p = C.PERM[t]
            new = {}
            for fi, f in enumerate(C.FACES):
                old_loc = int(p[C.CENTER_IDX[fi]])
                new[f] = cur2orig[C.FACES[old_loc // 9]]
            cur2orig = new
        else:
            out.append(cur2orig[t[0]] + t[1:])
    return out


def _is_pll_alg(seq: list[str]) -> bool:
    s = C.apply_alg(C.SOLVED.copy(), seq)
    if not np.all(s[:9] == 0):
        return False
    for p in ("DF", "DR", "DB", "DL", "FR", "FL", "BL", "BR", "DFR", "DFL", "DBL", "DBR"):
        for st in C.piece_of(p):
            if s[st] != C.SOLVED[st]:
                return False
    return not np.array_equal(s, C.SOLVED)


def build_library() -> dict[str, list[str]]:
    lib = {}
    for name, alg in PLL_ALGS.items():
        try:
            seq = to_face_turns(alg)
        except ValueError:
            continue
        if _is_pll_alg(seq):
            lib[name] = seq
    return lib


LIBRARY = build_library()
_MACROS: list[tuple[str, list[str], np.ndarray]] = []


def _macros():
    if not _MACROS:
        for m in ("U", "U'", "U2"):
            _MACROS.append((m, [m], C.PERM[m]))
        for name, seq in LIBRARY.items():
            for variant, sq in ((name, seq), (name + "_inv", C.invert_alg(seq))):
                p = np.arange(54)
                for mv in sq:
                    p = p[C.PERM[mv]]
                _MACROS.append((variant, sq, p))
    return _MACROS


_CACHE_PATH = os.path.join(CACHE_DIR, "pll.json")
_CACHE: dict | None = None


def pll_key(state: np.ndarray) -> str:
    return "".join(str(int(state[s])) for s in range(54) if s >= 9 and s < 54)


def solve_pll(state: np.ndarray) -> list[str]:
    """Shortest macro solution (face turns) from an OLL-solved canonical state to solved."""
    global _CACHE
    if _CACHE is None:
        _CACHE = json.load(open(_CACHE_PATH)) if os.path.exists(_CACHE_PATH) else {}
    key = pll_key(state)
    if key in _CACHE:
        return list(_CACHE[key])
    goal = C.SOLVED.tobytes()
    start = state.astype(np.uint8)
    dist = {start.tobytes(): 0}
    prev: dict[bytes, tuple[bytes, list[str]]] = {}
    heap = [(0, start.tobytes())]
    macros = _macros()
    found = None
    while heap:
        d, sb = heapq.heappop(heap)
        if d > dist[sb]:
            continue
        if sb == goal:
            found = sb
            break
        s = np.frombuffer(sb, dtype=np.uint8)
        for _, seq, p in macros:
            nb = s[p].tobytes()
            nd = d + len(seq)
            if nd < dist.get(nb, 1 << 30):
                dist[nb] = nd
                prev[nb] = (sb, seq)
                heapq.heappush(heap, (nd, nb))
    if found is None:
        raise RuntimeError("PLL state unreachable with library")
    out: list[list[str]] = []
    cur = found
    while cur in prev:
        cur, seq = prev[cur]
        out.append(seq)
    moves = [m for seq in reversed(out) for m in seq]
    moves = simplify(moves)
    _CACHE[key] = moves
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(_CACHE, open(_CACHE_PATH, "w"))
    return moves


def simplify(moves: list[str]) -> list[str]:
    """Merge consecutive same-face turns."""
    stack: list[tuple[str, int]] = []
    q = {"": 1, "'": 3, "2": 2}
    for m in moves:
        f, k = m[0], q[m[1:]]
        if stack and stack[-1][0] == f:
            k = (stack[-1][1] + k) % 4
            stack.pop()
        if k:
            stack.append((f, k))
    inv = {1: "", 2: "2", 3: "'"}
    return [f + inv[k] for f, k in stack]
