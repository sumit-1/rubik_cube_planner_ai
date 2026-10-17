"""Sticker-level Rubik's cube simulator.

State: numpy uint8 array of 54 colors in face order U R F D L B (9 stickers per face,
row-major, reading frame defined in `_FRAMES`). Every action (face turn or whole-cube
rotation) is a fixed permutation of the 54 sticker locations, so a rotation is just
another permutation and "the view" is always the U/F/R faces of the *current* state.

Convention: ``new_state = state[PERM[action]]`` (PERM[a][dest] = src).
"""
from __future__ import annotations

import numpy as np

FACES = "URFDLB"
FACE_IDX = {f: i for i, f in enumerate(FACES)}
_NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}
# (up, right) reading vectors for each face, as seen from outside the cube.
_FRAMES = {
    "U": ((0, 0, -1), (1, 0, 0)),
    "R": ((0, 1, 0), (0, 0, -1)),
    "F": ((0, 1, 0), (1, 0, 0)),
    "D": ((0, 0, 1), (1, 0, 0)),
    "L": ((0, 1, 0), (0, 0, 1)),
    "B": ((0, 1, 0), (-1, 0, 0)),
}

MOVES = [f + m for f in "URFDLB" for m in ("", "'", "2")]  # 18 face turns
ROTS = [r + m for r in "xyz" for m in ("", "'", "2")]  # 9 whole-cube rotations
CUBE_ACTIONS = MOVES + ROTS


def _v(t):
    return np.array(t, dtype=np.int64)


# ---- geometry ---------------------------------------------------------------
STICKER_POS = np.zeros((54, 3), dtype=np.int64)  # cubie coordinates
STICKER_NRM = np.zeros((54, 3), dtype=np.int64)  # outward normal
for _fi, _f in enumerate(FACES):
    _n = _v(_NORMALS[_f])
    _up, _right = (_v(a) for a in _FRAMES[_f])
    for _r in range(3):
        for _c in range(3):
            _i = _fi * 9 + _r * 3 + _c
            STICKER_POS[_i] = _n + _up * (1 - _r) + _right * (_c - 1)
            STICKER_NRM[_i] = _n

_LOOKUP = {(tuple(STICKER_POS[i]), tuple(STICKER_NRM[i])): i for i in range(54)}


def _rot_cw(vec: np.ndarray, axis: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate `vec` clockwise (seen from outside along +axis) by quarter_turns."""
    v = vec.copy()
    for _ in range(quarter_turns % 4):
        v = -np.cross(axis, v) + axis * int(axis @ v)
    return v


def _make_perm(axis: np.ndarray, quarter_turns: int, layer_only: bool) -> np.ndarray:
    perm = np.arange(54)
    for i in range(54):
        if layer_only and STICKER_POS[i] @ axis != 1:
            continue
        p2 = _rot_cw(STICKER_POS[i], axis, quarter_turns)
        n2 = _rot_cw(STICKER_NRM[i], axis, quarter_turns)
        dest = _LOOKUP[(tuple(p2), tuple(n2))]
        perm[dest] = i
    return perm


PERM: dict[str, np.ndarray] = {}
_MOD_TURNS = {"": 1, "'": 3, "2": 2}
for _f in FACES:
    for _m, _q in _MOD_TURNS.items():
        PERM[_f + _m] = _make_perm(_v(_NORMALS[_f]), _q, True)
for _r, _f in zip("xyz", "RUF"):
    for _m, _q in _MOD_TURNS.items():
        PERM[_r + _m] = _make_perm(_v(_NORMALS[_f]), _q, False)

# (dest) arrays: DEST[a][src] = location where the sticker at src ends up.
DEST: dict[str, np.ndarray] = {}
for _k, _p in PERM.items():
    _d = np.empty(54, dtype=np.int64)
    _d[_p] = np.arange(54)
    DEST[_k] = _d

SOLVED = np.repeat(np.arange(6, dtype=np.uint8), 9)

# ---- pieces -------------------------------------------------------------------
_by_cubie: dict[tuple, list[int]] = {}
for _i in range(54):
    _by_cubie.setdefault(tuple(STICKER_POS[_i]), []).append(_i)
EDGES = [tuple(sorted(v)) for k, v in _by_cubie.items() if sum(abs(c) for c in k) == 2]
CORNERS = [tuple(sorted(v)) for k, v in _by_cubie.items() if sum(abs(c) for c in k) == 3]
EDGES.sort()
CORNERS.sort()
EDGE_STICKERS = np.array([s for e in EDGES for s in e], dtype=np.int64)  # 24
CORNER_STICKERS = np.array([s for c in CORNERS for s in c], dtype=np.int64)  # 24
EDGE24 = -np.ones(54, dtype=np.int64)
EDGE24[EDGE_STICKERS] = np.arange(24)
CORNER24 = -np.ones(54, dtype=np.int64)
CORNER24[CORNER_STICKERS] = np.arange(24)
assert len(EDGES) == 12 and len(CORNERS) == 8


def piece_of(stickers_by_face: str) -> tuple:
    """Return the (sorted) sticker-index tuple of the piece named like 'DF' or 'DFR'."""
    cubie = np.zeros(3, dtype=np.int64)
    for ch in stickers_by_face:
        cubie += _v(_NORMALS[ch])
    return tuple(sorted(_by_cubie[tuple(cubie)]))


def sticker_of(piece: str, face: str) -> int:
    """Home sticker index of the `face`-facing sticker of piece (e.g. 'DFR','D')."""
    cubie = np.zeros(3, dtype=np.int64)
    for ch in piece:
        cubie += _v(_NORMALS[ch])
    return _LOOKUP[(tuple(cubie), _NORMALS[face])]


# ---- algorithm parsing -------------------------------------------------------------
_EXTRA = {}  # slice moves & wide moves expressed through compositions


def _compose(*names: str) -> np.ndarray:
    p = np.arange(54)
    for n in names:
        p = p[PERM[n]]
    return p


# slices (M follows L, E follows D, S follows F); wide moves (r = R + M' etc.)
for _m, _q in _MOD_TURNS.items():
    _suf = {"": "", "'": "'", "2": "2"}[_m]
    _inv = {"": "'", "'": "", "2": "2"}[_m]
    PERM["M" + _m] = _compose("R" + _suf, "L" + _inv, "x" + _inv)
    PERM["E" + _m] = _compose("U" + _suf, "D" + _inv, "y" + _inv)
    PERM["S" + _m] = _compose("F" + _inv, "B" + _suf, "z" + _suf)
    PERM["r" + _m] = _compose("L" + _suf, "x" + _suf)
    PERM["l" + _m] = _compose("R" + _suf, "x" + _inv)
    PERM["u" + _m] = _compose("D" + _suf, "y" + _suf)
    PERM["d" + _m] = _compose("U" + _suf, "y" + _inv)
    PERM["f" + _m] = _compose("B" + _suf, "z" + _suf)
    PERM["b" + _m] = _compose("F" + _suf, "z" + _inv)


def parse_alg(alg: str | list[str]) -> list[str]:
    toks = alg.replace("’", "'").split() if isinstance(alg, str) else list(alg)
    out = []
    for t in toks:
        if t.endswith("2'"):
            t = t[:-2] + "2"
        if t not in PERM:
            raise ValueError(f"unknown move {t!r}")
        out.append(t)
    return out


def invert_move(m: str) -> str:
    if m.endswith("2"):
        return m
    if m.endswith("'"):
        return m[:-1]
    return m + "'"


def invert_alg(alg: str | list[str]) -> list[str]:
    return [invert_move(m) for m in reversed(parse_alg(alg))]


def apply_move(state: np.ndarray, move: str) -> np.ndarray:
    return state[PERM[move]]


def apply_alg(state: np.ndarray, alg: str | list[str]) -> np.ndarray:
    for m in parse_alg(alg):
        state = state[PERM[m]]
    return state


def random_scramble(rng: np.random.Generator, n: int) -> list[str]:
    """n random face turns with no same-face repeats and no immediate axis-undo."""
    out: list[str] = []
    last, last2 = -1, -1
    axis = {0: 0, 3: 0, 1: 1, 4: 1, 2: 2, 5: 2}  # URFDLB face -> axis (U/D,R/L,F/B)
    # face order in MOVES is U R F D L B
    while len(out) < n:
        f = int(rng.integers(6))
        if f == last:
            continue
        if last2 >= 0 and axis[f] == axis[last] and f == last2:
            continue
        out.append(MOVES[f * 3 + int(rng.integers(3))])
        last2, last = last, f
    return out


# ---- views ---------------------------------------------------------------------------
VIEW_UFR = np.arange(27)  # U, R, F faces are faces 0..2 -> first 27 stickers


def view_ufr(state: np.ndarray) -> np.ndarray:
    return state[:27]


def is_solved(state: np.ndarray) -> bool:
    """Solved up to whole-cube orientation (each face uniform)."""
    return bool(np.all(state.reshape(6, 9) == state.reshape(6, 9)[:, :1]))


def is_solved_oriented(state: np.ndarray) -> bool:
    return bool(np.array_equal(state, SOLVED))


# ---- orientation handling ---------------------------------------------------------------
_ROT_BASE = ["x", "y", "z", "x'", "y'", "z'"]


def _build_orient_table():
    table = {tuple(SOLVED[[4, 13, 22, 31, 40, 49]]): (np.arange(54), [])}
    frontier = [(np.arange(54), [])]
    while frontier:
        nxt = []
        for p, seq in frontier:
            for r in _ROT_BASE:
                p2 = p[PERM[r]]
                key = tuple(SOLVED[p2][[4, 13, 22, 31, 40, 49]])
                if key not in table:
                    table[key] = (p2, seq + [r])
                    nxt.append((p2, seq + [r]))
        frontier = nxt
    return table


# Maps a tuple of 6 center colors -> (perm, rotation sequence) bringing the cube to the
# canonical orientation (centers U R F D L B == 0..5), i.e. canonical = state[perm].
# NOTE: key is computed from `SOLVED[p]` which equals the center colors after rotating a
# solved cube by p; we invert it below.
_ORIENT_RAW = _build_orient_table()
CENTER_IDX = [4, 13, 22, 31, 40, 49]
ORIENT_TO_CANON: dict[tuple, tuple[np.ndarray, list[str]]] = {}
for _key, (_p, _seq) in _ORIENT_RAW.items():
    # A solved cube rotated by _seq has centers `_key`; to undo it apply the inverse.
    ORIENT_TO_CANON[_key] = (None, invert_alg(_seq))  # type: ignore[assignment]
    _inv = np.arange(54)
    for _m in invert_alg(_seq):
        _inv = _inv[PERM[_m]]
    ORIENT_TO_CANON[_key] = (_inv, invert_alg(_seq))
assert len(ORIENT_TO_CANON) == 24


def canonicalize(state: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Rotate whole cube so centers are in standard places. Returns (state, rotations)."""
    key = tuple(int(c) for c in state[CENTER_IDX])
    perm, seq = ORIENT_TO_CANON[key]
    return state[perm], seq
