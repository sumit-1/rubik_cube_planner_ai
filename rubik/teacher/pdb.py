"""Pattern databases and numba IDA* used by the cross / F2L teachers.

Everything lives in "24-space": a tracked piece is represented by the location of its
canonical sticker among the 24 edge-sticker (or 24 corner-sticker) locations. That index
encodes slot *and* orientation, so a piece is solved iff its canonical sticker is home.
"""
from __future__ import annotations

import os
import time

import numba as nb
import numpy as np

from .. import cube as C

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")

# ---- move tables in 24-space --------------------------------------------------------------
EM = np.zeros((18, 24), dtype=np.int64)
CM = np.zeros((18, 24), dtype=np.int64)
for _mi, _m in enumerate(C.MOVES):
    for _i in range(24):
        EM[_mi, _i] = C.EDGE24[C.DEST[_m][C.EDGE_STICKERS[_i]]]
        CM[_mi, _i] = C.CORNER24[C.DEST[_m][C.CORNER_STICKERS[_i]]]
    assert (EM[_mi] >= 0).all() and (CM[_mi] >= 0).all()


def _rot_tables(rot: str, times: int):
    e = np.arange(24)
    c = np.arange(24)
    for _ in range(times):
        e = np.array([C.EDGE24[C.DEST[rot][C.EDGE_STICKERS[i]]] for i in e])
        c = np.array([C.CORNER24[C.DEST[rot][C.CORNER_STICKERS[i]]] for i in c])
    return e, c


# slot a is y^a applied to the FR slot. PM*[a] maps positions through y^-a (conj to FR).
PM_E = np.zeros((4, 24), dtype=np.int64)
PM_C = np.zeros((4, 24), dtype=np.int64)
for _a in range(4):
    PM_E[_a], PM_C[_a] = _rot_tables("y'", _a)


def _y_pow_dest(a: int) -> np.ndarray:
    d = np.arange(54)
    for _ in range(a):
        d = C.DEST["y"][d]
    return d


# ---- tracked pieces -------------------------------------------------------------------------
# tracked layout: 0..3 cross edges (DF, DR, DB, DL; canonical = D sticker),
# then for slot a: 4+2a edge, 5+2a corner.
CROSS_HOME_STICKERS = [C.sticker_of(p, "D") for p in ("DF", "DR", "DB", "DL")]
_E_BASE = C.sticker_of("FR", "F")
_K_BASE = C.sticker_of("DFR", "D")
TRACK_HOME = []  # sticker index of each tracked piece's canonical sticker
TRACK_KIND = []  # 0 edge, 1 corner
for _s in CROSS_HOME_STICKERS:
    TRACK_HOME.append(_s)
    TRACK_KIND.append(0)
for _a in range(4):
    _d = _y_pow_dest(_a)
    TRACK_HOME += [int(_d[_E_BASE]), int(_d[_K_BASE])]
    TRACK_KIND += [0, 1]
TRACK_HOME = np.array(TRACK_HOME, dtype=np.int64)
TRACK_KIND = np.array(TRACK_KIND, dtype=np.int64)
HOMES24 = np.array(
    [C.EDGE24[h] if k == 0 else C.CORNER24[h] for h, k in zip(TRACK_HOME, TRACK_KIND)],
    dtype=np.int64,
)
# For slot a, PDB digit u of the cross part corresponds to tracked cross piece SIGMA[a,u]
SIGMA = np.zeros((4, 4), dtype=np.int64)
for _a in range(4):
    _d = _y_pow_dest(_a)
    for _u in range(4):
        _target = _d[CROSS_HOME_STICKERS[_u]]
        SIGMA[_a, _u] = CROSS_HOME_STICKERS.index(int(_target))

# piece identification by colors (canonical frame, solved colors)
_PIECE_COLORS = []
for _t in range(12):
    _h = int(TRACK_HOME[_t])
    _piece = [e for e in C.EDGES if _h in e] if TRACK_KIND[_t] == 0 else [
        c for c in C.CORNERS if _h in c
    ]
    _st = _piece[0]
    _PIECE_COLORS.append((tuple(sorted(int(C.SOLVED[s]) for s in _st)), int(C.SOLVED[_h])))


def track_positions(state: np.ndarray) -> np.ndarray:
    """24-space locations of the 12 tracked canonical stickers in a canonical-frame state."""
    edge_loc = {}
    for e in C.EDGES:
        key = tuple(sorted(int(state[s]) for s in e))
        edge_loc[key] = e
    corner_loc = {}
    for c in C.CORNERS:
        key = tuple(sorted(int(state[s]) for s in c))
        corner_loc[key] = c
    pos = np.zeros(12, dtype=np.int64)
    for t in range(12):
        key, canon = _PIECE_COLORS[t]
        if TRACK_KIND[t] == 0:
            st = edge_loc[key]
            loc = [s for s in st if state[s] == canon][0]
            pos[t] = C.EDGE24[loc]
        else:
            st = corner_loc[key]
            loc = [s for s in st if state[s] == canon][0]
            pos[t] = C.CORNER24[loc]
    return pos


# ---- PDB build ---------------------------------------------------------------------------------
@nb.njit(cache=True)
def _bfs_level(dist, idxs, kinds, em, cm, level):
    n = kinds.shape[0]
    pw = np.empty(n, np.int64)
    pw[0] = 1
    for t in range(1, n):
        pw[t] = pw[t - 1] * 24
    pos = np.empty(n, np.int64)
    cnt = 0
    for ii in range(idxs.shape[0]):
        idx = idxs[ii]
        for t in range(n):
            pos[t] = (idx // pw[t]) % 24
        for m in range(18):
            new = 0
            for t in range(n):
                if kinds[t] == 0:
                    new += em[m, pos[t]] * pw[t]
                else:
                    new += cm[m, pos[t]] * pw[t]
            if dist[new] == 255:
                dist[new] = level + 1
                cnt += 1
    return cnt


def build_pdb(kinds: np.ndarray, goal_positions: np.ndarray) -> np.ndarray:
    n = len(kinds)
    size = 24 ** n
    dist = np.full(size, 255, dtype=np.uint8)
    goal = int(sum(int(p) * 24 ** t for t, p in enumerate(goal_positions)))
    dist[goal] = 0
    level = 0
    t0 = time.time()
    while True:
        idxs = np.flatnonzero(dist == level)
        if idxs.size == 0:
            break
        _bfs_level(dist, idxs, kinds, EM, CM, level)
        level += 1
    print(f"[pdb] built n={n} depth={level} in {time.time() - t0:.1f}s", flush=True)
    return dist


def _load_or_build(name: str, kinds, goal) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    arr = build_pdb(np.asarray(kinds, dtype=np.int64), np.asarray(goal, dtype=np.int64))
    np.save(path, arr)
    return arr


_CROSS_PDB = None
_PAIR_PDB = None


def cross_pdb() -> np.ndarray:
    global _CROSS_PDB
    if _CROSS_PDB is None:
        _CROSS_PDB = np.ascontiguousarray(
            _load_or_build("pdb_cross", [0, 0, 0, 0], HOMES24[:4])
        )
    return _CROSS_PDB


def pair_pdb() -> np.ndarray:
    global _PAIR_PDB
    if _PAIR_PDB is None:
        kinds = [0, 0, 0, 0, 0, 1]
        goal = list(HOMES24[:4]) + [HOMES24[4], HOMES24[5]]
        _PAIR_PDB = _load_or_build("pdb_pair", kinds, goal)
    return _PAIR_PDB


# ---- IDA* --------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _heur(pos, active, cpdb, pdb, pm_e, pm_c, sigma):
    # cross-only baseline
    idx = 0
    mul = 1
    for u in range(4):
        idx += pos[u] * mul
        mul *= 24
    h = np.int64(cpdb[idx])
    for a in range(4):
        if active[a]:
            idx = 0
            mul = 1
            for u in range(4):
                idx += pm_e[a, pos[sigma[a, u]]] * mul
                mul *= 24
            idx += pm_e[a, pos[4 + 2 * a]] * mul
            mul *= 24
            idx += pm_c[a, pos[5 + 2 * a]] * mul
            d = np.int64(pdb[idx])
            if d > h:
                h = d
    return h


@nb.njit(cache=True)
def _is_goal(pos, homes, active, need_cross):
    if need_cross:
        for t in range(4):
            if pos[t] != homes[t]:
                return False
    for a in range(4):
        if active[a]:
            if pos[4 + 2 * a] != homes[4 + 2 * a] or pos[5 + 2 * a] != homes[5 + 2 * a]:
                return False
    return True


@nb.njit(cache=True)
def _ida(pos0, homes, active, need_cross, em, cm, cpdb, pdb, pm_e, pm_c, sigma, kinds,
         maxd, out, node_cap):
    n = 12
    if _is_goal(pos0, homes, active, need_cross):
        return 0
    h0 = _heur(pos0, active, cpdb, pdb, pm_e, pm_c, sigma)
    stack = np.zeros((maxd + 2, n), np.int64)
    nxt = np.zeros(maxd + 2, np.int64)
    last = np.full(maxd + 2, -1, np.int64)
    nodes = 0
    for limit in range(h0, maxd + 1):
        stack[0, :] = pos0
        depth = 0
        nxt[0] = 0
        last[0] = -1
        while depth >= 0:
            if nxt[depth] >= 18:
                depth -= 1
                continue
            m = nxt[depth]
            nxt[depth] += 1
            f = m // 3
            if depth > 0:
                lf = last[depth]
                if f == lf or (f == (lf + 3) % 6 and f < lf):
                    continue
            for t in range(n):
                if kinds[t] == 0:
                    stack[depth + 1, t] = em[m, stack[depth, t]]
                else:
                    stack[depth + 1, t] = cm[m, stack[depth, t]]
            nodes += 1
            if nodes > node_cap:
                return -2
            h = _heur(stack[depth + 1], active, cpdb, pdb, pm_e, pm_c, sigma)
            if depth + 1 + h > limit:
                continue
            out[depth] = m
            if _is_goal(stack[depth + 1], homes, active, need_cross):
                return depth + 1
            if depth + 1 < limit:
                depth += 1
                nxt[depth] = 0
                last[depth] = f
    return -1


def ida_solve(pos: np.ndarray, active: tuple, need_cross: bool = True,
              maxd: int = 16, node_cap: int = 400_000_000) -> list[str]:
    out = np.zeros(maxd + 2, dtype=np.int64)
    r = _ida(
        pos.astype(np.int64), HOMES24, np.array(active, dtype=np.bool_), need_cross,
        EM, CM, cross_pdb(), pair_pdb(), PM_E, PM_C, SIGMA, TRACK_KIND, maxd, out, node_cap,
    )
    if r < 0:
        raise RuntimeError(f"IDA* failed ({r})")
    return [C.MOVES[int(m)] for m in out[:r]]
