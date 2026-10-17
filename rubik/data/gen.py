"""Dataset generator.

A dataset is a flat array of *decision steps*. Step i stores the full cube state in the
current (possibly rotated) frame *before* the action, the action token that follows, a loss
weight code, a stage id, and the index of the first step of its trajectory (the history of
a step = tokens[traj_start[i]:i]).

Kinds: rev, cross, f2l_pair, f2l, oll, pll, full.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool

import numpy as np

from .. import cube as C
from .. import tokenizer as T
from ..teacher import cfop, pll

STAGE_ID = {"rev": 0, "cross": 1, "f2l": 2, "oll": 3, "pll": 4, "extra": 5}
KINDS = ["rev", "cross", "f2l_pair", "f2l", "oll", "pll", "full", "oll_pll", "f2l_oll_pll"]

W_NONE, W_SCRATCH, W_FULL = 0, 1, 2


@dataclass
class GenConfig:
    kind: str = "full"
    n: int = 1000                 # trajectories
    seed: int = 0
    scramble_len: int = 25        # for non-rev kinds
    rev_min: int = 1
    rev_max: int = 8
    p_inspect: float = 1.0        # open with look-around (rotate to opposite corner and back)
    p_stage_glance: float = 1.0   # glance (y2, y2) at stage boundaries so hidden F2L pieces are refreshed in memory
    f2l_rotate: bool = False      # rotate y before each F2L pair so the active slot is always front-right (visible) and D-face cross is visible during x/z2 inspection
    p_rot: float = 0.03           # random whole-cube rotation inserted before a step (no loss)
    p_scratch: float = 0.08       # random scratch token inserted as a (low weight) target
    random_orientation: bool = True


def _inspect_seq() -> list[str]:
    """Shortest rotation sequence after which the view shows the three complementary faces."""
    for L in (1, 2, 3):
        for seq in itertools.product(C.ROTS, repeat=L):
            s = C.apply_alg(C.SOLVED.copy(), list(seq))
            if sorted(int(s[i]) for i in (4, 13, 22)) == [3, 4, 5]:
                return list(seq)
    raise RuntimeError


INSPECT = _inspect_seq()
INSPECT_BACK = C.invert_alg(INSPECT)

_ORIENT_SEQS = [seq for (_, seq) in C.ORIENT_TO_CANON.values()]
_PLL_STATES: list[np.ndarray] | None = None


def pll_states() -> list[np.ndarray]:
    global _PLL_STATES
    if _PLL_STATES is None:
        seen = {}
        for seq in list(pll.LIBRARY.values()) + [[]]:
            for pre in range(4):
                for post in range(4):
                    s = C.apply_alg(C.SOLVED.copy(), ["U"] * pre + seq + ["U"] * post)
                    seen[s.tobytes()] = s
        _PLL_STATES = list(seen.values())
    return _PLL_STATES


def _scramble_state(rng, n):
    return C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, n))


def make_start_and_plan(rng, cfg: GenConfig):
    """Returns (canonical start state, plan [(stage_name, canon_move)], end_with_done)."""
    k = cfg.kind
    if k == "rev":
        n = int(rng.integers(cfg.rev_min, cfg.rev_max + 1))
        scr = C.random_scramble(rng, n)
        s = C.apply_alg(C.SOLVED.copy(), scr)
        plan = [("rev", m) for m in C.invert_alg(scr)]
        return s, plan, True
    if k == "pll":
        states = pll_states()
        s = states[int(rng.integers(len(states)))]
        _, plan = cfop.advance(s, ["pll"])
        return s, plan, True
    s = _scramble_state(rng, cfg.scramble_len)
    if k == "full":
        _, plan = cfop.advance(s, cfop.STAGES)
        return s, plan, True
    if k == "cross":
        _, plan = cfop.advance(s, ["cross"])
        return s, plan, False
    s, _ = cfop.advance(s, ["cross"])
    if k == "f2l_oll_pll":  # suffix curriculum: everything after the cross, ends solved
        _, plan = cfop.advance(s, ["f2l0", "f2l1", "f2l2", "f2l3", "oll", "pll"])
        return s, plan, True
    if k == "f2l":
        _, plan = cfop.advance(s, ["f2l0", "f2l1", "f2l2", "f2l3"])
        return s, plan, False
    if k == "f2l_pair":
        a = int(rng.integers(4))
        s, _ = cfop.advance(s, [f"f2l{i}" for i in range(a)])
        _, plan = cfop.advance(s, [f"f2l{a}"])
        return s, plan, False
    s, _ = cfop.advance(s, ["f2l0", "f2l1", "f2l2", "f2l3"])
    if k == "oll":
        _, plan = cfop.advance(s, ["oll"])
        return s, plan, False
    if k == "oll_pll":  # suffix curriculum: last layer only, ends solved
        _, plan = cfop.advance(s, ["oll", "pll"])
        return s, plan, True
    raise ValueError(k)


def build_trajectory(rng, cfg: GenConfig):
    canon, plan, done = make_start_and_plan(rng, cfg)
    cur = canon
    if cfg.random_orientation:
        cur = C.apply_alg(canon, _ORIENT_SEQS[int(rng.integers(24))])
    states, toks, weights, stages = [], [], [], []

    def emit(token_name_or_id, w, stage):
        nonlocal cur
        t = token_name_or_id if isinstance(token_name_or_id, int) else T.TOK[token_name_or_id]
        states.append(cur)
        toks.append(t)
        weights.append(w)
        stages.append(STAGE_ID[stage])
        if T.is_cube_token(t):
            cur = cur[T.PERM_TABLE[t]]

    if rng.random() < cfg.p_inspect and plan:
        for m in INSPECT:
            emit(m, W_FULL, "extra")
        for m in INSPECT_BACK:
            emit(m, W_FULL, "extra")
    prev_stage = None
    for stage, m in plan:
        if stage != prev_stage:
            if cfg.f2l_rotate and prev_stage is not None and stage in ("f2l1", "f2l2", "f2l3"):
                # Rotate y' so the next F2L slot becomes the visible front-right (FR) slot,
                # and glance at the back (y2, y2) so any piece in the back slots is seen.
                emit("y'", W_FULL, "extra")
                if rng.random() < cfg.p_stage_glance:
                    emit("y2", W_FULL, "extra")
                    emit("y2", W_FULL, "extra")
            elif cfg.f2l_rotate and prev_stage == "f2l3" and stage == "oll":
                emit("y'", W_FULL, "extra")
            elif prev_stage is not None and stage in ("f2l0", "f2l1", "f2l2", "f2l3") and rng.random() < cfg.p_stage_glance:
                emit("y2", W_FULL, "extra")
                emit("y2", W_FULL, "extra")
            elif prev_stage is None and stage in ("f2l0", "f2l1", "f2l2", "f2l3") and cfg.p_inspect == 0.0 and rng.random() < cfg.p_stage_glance:
                emit("y2", W_FULL, "extra")
                emit("y2", W_FULL, "extra")
            prev_stage = stage
        if rng.random() < cfg.p_rot:
            emit(T.TOK[C.ROTS[int(rng.integers(9))]], W_NONE, "extra")
        if rng.random() < cfg.p_scratch:
            emit(int(T.SCRATCH_BASE + rng.integers(T.N_SCRATCH)), W_SCRATCH, "extra")
        emit(cfop.to_current_token(m, cur), W_FULL, "rev" if stage == "rev" else _grp(stage))
    if done:
        emit(T.DONE, W_FULL, "extra")
    if not toks:
        return None
    return np.stack(states), np.array(toks, np.int16), np.array(weights, np.uint8), np.array(stages, np.uint8)


def _grp(stage: str) -> str:
    return {"cross": "cross", "oll": "oll", "pll": "pll"}.get(stage, "f2l")


def generate(cfg: GenConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    S, K, W, G, start = [], [], [], [], []
    off = 0
    for _ in range(cfg.n):
        traj = build_trajectory(rng, cfg)
        if traj is None:
            continue
        s, t, w, g = traj
        S.append(s)
        K.append(t)
        W.append(w)
        G.append(g)
        start.append(np.full(len(t), off, dtype=np.int32))
        off += len(t)
    return dict(
        states=np.concatenate(S).astype(np.uint8),
        tokens=np.concatenate(K),
        weights=np.concatenate(W),
        stages=np.concatenate(G),
        traj_start=np.concatenate(start),
    )


def _worker(cfg_dict):
    cfg = GenConfig(**cfg_dict)
    return generate(cfg)


def warm_caches(max_samples: int = 20000, seed: int = 999):
    """Fill the OLL (216 patterns) and PLL (288 states) caches in the parent process so that
    worker processes never write them concurrently."""
    from ..teacher import oll as oll_mod
    rng = np.random.default_rng(seed)
    f2l_stages = ["cross", "f2l0", "f2l1", "f2l2", "f2l3"]
    seen = set()
    for _ in range(max_samples):
        s = _scramble_state(rng, 25)
        s, _ = cfop.advance(s, f2l_stages)
        seen.add(oll_mod.oll_case_key(s))
        oll_mod.solve_oll(s)
        if len(seen) >= 216:
            break
    for s in pll_states():
        pll.solve_pll(s)
    # PLL states reachable after OLL are a subset of the 288 states; also warm directly.
    return len(seen)


def generate_parallel(cfg: GenConfig, procs: int = 8, chunks: int | None = None):
    warm_caches()
    chunks = chunks or procs
    per = cfg.n // chunks
    cfgs = []
    for i in range(chunks):
        d = asdict(cfg)
        d["n"] = per if i < chunks - 1 else cfg.n - per * (chunks - 1)
        d["seed"] = cfg.seed * 10007 + i
        cfgs.append(d)
    with Pool(procs) as p:
        parts = p.map(_worker, cfgs)
    off = 0
    out = {k: [] for k in parts[0]}
    for part in parts:
        for k, v in part.items():
            out[k].append(v + off if k == "traj_start" else v)
        off += len(part["tokens"])
    return {k: np.concatenate(v) for k, v in out.items()}


def save(path: str, data: dict, cfg: GenConfig):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **data, cfg=np.array([str(asdict(cfg))]))


def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ("states", "tokens", "weights", "stages", "traj_start")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--rev_min", type=int, default=1)
    ap.add_argument("--rev_max", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p_inspect", type=float, default=1.0)
    ap.add_argument("--p_rot", type=float, default=0.03)
    ap.add_argument("--p_scratch", type=float, default=0.08)
    ap.add_argument("--f2l_rotate", action="store_true")
    ap.add_argument("--canonical_start", action="store_true")
    a = ap.parse_args()
    cfg = GenConfig(kind=a.kind, n=a.n, seed=a.seed, rev_min=a.rev_min, rev_max=a.rev_max,
                    p_inspect=a.p_inspect, p_rot=a.p_rot, p_scratch=a.p_scratch,
                    f2l_rotate=a.f2l_rotate, random_orientation=not a.canonical_start)
    t0 = time.time()
    data = generate_parallel(cfg, a.procs, max(a.procs, 1))
    save(a.out, data, cfg)
    print(f"{a.kind}: {a.n} trajectories, {len(data['tokens'])} steps, {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
