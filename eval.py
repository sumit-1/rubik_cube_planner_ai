"""Closed-loop evaluation: solve rate per stage / scramble depth, with ablation switches."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from rubik import cube as C
from rubik import tokenizer as T
from rubik.data import gen
from rubik.gpucube import BatchCube
from rubik.rmodel import load_any
from rubik.rollout import rollout
from rubik.teacher import cfop, cross, f2l, oll

STAGE_STEPS = {"cross": 40, "f2l_pair": 60, "f2l": 150, "oll": 80, "pll": 80, "full": 250,
               "oll_pll": 100, "f2l_oll_pll": 190}
SOLVE_KINDS = ("full", "pll", "oll_pll", "f2l_oll_pll")  # success = cube solved
F2L_PIECES = ("DF", "DR", "DB", "DL", "FR", "FL", "BL", "BR", "DFR", "DFL", "DBL", "DBR")


def _rand_orient(rng, canon):
    return C.apply_alg(canon, gen._ORIENT_SEQS[int(rng.integers(24))])


def make_tasks(kind: str, n: int, seed: int, depth: int | None = None, canonical: bool = True):
    """Return (start states in current frame [n,54] uint8, per-task meta list, max_steps)."""
    rng = np.random.default_rng(seed)
    starts, meta = [], []
    for _ in range(n):
        if kind.startswith("rev"):
            k = depth
            s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, k))
            meta.append(k)
        elif kind == "pll":
            ps = gen.pll_states()
            s = ps[int(rng.integers(len(ps)))]
            meta.append(0)
        else:
            s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 25))
            a = 0
            if kind != "cross" and kind != "full":
                s, _ = cfop.advance(s, ["cross"])
            if kind == "f2l_pair":
                a = int(rng.integers(4))
                s, _ = cfop.advance(s, [f"f2l{i}" for i in range(a)])
            elif kind in ("oll", "oll_pll"):
                s, _ = cfop.advance(s, [f"f2l{i}" for i in range(4)])
            meta.append(a)
        starts.append(s if canonical else _rand_orient(rng, s))
    ms = (3 * depth + 12) if kind.startswith("rev") else STAGE_STEPS[kind]
    return np.stack(starts).astype(np.uint8), meta, ms


def stage_success(kind: str, final: np.ndarray, meta) -> np.ndarray:
    ok = np.zeros(len(final), dtype=bool)
    for i, s in enumerate(final):
        canon, _ = C.canonicalize(s)
        if kind.startswith("rev") or kind in SOLVE_KINDS:
            ok[i] = C.is_solved(s)
        elif kind == "cross":
            ok[i] = cross.cross_solved(canon)
        elif kind == "f2l_pair":
            ok[i] = cross.cross_solved(canon) and all(f2l.pairs_solved(canon)[: meta[i] + 1])
        elif kind == "f2l":
            ok[i] = cross.cross_solved(canon) and all(f2l.pairs_solved(canon))
        elif kind == "oll":
            ok[i] = (cross.cross_solved(canon) and all(f2l.pairs_solved(canon))
                     and oll.oll_oriented(canon))
    return ok


@torch.no_grad()
def run_task(model, env, kind, n, seed, depth=None, view_mode="ufr", bs=512, **kw):
    if not (kind.startswith("rev") or kind in SOLVE_KINDS):
        # stage tasks (no DONE in stage data): success if the stage goal is hit at any step
        return stage_success_any_time(model, env, kind, n, seed, view_mode=view_mode, bs=bs, **kw)
    starts, meta, max_steps = make_tasks(kind, n, seed, depth)
    succ = []
    for i in range(0, n, bs):
        st = torch.as_tensor(starts[i:i + bs].astype(np.int64), device=env.device)
        out = rollout(model, env, st, max_steps, view_mode=view_mode, **kw)
        succ.append(out["solved"].cpu().numpy())
    return float(np.concatenate(succ).mean())


@torch.no_grad()
def stage_success_any_time(model, env, kind, n, seed, view_mode="ufr", bs=512, **kw):
    """Success if the stage goal is reached at *any* step (checks every step)."""
    starts, meta, max_steps = make_tasks(kind, n, seed)
    total = 0
    for i in range(0, n, bs):
        st = torch.as_tensor(starts[i:i + bs].astype(np.int64), device=env.device)
        out = rollout(model, env, st, max_steps, view_mode=view_mode, auto_stop=False, record=True, **kw)
        recs = out["rec_states"].cpu().numpy()  # states before each action
        fin = out["state"].cpu().numpy()
        m = meta[i:i + bs]
        for j in range(len(st)):
            hit = False
            for t in range(recs.shape[1]):
                if stage_success(kind, recs[j, t:t + 1], [m[j]])[0]:
                    hit = True
                    break
            hit = hit or stage_success(kind, fin[j:j + 1], [m[j]])[0]
            total += hit
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tasks", default="rev,cross,f2l_pair,f2l,oll,pll,full")
    ap.add_argument("--rev_depths", default="1,2,3,4,5,6,8")
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--allow_scratch", action="store_true")
    ap.add_argument("--no_rot", action="store_true")
    ap.add_argument("--reset_mem", action="store_true", help="recurrent ablation: wipe memory each step")
    ap.add_argument("--force_inspect", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    model = load_any(a.ckpt, "cuda")
    view_mode = model.extra.get("view", "ufr")
    env = BatchCube("cuda")
    kw = dict(greedy=not a.sample, temperature=a.temp, allow_scratch=a.allow_scratch,
              allow_rot=not a.no_rot, reset_mem=a.reset_mem,
              prefix=[T.TOK[m] for m in (gen.INSPECT + gen.INSPECT_BACK)] if a.force_inspect else None)
    res = {"ckpt": a.ckpt, "view": view_mode, "flags": {k: (v if k != "prefix" else bool(v)) for k, v in kw.items()}}
    t0 = time.time()
    for task in a.tasks.split(","):
        if task == "rev":
            res["rev"] = {}
            for d in [int(x) for x in a.rev_depths.split(",")]:
                res["rev"][d] = run_task(model, env, f"rev{d}", a.n, a.seed, depth=d, view_mode=view_mode, **kw)
                print(f"rev depth {d}: solved {res['rev'][d]:.3f}", flush=True)
        else:
            res[task] = run_task(model, env, task, a.n, a.seed, view_mode=view_mode, **kw)
            print(f"{task}: success {res[task]:.3f}", flush=True)
    print(f"eval time {time.time() - t0:.0f}s")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
