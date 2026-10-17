"""Train the recurrent 3-face solver with:
  1. Full-trajectory BPTT (W=64) across stage-glanced trajectories (data_rnn2)
  2. Auxiliary next-view prediction loss (predicting the 27 visible stickers after each move/rotation)
  3. Online DAgger (Dataset Aggregation): periodically rolls out the model in closed-loop,
     catches states where it deviates, queries the CFOP teacher on the actual reached state
     while preserving the model's recurrent memory, and trains on those recovery steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rubik import cube as C
from rubik import tokenizer as T
from rubik.data import gen
from rubik.data.seqloader import TrajData
from rubik.gpucube import BatchCube
from rubik.rmodel import RecConfig, RecurrentCubeLM, load_rckpt, save_rckpt
from rubik.rollout import allowed_mask
from rubik.teacher import cfop, cross, f2l, oll

PHASES = {
    "short": {"rev": 0.2, "cross": 0.15, "f2l_pair": 0.2, "oll": 0.1, "pll": 0.15, "oll_pll": 0.2},
    "all": {"full": 0.30, "f2l_oll_pll": 0.20, "oll_pll": 0.12, "f2l": 0.12, "pll": 0.05, "oll": 0.05,
            "f2l_pair": 0.08, "cross": 0.04, "rev": 0.04},
}


def lr_at(it, total, base, warmup=150, min_frac=0.1):
    if it < warmup:
        return base * (it + 1) / warmup
    p = (it - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def unroll_joint(model, states, tokens, w, W, H, V, vp_weight=0.25, train=True):
    """Unroll recurrent model with full-trajectory BPTT (chunk W) and joint action + next-view loss."""
    B, Tn = tokens.shape
    padded = F.pad(tokens, (H, 0))
    mem = model.init_mem(B)
    total_w = w.sum().clamp(min=1.0)
    ok_sum = torch.zeros((), device=tokens.device)
    loss_sum = torch.zeros((), device=tokens.device)
    vp_ok_sum = torch.zeros((), device=tokens.device)
    vp_cnt = max(1, B * max(1, Tn - 1) * V)

    def step_both(view, recent, mem_in, act_tok):
        logits, new_mem = model.step(view, recent, mem_in)
        vp = model.predict_next_view(new_mem, act_tok)
        return logits, vp, new_mem

    for c0 in range(0, Tn, W):
        if train and c0 > 0:
            mem = mem.detach()
        loss_c = 0.0
        for t in range(c0, min(Tn, c0 + W)):
            view = states[:, t, :V].long() + T.COLOR_BASE
            recent = padded[:, t:t + H]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if train:
                    logits, vp, mem = checkpoint(step_both, view, recent, mem, tokens[:, t], use_reentrant=False)
                else:
                    logits, vp, mem = step_both(view, recent, mem, tokens[:, t])
            lg = logits.float()
            ce_act = F.cross_entropy(lg, tokens[:, t], reduction="none") * w[:, t]
            loss_step = ce_act.sum()
            if vp_weight > 0 and t + 1 < Tn:
                nxt_colors = states[:, t + 1, :V].long()
                ce_vp = F.cross_entropy(vp.float().view(-1, 6), nxt_colors.reshape(-1), reduction="none").view(B, V).mean(-1)
                loss_step = loss_step + vp_weight * (ce_vp * w[:, t]).sum()
                with torch.no_grad():
                    vp_ok_sum += (vp.argmax(-1) == nxt_colors).float().sum()
            loss_c = loss_c + loss_step
            with torch.no_grad():
                ok = ((lg.argmax(-1) == tokens[:, t]).float() * w[:, t])
                ok_sum += ok.sum()
        loss_c = loss_c / total_w
        loss_sum = loss_sum + loss_c.detach()
        if train:
            loss_c.backward()
    return loss_sum.item(), (ok_sum / total_w).item(), (vp_ok_sum / vp_cnt).item()


def teacher_remaining_tokens(state_cur: np.ndarray) -> list[int]:
    """Query the CFOP teacher from an arbitrary current-frame cube state `state_cur`.
    Detects which earliest CFOP stage is incomplete and solves from there to the end."""
    if C.is_solved(state_cur):
        return [T.DONE]
    canon, _ = C.canonicalize(state_cur)
    if not cross.cross_solved(canon):
        stages = cfop.STAGES
    else:
        ps = f2l.pairs_solved(canon)
        if not all(ps):
            first_u = ps.index(False)
            stages = [f"f2l{i}" for i in range(first_u, 4)] + ["oll", "pll"]
        elif not oll.oll_oriented(canon):
            stages = ["oll", "pll"]
        else:
            stages = ["pll"]
    plan = []
    s = canon
    for st in stages:
        mv = cfop.solve_stage(s, st)
        s = C.apply_alg(s, mv)
        plan.append((st, mv))
    # Convert into current-frame tokens with y2 glances before F2L pairs
    cur = state_cur.copy()
    toks: list[int] = []
    for idx, (st, mv) in enumerate(plan):
        if idx > 0 and st in ("f2l1", "f2l2", "f2l3"):
            for r in ("y'", "y2", "y2"):
                rt = T.TOK[r]
                toks.append(rt)
                cur = cur[T.PERM_TABLE[rt]]
        elif idx > 0 and st == "oll":
            rt = T.TOK["y'"]
            toks.append(rt)
            cur = cur[T.PERM_TABLE[rt]]
        for m in mv:
            tk = T.TOK[cfop.to_current_token(m, cur)]
            toks.append(tk)
            cur = cur[T.PERM_TABLE[tk]]
    toks.append(T.DONE)
    return toks


def dagger_step(model, env: BatchCube, rng: np.random.Generator, B: int = 32, rollout_k: int = 10,
                horizon: int = 20, vp_weight: float = 0.25):
    """Roll out the model in closed-loop for `rollout_k` steps from mixed starting states
    (keeping its recurrent memory!), then query the teacher on the actual reached states for
    the next `horizon` steps and backpropagate through the continuation."""
    dev = env.device
    H = model.cfg.window
    V = model.cfg.view_len
    allowed = allowed_mask(dev, False, True, True)
    # Mix of full scrambles, post-cross, post-2-pairs, and OLL states
    starts = []
    for i in range(B):
        s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 25))
        r = i % 4
        if r == 1:
            s, _ = cfop.advance(s, ["cross"])
        elif r == 2:
            s, _ = cfop.advance(s, ["cross", "f2l0", "f2l1"])
        elif r == 3:
            s, _ = cfop.advance(s, ["cross", "f2l0", "f2l1", "f2l2", "f2l3"])
        starts.append(s)
    state = torch.as_tensor(np.stack(starts).astype(np.int64), device=dev)
    recent = torch.zeros(B, H, dtype=torch.long, device=dev)
    mem = model.init_mem(B)

    # First execute inspection (x2, x2) + rollout_k model steps with gradient on the last 4 rollout steps
    inspect_ids = [T.TOK[m] for m in (gen.INSPECT + gen.INSPECT_BACK)]
    total_roll = len(inspect_ids) + int(rng.integers(2, rollout_k + 1))
    for t in range(total_roll):
        view = state[:, :V] + T.COLOR_BASE
        if t < total_roll - 4:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, mem = model.step(view, recent, mem)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, mem = checkpoint(model.step, view, recent, mem, use_reentrant=False)
        if t < len(inspect_ids):
            tok = torch.full((B,), inspect_ids[t], dtype=torch.long, device=dev)
        else:
            tok = logits.float().masked_fill(~allowed, float("-inf")).argmax(-1)
            tok = torch.where(tok == T.DONE, torch.full_like(tok, T.TOK["U"]), tok)
        state = env.step(state, tok)
        recent = torch.cat([recent[:, 1:], tok[:, None]], 1)

    # Query teacher on each reached state
    st_np = state.cpu().numpy().astype(np.uint8)
    plans = [teacher_remaining_tokens(st_np[b])[:horizon] for b in range(B)]
    Tm = max(len(p) for p in plans)
    t_toks = torch.zeros(B, Tm, dtype=torch.long, device=dev)
    t_w = torch.zeros(B, Tm, dtype=torch.float32, device=dev)
    for b, p in enumerate(plans):
        t_toks[b, :len(p)] = torch.as_tensor(p, device=dev)
        t_w[b, :len(p)] = 1.0

    # Unroll teacher recovery from the model's current (state, recent, mem)
    total_w = t_w.sum().clamp(min=1.0)
    loss_c = 0.0
    ok_sum = 0.0
    cur_state = state
    for t in range(Tm):
        view = cur_state[:, :V] + T.COLOR_BASE
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, mem = checkpoint(model.step, view, recent, mem, use_reentrant=False)
        lg = logits.float()
        ce = F.cross_entropy(lg, t_toks[:, t], reduction="none") * t_w[:, t]
        loss_c = loss_c + ce.sum()
        with torch.no_grad():
            ok_sum += float(((lg.argmax(-1) == t_toks[:, t]).float() * t_w[:, t]).sum().item())
        cur_state = env.step(cur_state, t_toks[:, t])
        recent = torch.cat([recent[:, 1:], t_toks[:, t:t + 1]], 1)
    loss_c = loss_c / total_w
    loss_c.backward()
    return float(loss_c.detach().item()), ok_sum / float(total_w.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_rnn2")
    ap.add_argument("--out", default="runs/rnn_v2")
    ap.add_argument("--init", default="runs/rnn_mem/final.pt")
    ap.add_argument("--iters_short", type=int, default=600)
    ap.add_argument("--iters_all", type=int, default=2400)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=64, help="Full-trajectory BPTT window")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--vp_weight", type=float, default=0.25)
    ap.add_argument("--dagger_every", type=int, default=4, help="Run online DAgger step every N iters in phase all")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--eval_every", type=int, default=500)
    a = ap.parse_args()

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    V = 27
    if a.init and os.path.exists(a.init):
        model = load_rckpt(a.init, dev)
        print(f"loaded init checkpoint {a.init} ({model.n_params() / 1e6:.2f}M params)", flush=True)
    else:
        model = RecurrentCubeLM(RecConfig()).to(dev)
    H = model.cfg.window

    names = sorted({k for ph in PHASES.values() for k in ph})
    data = {}
    for n in names:
        p = os.path.join(a.data_dir, f"train_{n}.npz")
        if os.path.exists(p):
            data[n] = TrajData(p, dev)
    print("loaded:", {k: v.train_start.numel() for k, v in data.items()}, flush=True)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    env = BatchCube(dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rng = np.random.default_rng(42)
    W = a.chunk
    model.train()
    total_iters = a.iters_short + a.iters_all
    it = 0
    t0 = time.time()
    ema = {}
    for phase, n_it in (("short", a.iters_short), ("all", a.iters_all)):
        mix = {k: v for k, v in PHASES[phase].items() if k in data}
        keys = list(mix)
        probs = np.array([mix[k] for k in keys])
        probs = probs / probs.sum()
        print(f"=== phase {phase}: {n_it} iterations W={W} mix={dict(zip(keys, probs.round(3)))}", flush=True)
        for _ in range(n_it):
            for g in opt.param_groups:
                g["lr"] = lr_at(it, total_iters, a.lr)
            opt.zero_grad(set_to_none=True)
            if phase == "all" and a.dagger_every > 0 and (it % a.dagger_every == 0):
                loss, acc = dagger_step(model, env, rng, B=32, rollout_k=10, horizon=20, vp_weight=a.vp_weight)
                vp_acc = 0.0
                k = "dagger"
                Tlen = 20
            else:
                k = keys[int(rng.choice(len(keys), p=probs))]
                states, tokens, w, stage, valid = data[k].sample(a.bs)
                loss, acc, vp_acc = unroll_joint(model, states, tokens, w, W, H, V, a.vp_weight, train=True)
                Tlen = tokens.shape[1]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            e = ema.setdefault(k, [loss, acc])
            e[0] = 0.95 * e[0] + 0.05 * loss
            e[1] = 0.95 * e[1] + 0.05 * acc
            it += 1
            if it % 25 == 0:
                print(f"[{phase}] it {it}/{total_iters} {k} T={Tlen} loss {loss:.3f} acc {acc:.3f} "
                      f"vp {vp_acc:.3f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
            if it % a.save_every == 0:
                ex = {"view": "ufr", "phase": phase, "iter": it}
                save_rckpt(os.path.join(a.out, "last.pt"), model, ex)
                if it % (a.save_every * 2) == 0:
                    save_rckpt(os.path.join(a.out, f"it{it}.pt"), model, ex)
            if it % a.eval_every == 0:
                from eval import run_task
                cl = {
                    "rev4": run_task(model, env, "rev4", 128, 7, depth=4),
                    "cross": run_task(model, env, "cross", 128, 7),
                    "f2l_pair": run_task(model, env, "f2l_pair", 128, 7),
                    "f2l": run_task(model, env, "f2l", 128, 7),
                    "oll_pll": run_task(model, env, "oll_pll", 128, 7),
                    "f2l_oll_pll": run_task(model, env, "f2l_oll_pll", 128, 7),
                    "full": run_task(model, env, "full", 128, 7),
                }
                res = {"it": it, "phase": phase, "ema": {kk: [round(x, 4) for x in v] for kk, v in ema.items()},
                       "closed_loop": cl, "time": round(time.time() - t0, 1)}
                print(json.dumps(res), flush=True)
                log.write(json.dumps(res) + "\n")
                log.flush()
    save_rckpt(os.path.join(a.out, "final.pt"), model, {"view": "ufr", "iter": it})
    save_rckpt(os.path.join(a.out, "last.pt"), model, {"view": "ufr", "iter": it})


if __name__ == "__main__":
    main()
