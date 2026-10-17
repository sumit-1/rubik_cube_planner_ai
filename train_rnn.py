"""Train the recurrent duplex model (continuous memory feedback) with truncated BPTT.

Each iteration samples B whole trajectories from one dataset, then unrolls them step by step:
the memory written at step t is read at step t+1, and gradients flow back through the memory
inside each W-step chunk (the memory is detached at chunk boundaries but its *content* carries
on). Only the expert action is supervised - never the memory content.
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

from rubik import tokenizer as T
from rubik.data.seqloader import TrajData
from rubik.gpucube import BatchCube
from rubik.rmodel import RecConfig, RecurrentCubeLM, load_rckpt, save_rckpt

PHASES = {
    "short": {"rev": 0.2, "cross": 0.15, "f2l_pair": 0.2, "oll": 0.1, "pll": 0.15, "oll_pll": 0.2},
    "all": {"full": 0.3, "f2l_oll_pll": 0.2, "oll_pll": 0.12, "f2l": 0.1, "pll": 0.05, "oll": 0.05,
            "f2l_pair": 0.08, "cross": 0.05, "rev": 0.05},
}


def lr_at(it, total, base, warmup=200, min_frac=0.1):
    if it < warmup:
        return base * (it + 1) / warmup
    p = (it - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def unroll(model, states, tokens, w, W, H, V, train=True, scaler_total=None):
    B, Tn = tokens.shape
    padded = F.pad(tokens, (H, 0))
    mem = model.init_mem(B)
    total_w = scaler_total if scaler_total is not None else w.sum().clamp(min=1.0)
    ok_sum = torch.zeros((), device=tokens.device)
    loss_sum = torch.zeros((), device=tokens.device)
    per_t_ok = []
    for c0 in range(0, Tn, W):
        if train:
            mem = mem.detach()
        loss_c = 0.0
        for t in range(c0, min(Tn, c0 + W)):
            view = states[:, t, :V].long() + T.COLOR_BASE
            recent = padded[:, t:t + H]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if train:  # activation checkpointing per step: keeps BPTT memory small
                    logits, mem = checkpoint(model.step, view, recent, mem, use_reentrant=False)
                else:
                    logits, mem = model.step(view, recent, mem)
            lg = logits.float()
            ce = F.cross_entropy(lg, tokens[:, t], reduction="none") * w[:, t]
            loss_c = loss_c + ce.sum()
            with torch.no_grad():
                ok = ((lg.argmax(-1) == tokens[:, t]).float() * w[:, t])
                ok_sum += ok.sum()
                per_t_ok.append(ok)
        loss_c = loss_c / total_w
        loss_sum = loss_sum + loss_c.detach()
        if train:
            loss_c.backward()
    return loss_sum.item(), (ok_sum / total_w).item(), torch.stack(per_t_ok, 1)


@torch.no_grad()
def val_acc(model, ds, W, H, V, n=256):
    model.eval()
    states, tokens, w, stage, valid = ds.sample(min(n, ds.val_start.numel()), "val")
    _, acc, per_t = unroll(model, states, tokens, w, W, H, V, train=False)
    stage_acc = {}
    for s in stage.unique().tolist():
        m = (stage == s) & (w > 0)
        if m.any():
            stage_acc[int(s)] = round((per_t[:, : m.shape[1]] * m).sum().item() / m.sum().item(), 4)
    model.train()
    return acc, stage_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_rnn")
    ap.add_argument("--out", default="runs/rnn")
    ap.add_argument("--init", default="")
    ap.add_argument("--iters_short", type=int, default=2000)
    ap.add_argument("--iters_all", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16, help="BPTT window (steps)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mem_slots", type=int, default=16)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--view", default="ufr", choices=["ufr", "full"])
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--no_memory_grad", action="store_true",
                    help="ablation: detach memory every step (no BPTT through the feedback loop)")
    a = ap.parse_args()

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    V = T.view_len(a.view)
    cfg = RecConfig(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, mem_slots=a.mem_slots,
                    window=a.window, view_len=V)
    model = load_rckpt(a.init, dev) if a.init else RecurrentCubeLM(cfg).to(dev)
    H = model.cfg.window
    print(f"recurrent model: {model.n_params() / 1e6:.2f}M params, K={a.mem_slots}, H={H}, view={a.view}",
          flush=True)
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
    rng = np.random.default_rng(0)
    W = 1 if a.no_memory_grad else a.chunk
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
        print(f"=== phase {phase}: {n_it} iterations mix={dict(zip(keys, probs.round(3)))}", flush=True)
        for _ in range(n_it):
            for g in opt.param_groups:
                g["lr"] = lr_at(it, total_iters, a.lr)
            k = keys[int(rng.choice(len(keys), p=probs))]
            states, tokens, w, stage, valid = data[k].sample(a.bs)
            opt.zero_grad(set_to_none=True)
            loss, acc, _ = unroll(model, states, tokens, w, W, H, V, train=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            e = ema.setdefault(k, [loss, acc])
            e[0] = 0.95 * e[0] + 0.05 * loss
            e[1] = 0.95 * e[1] + 0.05 * acc
            it += 1
            if it % 25 == 0:
                print(f"[{phase}] it {it}/{total_iters} {k} T={tokens.shape[1]} loss {loss:.3f} acc {acc:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
            if it % a.save_every == 0:
                ex = {"view": a.view, "phase": phase, "iter": it}
                save_rckpt(os.path.join(a.out, "last.pt"), model, ex)
                if it % (a.save_every * 4) == 0:
                    save_rckpt(os.path.join(a.out, f"it{it}.pt"), model, ex)
            if it % a.eval_every == 0:
                res = {"it": it, "phase": phase, "ema": {kk: [round(x, 4) for x in v] for kk, v in ema.items()}}
                res["val"] = {kk: val_acc(model, data[kk], W, H, V)[0] for kk in data}
                try:
                    from eval import run_task
                    res["closed_loop"] = {
                        "rev4": run_task(model, env, "rev4", 128, 7, depth=4, view_mode=a.view),
                        "oll_pll": run_task(model, env, "oll_pll", 128, 7, view_mode=a.view),
                    }
                except Exception as ex_:  # eval is best-effort during training
                    res["closed_loop_err"] = str(ex_)
                print(json.dumps(res), flush=True)
                log.write(json.dumps(res) + "\n")
                log.flush()
    save_rckpt(os.path.join(a.out, "final.pt"), model, {"view": a.view, "iter": it})
    save_rckpt(os.path.join(a.out, "last.pt"), model, {"view": a.view, "iter": it})


if __name__ == "__main__":
    main()
