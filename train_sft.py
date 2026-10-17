"""Phase 1: supervised curriculum, one decision step at a time.

Each sample is (current 3-face view, history of own action tokens) -> next action token.
Stages are trained in sequence; each phase also replays earlier stages.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from rubik import tokenizer as T
from rubik.data.loader import StepData
from rubik.model import CubeLM, ModelConfig, load_ckpt, save_ckpt

# (phase name, {dataset: mix weight}, relative steps)
CURRICULUM = [
    ("rev", {"rev": 1.0}, 1.0),
    ("cross", {"cross": 0.7, "rev": 0.3}, 1.0),
    ("f2l_pair", {"f2l_pair": 0.6, "cross": 0.2, "rev": 0.2}, 1.0),
    ("f2l", {"f2l": 0.5, "f2l_pair": 0.2, "cross": 0.2, "rev": 0.1}, 1.0),
    ("oll", {"oll": 0.5, "f2l": 0.2, "f2l_pair": 0.1, "cross": 0.1, "rev": 0.1}, 1.0),
    ("pll", {"pll": 0.4, "oll": 0.2, "f2l": 0.15, "f2l_pair": 0.1, "cross": 0.1, "rev": 0.05}, 1.0),
    ("full", {"full": 0.4, "pll": 0.15, "oll": 0.15, "f2l": 0.1, "f2l_pair": 0.07, "cross": 0.07, "rev": 0.06}, 1.5),
]

# Backward (suffix) curriculum: learn the end of the solve first, then extend it earlier.
# Every dataset here ends in a solved cube (with <done>), so stage chaining is learned.
CURRICULUM_BACKWARD = [
    ("pll", {"pll": 1.0}, 1.0),
    ("oll_pll", {"oll_pll": 0.65, "pll": 0.35}, 1.0),
    ("f2l_oll_pll", {"f2l_oll_pll": 0.6, "oll_pll": 0.25, "pll": 0.15}, 1.0),
    ("full", {"full": 0.55, "f2l_oll_pll": 0.2, "oll_pll": 0.15, "pll": 0.1}, 1.5),
]

VISIBLE_FACE_TOKENS = None


def move_visibility_masks(device):
    """bool mask over vocab: face-turn tokens of visible faces (U,R,F) / hidden (D,L,B)."""
    vis = torch.zeros(T.VOCAB, dtype=torch.bool, device=device)
    hid = torch.zeros(T.VOCAB, dtype=torch.bool, device=device)
    for i, n in enumerate(T.NAMES[T.MOVE_BASE:T.ROT_BASE]):
        (vis if n[0] in "URF" else hid)[T.MOVE_BASE + i] = True
    return vis, hid


@torch.no_grad()
def evaluate(model, data: StepData, n=8192, bs=1024):
    model.eval()
    stats = {"acc": 0.0, "n": 0, "acc_vis": 0.0, "n_vis": 0, "acc_hid": 0.0, "n_hid": 0,
             "acc_scr_free": 0.0}
    vis, hid = move_visibility_masks(data.device)
    if data.val_idx.numel() == 0:
        return {}
    g = torch.Generator(device="cpu").manual_seed(0)
    pool = data.val_idx
    sel = pool[torch.randint(0, pool.numel(), (n,), generator=g).to(data.device)]
    per_stage = {}
    for i in range(0, n, bs):
        view, hist, t, tgt, w, stg = data.batch(sel[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(view, hist, t)
        # evaluate move accuracy with scratch tokens suppressed (the target is never scratch here)
        lg = logits.float().clone()
        lg[:, T.SCRATCH_BASE:] = -1e9
        pred = lg.argmax(-1)
        full = w >= 1.0
        ok = (pred == tgt) & full
        stats["acc"] += ok.sum().item()
        stats["n"] += full.sum().item()
        v = full & vis[tgt]
        h = full & hid[tgt]
        stats["acc_vis"] += (ok & v).sum().item()
        stats["n_vis"] += v.sum().item()
        stats["acc_hid"] += (ok & h).sum().item()
        stats["n_hid"] += h.sum().item()
        for s in stg.unique().tolist():
            m = full & (stg == s)
            a = per_stage.setdefault(s, [0, 0])
            a[0] += (ok & m).sum().item()
            a[1] += m.sum().item()
    out = {"acc": stats["acc"] / max(stats["n"], 1),
           "acc_visible_face_moves": stats["acc_vis"] / max(stats["n_vis"], 1),
           "acc_hidden_face_moves": stats["acc_hid"] / max(stats["n_hid"], 1)}
    for s, (a, c) in per_stage.items():
        out[f"acc_stage{s}"] = a / max(c, 1)
    model.train()
    return out


def lr_at(step, total, base, warmup=200, min_frac=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--steps_per_phase", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--view", default="ufr", choices=["ufr", "full"])
    ap.add_argument("--w_scratch", type=float, default=0.25)
    ap.add_argument("--phases", default="", help="comma-separated subset of phases to run")
    ap.add_argument("--init", default="", help="checkpoint to start from")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--curriculum", default="forward", choices=["forward", "backward"])
    a = ap.parse_args()

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = ModelConfig(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, view_len=T.view_len(a.view))
    model = load_ckpt(a.init, dev) if a.init else CubeLM(cfg).to(dev)
    print(f"model params: {model.n_params() / 1e6:.2f}M view={a.view}", flush=True)

    CUR = CURRICULUM_BACKWARD if a.curriculum == "backward" else CURRICULUM
    names = sorted({k for _, mix, _ in CUR for k in mix})
    data = {}
    for n in names:
        p = os.path.join(a.data_dir, f"train_{n}.npz")
        if os.path.exists(p):
            data[n] = StepData(p, dev, view_mode=a.view)
            print(f"loaded {n}: {data[n].M} steps", flush=True)
    wanted = [x for x in a.phases.split(",") if x] or [p[0] for p in CUR]
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    model.train()
    for pname, mix, rel in CUR:
        if pname not in wanted:
            continue
        mix = {k: v for k, v in mix.items() if k in data}
        tot = sum(mix.values())
        counts = {k: max(1, int(round(a.bs * v / tot))) for k, v in mix.items()}
        steps = int(a.steps_per_phase * rel)
        print(f"=== phase {pname}: {steps} steps mix={counts}", flush=True)
        t0 = time.time()
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, a.lr)
            parts = [data[k].sample(c, "train", a.w_scratch) for k, c in counts.items()]
            L = max(p[1].shape[1] for p in parts)
            view = torch.cat([p[0] for p in parts])
            hist = torch.cat([F.pad(p[1], (0, L - p[1].shape[1])) for p in parts])
            t = torch.cat([p[2] for p in parts])
            tgt = torch.cat([p[3] for p in parts])
            w = torch.cat([p[4] for p in parts])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(view, hist, t)
            loss = (F.cross_entropy(logits.float(), tgt, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                print(f"[{pname}] step {step}/{steps} loss {loss.item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
            if (step + 1) % a.save_every == 0:
                ex = {"view": a.view, "phase": pname, "step": step + 1}
                save_ckpt(os.path.join(a.out, "last.pt"), model, ex)  # rolling checkpoint
                if (step + 1) % (a.save_every * 4) == 0:
                    save_ckpt(os.path.join(a.out, f"{pname}_step{step + 1}.pt"), model, ex)
            if (step + 1) % a.eval_every == 0 or step == steps - 1:
                res = {"phase": pname, "step": step + 1, "loss": loss.item()}
                for k in counts:
                    res[k] = evaluate(model, data[k], n=4096)
                print(json.dumps(res), flush=True)
                log.write(json.dumps(res) + "\n")
                log.flush()
        save_ckpt(os.path.join(a.out, f"{pname}.pt"), model, {"view": a.view, "phase": pname})
        save_ckpt(os.path.join(a.out, "last.pt"), model, {"view": a.view, "phase": pname})


if __name__ == "__main__":
    main()

