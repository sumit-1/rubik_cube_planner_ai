"""Self-supervised 3-face world-model pretraining for the recurrent memory.

Trains RecurrentCubeLM on random move + rotation sequences starting from random cube states.
Each trajectory opens with an inspection rotation (`x2`, `x2`) so all 6 faces are seen once
through the 3-face U/F/R window, then executes random face turns and whole-cube rotations.
The ONLY objective is to predict the 27 visible stickers of the next step (`next_view`),
especially stickers that were on hidden faces and just rotated into the 3-face view.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rubik import cube as C
from rubik import tokenizer as T
from rubik.gpucube import BatchCube
from rubik.rmodel import RecConfig, RecurrentCubeLM, load_rckpt, save_rckpt


def sample_wm_batch(env: BatchCube, B: int, L: int, p_rot: float = 0.25):
    """Generate B random walk trajectories of length L on GPU.
    Returns (views [B,L,27], tokens [B,L], next_colors [B,L,27], changed_mask [B,L,27])."""
    dev = env.device
    # Start from random 20-move scrambles on GPU
    st = env.solved_batch(B)
    move_ids = torch.as_tensor(T.MOVE_IDS, device=dev)
    rot_ids = torch.as_tensor(T.ROT_IDS, device=dev)
    for _ in range(20):
        m = move_ids[torch.randint(0, len(move_ids), (B,), device=dev)]
        st = env.step(st, m)

    # Sequence of tokens: first 2 steps are x2, x2 (inspection), then random moves + rotations
    toks = torch.zeros(B, L, dtype=torch.long, device=dev)
    x2_id = T.TOK["x2"]
    y2_id = T.TOK["y2"]
    toks[:, 0] = x2_id
    toks[:, 1] = x2_id
    for t in range(2, L):
        if t in (12, 24) and L > 24:
            # Periodic y2 glance
            toks[:, t] = y2_id
        else:
            is_r = torch.rand(B, device=dev) < p_rot
            rm = move_ids[torch.randint(0, len(move_ids), (B,), device=dev)]
            rr = rot_ids[torch.randint(0, len(rot_ids), (B,), device=dev)]
            toks[:, t] = torch.where(is_r, rr, rm)

    all_states = env.run(st, toks)  # [B, L+1, 54]
    cur_colors = all_states[:, :-1, :27]  # [B, L, 27] in 0..5
    nxt_colors = all_states[:, 1:, :27]   # [B, L, 27] in 0..5
    views = cur_colors + T.COLOR_BASE
    changed = nxt_colors != cur_colors
    return views, toks, nxt_colors, changed


def lr_at(it, total, base, warmup=100, min_frac=0.1):
    if it < warmup:
        return base * (it + 1) / warmup
    p = (it - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/wm_pretrain")
    ap.add_argument("--init", default="")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mem_slots", type=int, default=16)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=200)
    a = ap.parse_args()

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    cfg = RecConfig(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                    mem_slots=a.mem_slots, window=a.window, view_len=27)
    model = load_rckpt(a.init, dev) if a.init else RecurrentCubeLM(cfg).to(dev)
    H = model.cfg.window
    env = BatchCube(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.02, fused=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()

    def step_vp(view, recent, mem, act_tok):
        _, new_mem = model.step(view, recent, mem)
        vp = model.predict_next_view(new_mem, act_tok)
        return vp, new_mem

    model.train()
    for it in range(1, a.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it - 1, a.iters, a.lr)
        views, toks, nxt_c, changed = sample_wm_batch(env, a.bs, a.seq_len)
        padded = F.pad(toks, (H, 0))
        mem = model.init_mem(a.bs)
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        ok_all = 0
        ok_chg = 0
        n_chg = int(changed[:, 2:].sum().item())
        for t in range(a.seq_len):
            recent = padded[:, t:t + H]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vp, mem = checkpoint(step_vp, views[:, t], recent, mem, toks[:, t], use_reentrant=False)
            # Only score steps t >= 2 (after inspection has seen all 6 faces)
            if t >= 2:
                vp_f = vp.float()
                ce = F.cross_entropy(vp_f.view(-1, 6), nxt_c[:, t].reshape(-1), reduction="none").view(a.bs, 27)
                # Upweight stickers that changed (moved from another face/position) by 3x
                w_st = torch.where(changed[:, t], 3.0, 1.0)
                loss_total = loss_total + (ce * w_st).mean() / (a.seq_len - 2)
                with torch.no_grad():
                    pred = vp_f.argmax(-1)
                    eq = pred == nxt_c[:, t]
                    ok_all += int(eq.sum().item())
                    ok_chg += int((eq & changed[:, t]).sum().item())
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 50 == 0 or it == 1:
            acc_all = ok_all / (a.bs * (a.seq_len - 2) * 27)
            acc_chg = ok_chg / max(1, n_chg)
            rec = dict(it=it, loss=round(loss_total.item(), 4),
                       acc_all=round(acc_all, 4), acc_changed_stickers=round(acc_chg, 4),
                       time=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if it % a.eval_every == 0 or it == a.iters:
            save_rckpt(os.path.join(a.out, "last.pt"), model, {"view": "ufr", "wm_iter": it})
    save_rckpt(os.path.join(a.out, "final.pt"), model, {"view": "ufr", "wm_iter": a.iters})


if __name__ == "__main__":
    main()
