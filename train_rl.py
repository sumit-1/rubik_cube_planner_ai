"""Phase 2: GRPO from the SFT checkpoint.

Episodes: scramble of depth k from solved (random orientation), model acts with sampling.
Scratch tokens may be emitted freely (no cube effect) so their use can emerge through reward.
Reward: +1 if solved, small step penalty. Advantage: group-relative (G samples per start).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from rubik import cube as C
from rubik import tokenizer as T
from rubik.data import gen
from rubik.gpucube import BatchCube
from rubik.model import load_ckpt, save_ckpt
from rubik.rollout import allowed_mask, rollout


def make_starts(rng, depths: np.ndarray) -> np.ndarray:
    out = []
    for k in depths:
        s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, int(k)))
        s = C.apply_alg(s, gen._ORIENT_SEQS[int(rng.integers(24))])
        out.append(s)
    return np.stack(out).astype(np.int64)


def policy_logp(model, view, hist, t, tgt, allowed):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(view, hist, t)
    logits = logits.float().masked_fill(~allowed, float("-inf"))
    lp = torch.log_softmax(logits, -1)
    return lp.gather(1, tgt[:, None]).squeeze(1), lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--starts", type=int, default=64, help="start states per iteration")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02, help="KL to the SFT reference")
    ap.add_argument("--step_pen", type=float, default=0.005)
    ap.add_argument("--k0", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=14)
    ap.add_argument("--promote", type=float, default=0.75)
    ap.add_argument("--no_scratch", action="store_true", help="baseline: scratch tokens disabled")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    model = load_ckpt(a.init, dev)
    view_mode = model.extra.get("view", "ufr")
    V = T.view_len(view_mode)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    env = BatchCube(dev)
    allowed = allowed_mask(dev, not a.no_scratch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    k_cur = a.k0
    t0 = time.time()
    for it in range(a.iters):
        depths = rng.integers(1, k_cur + 1, size=a.starts)
        depths[: a.starts // 2] = k_cur  # half of the batch at the frontier depth
        starts = np.repeat(make_starts(rng, depths), a.group, axis=0)
        max_steps = int(2 * k_cur + 10)
        st = torch.as_tensor(starts, device=dev)
        out = rollout(model, env, st, max_steps, greedy=False, allow_scratch=not a.no_scratch,
                      record=True, view_mode=view_mode)
        solved = out["solved"].float()
        steps = out["hist_len"].float()
        R = solved - a.step_pen * steps
        Rg = R.view(-1, a.group)
        adv = ((Rg - Rg.mean(1, keepdim=True)) / (R.std() + 1e-6)).view(-1)

        alive = out["rec_alive"]
        toks = out["rec_tokens"]
        rs = out["rec_states"]
        b_idx, t_idx = alive.nonzero(as_tuple=True)
        n = b_idx.numel()
        perm = torch.randperm(n, device=dev)
        b_idx, t_idx = b_idx[perm], t_idx[perm]
        model.train()
        opt.zero_grad(set_to_none=True)
        tot_kl = tot_ent = tot_scr = 0.0
        for i in range(0, n, a.chunk):
            bi, ti = b_idx[i:i + a.chunk], t_idx[i:i + a.chunk]
            L = int(ti.max().item())
            ar = torch.arange(L, device=dev)[None, :]
            hist = torch.where(ar < ti[:, None], toks[bi, :L] if L > 0 else toks[bi, :0], torch.zeros(1, dtype=toks.dtype, device=dev))
            view = rs[bi, ti][:, :V].long() + T.COLOR_BASE
            tgt = toks[bi, ti]
            lp_tok, lp_all = policy_logp(model, view, hist, ti, tgt, allowed)
            with torch.no_grad():
                lp_ref, _ = policy_logp(ref, view, hist, ti, tgt, allowed)
            d = lp_ref - lp_tok
            kl = torch.exp(d) - 1 - d
            loss = (-(adv[bi] * lp_tok) + a.beta * kl).sum() / n
            loss.backward()
            with torch.no_grad():
                p = lp_all.exp()
                tot_ent += float(-(p * lp_all.clamp(min=-30)).nan_to_num().sum(-1).sum())
                tot_kl += float(kl.sum())
                tot_scr += float(((tgt >= T.SCRATCH_BASE) & (tgt < T.COLOR_BASE)).sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        fr = solved.view(-1, a.group)[: (a.starts // 2) * 1].mean().item()
        frontier = solved[: (a.starts // 2) * a.group].mean().item()
        rec = dict(it=it, k=k_cur, solve_all=solved.mean().item(), solve_frontier=frontier,
                   mean_steps=steps.mean().item(), kl=tot_kl / n, entropy=tot_ent / n,
                   scratch_frac=tot_scr / n, n_steps=n, time=time.time() - t0)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if frontier >= a.promote and k_cur < a.kmax:
            k_cur += 1
        if (it + 1) % 50 == 0 or it == a.iters - 1:
            save_ckpt(os.path.join(a.out, "last.pt"), model, {"view": view_mode, "k": k_cur})


if __name__ == "__main__":
    main()
