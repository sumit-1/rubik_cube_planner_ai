"""Closed-loop batched rollouts: the model acts, the simulator applies, the view is recomputed."""
from __future__ import annotations

import torch

from . import tokenizer as T
from .gpucube import BatchCube


def allowed_mask(device, allow_scratch: bool, allow_done: bool = True, allow_rot: bool = True):
    m = torch.zeros(T.VOCAB, dtype=torch.bool, device=device)
    m[T.MOVE_BASE:T.ROT_BASE] = True
    if allow_rot:
        m[T.ROT_BASE:T.SCRATCH_BASE] = True
    if allow_scratch:
        m[T.SCRATCH_BASE:T.COLOR_BASE] = True
    if allow_done:
        m[T.DONE] = True
    return m


@torch.no_grad()
def rollout_rnn(model, env: BatchCube, state0: torch.Tensor, max_steps: int, greedy: bool = True,
                temperature: float = 1.0, auto_stop: bool = True, prefix: list[int] | None = None,
                record: bool = False, view_mode: str = "ufr", allow_rot: bool = True,
                reset_mem: bool = False, **_):
    """Closed-loop rollout for the recurrent duplex model (memory carried between steps)."""
    dev = state0.device
    B = state0.shape[0]
    H = model.cfg.window
    state = state0.clone()
    hist = torch.zeros(B, max_steps, dtype=torch.long, device=dev)
    hlen = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    recent = torch.zeros(B, H, dtype=torch.long, device=dev)
    mem = model.init_mem(B)
    solved_ever = env.solved_any_orientation(state)
    alive = ~solved_ever if auto_stop else torch.ones(B, dtype=torch.bool, device=dev)
    solve_step = torch.where(solved_ever, torch.zeros_like(hlen), torch.full_like(hlen, -1))
    allowed = allowed_mask(dev, False, True, allow_rot)
    rec_states, rec_tokens, rec_alive = [], [], []
    done_flag = torch.zeros(B, dtype=torch.bool, device=dev)
    was_training = model.training
    model.eval()
    for step in range(max_steps):
        if not alive.any():
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, new_mem = model.step(env.view(state, view_mode), recent, mem)
        if prefix is not None and step < len(prefix):
            tok = torch.full((B,), int(prefix[step]), dtype=torch.long, device=dev)
        else:
            logits = logits.float().masked_fill(~allowed, float("-inf"))
            if greedy:
                tok = logits.argmax(-1)
            else:
                tok = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
        tok = torch.where(alive, tok, torch.full_like(tok, T.PAD))
        mem = torch.where(alive[:, None, None], new_mem, mem)
        if reset_mem:  # ablation: wipe the feedback memory every step
            mem = model.init_mem(B)
        recent = torch.where(alive[:, None], torch.cat([recent[:, 1:], tok[:, None]], 1), recent)
        if record:
            rec_states.append(state.to(torch.uint8))
            rec_tokens.append(tok)
            rec_alive.append(alive.clone())
        state = env.step(state, tok)
        hist[ar, hlen.clamp(max=max_steps - 1)] = torch.where(alive, tok, hist[ar, hlen.clamp(max=max_steps - 1)])
        hlen = hlen + alive.long()
        solved = env.solved_any_orientation(state)
        newly = solved & (solve_step < 0)
        solve_step = torch.where(newly, torch.full_like(solve_step, step + 1), solve_step)
        solved_ever |= solved
        done = tok == T.DONE
        done_flag |= done & alive
        alive = alive & ~done
        if auto_stop:
            alive = alive & ~solved
    model.train(was_training)
    out = dict(state=state, hist=hist, hist_len=hlen, solved=solved_ever, solve_step=solve_step,
               done=done_flag)
    if record:
        out["rec_states"] = torch.stack(rec_states, 1) if rec_states else None
        out["rec_tokens"] = torch.stack(rec_tokens, 1) if rec_tokens else None
        out["rec_alive"] = torch.stack(rec_alive, 1) if rec_alive else None
    return out


@torch.no_grad()
def rollout(model, env: BatchCube, state0: torch.Tensor, max_steps: int, greedy: bool = True,
            temperature: float = 1.0, allow_scratch: bool = False, auto_stop: bool = True,
            prefix: list[int] | None = None, record: bool = False, view_mode: str = "ufr",
            allow_rot: bool = True, reset_mem: bool = False):
    if getattr(model, "is_recurrent", False):
        return rollout_rnn(model, env, state0, max_steps, greedy, temperature, auto_stop, prefix,
                           record, view_mode, allow_rot, reset_mem)
    dev = state0.device
    B = state0.shape[0]
    state = state0.clone()
    hist = torch.zeros(B, max_steps, dtype=torch.long, device=dev)
    hlen = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    solved_ever = env.solved_any_orientation(state)
    alive = ~solved_ever if auto_stop else torch.ones(B, dtype=torch.bool, device=dev)
    solve_step = torch.where(solved_ever, torch.zeros_like(hlen), torch.full_like(hlen, -1))
    allowed = allowed_mask(dev, allow_scratch, True, allow_rot)
    rec_states, rec_tokens, rec_alive = [], [], []
    done_flag = torch.zeros(B, dtype=torch.bool, device=dev)
    was_training = model.training
    model.eval()
    for step in range(max_steps):
        if not alive.any():
            break
        if prefix is not None and step < len(prefix):
            tok = torch.full((B,), int(prefix[step]), dtype=torch.long, device=dev)
        else:
            L = int(hlen.max().item())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(env.view(state, view_mode), hist[:, :L], hlen)
            logits = logits.float().masked_fill(~allowed, float("-inf"))
            if greedy:
                tok = logits.argmax(-1)
            else:
                tok = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
        tok = torch.where(alive, tok, torch.full_like(tok, T.PAD))
        if record:
            rec_states.append(state.to(torch.uint8))
            rec_tokens.append(tok)
            rec_alive.append(alive.clone())
        state = env.step(state, tok)
        hist[ar, hlen.clamp(max=max_steps - 1)] = torch.where(alive, tok, hist[ar, hlen.clamp(max=max_steps - 1)])
        hlen = hlen + alive.long()
        solved = env.solved_any_orientation(state)
        newly = solved & (solve_step < 0)
        solve_step = torch.where(newly, torch.full_like(solve_step, step + 1), solve_step)
        solved_ever |= solved
        done = tok == T.DONE
        done_flag |= done & alive
        alive = alive & ~done
        if auto_stop:
            alive = alive & ~solved
    model.train(was_training)
    out = dict(state=state, hist=hist, hist_len=hlen, solved=solved_ever, solve_step=solve_step,
               done=done_flag)
    if record:
        out["rec_states"] = torch.stack(rec_states, 1) if rec_states else None
        out["rec_tokens"] = torch.stack(rec_tokens, 1) if rec_tokens else None
        out["rec_alive"] = torch.stack(rec_alive, 1) if rec_alive else None
    return out
