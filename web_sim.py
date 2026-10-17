"""Interactive Web Simulator for the Partial-View (3-Face) Rubik's Cube AI.

Run:
    .\.venv\Scripts\python.exe web_sim.py --port 8765
Then open http://127.0.0.1:8765 in your browser.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
import torch

from rubik import cube as C
from rubik import tokenizer as T
from rubik.data import gen
from rubik.rmodel import load_any
from rubik.rollout import allowed_mask
from rubik.teacher import cfop, cross, f2l, oll

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOADED_MODELS: dict[str, object] = {}

FEATURED_LABELS = {
    "runs/rnn_v3/final.pt": "★ Recurrent v3 (Best: Active-Slot + W=64 + ViewPred + DAgger)",
    "runs/rnn_v2/final.pt": "Recurrent v2 (W=64 + ViewPred + DAgger, fixed slots)",
    "runs/rnn_mem/final.pt": "Recurrent v1 (16-step BPTT memory)",
    "runs/rnn_nobptt/final.pt": "Ablation: Recurrent v1 NO-BPTT (detached memory)",
    "runs/wm_pretrain/final.pt": "World-Model Pretrain Only (untrained policy)",
    "runs/sft_ufr3/full.pt": "Feed-Forward Transformer (Forward Curriculum SFT)",
    "runs/rl_scratch/last.pt": "Feed-Forward + GRPO RL (with discrete scratchpad)",
    "runs/rl_noscratch/last.pt": "Feed-Forward + GRPO RL (no scratchpad)",
    "runs/sft_bwd2/full.pt": "Feed-Forward Transformer (Backward Curriculum SFT)",
}


def list_checkpoints() -> list[dict]:
    paths = sorted(glob.glob("runs/**/*.pt", recursive=True))
    paths = [p.replace("\\", "/") for p in paths]
    out = []
    # Put featured first in order
    seen = set()
    for p, label in FEATURED_LABELS.items():
        if os.path.exists(p):
            out.append({"path": p, "label": f"{label} [{p}]"})
            seen.add(p)
    for p in sorted(paths, key=lambda x: os.path.getmtime(x), reverse=True):
        if p not in seen:
            out.append({"path": p, "label": p})
    return out


def get_model(ckpt_path: str):
    ckpt_path = ckpt_path.replace("\\", "/")
    if ckpt_path not in LOADED_MODELS:
        print(f"[web_sim] Loading checkpoint {ckpt_path} on {DEVICE}...", flush=True)
        LOADED_MODELS[ckpt_path] = load_any(ckpt_path, DEVICE).eval()
    return LOADED_MODELS[ckpt_path]


def stage_badges(state: np.ndarray) -> dict:
    solved = bool(C.is_solved(state))
    canon, _ = C.canonicalize(state)
    c_ok = bool(cross.cross_solved(canon))
    ps = [bool(x) for x in f2l.pairs_solved(canon)] if c_ok else [False, False, False, False]
    f2l_ok = c_ok and all(ps)
    oll_ok = f2l_ok and bool(oll.oll_oriented(canon))
    return {
        "cross": c_ok or solved,
        "f2l_pairs": [True, True, True, True] if solved else ps,
        "f2l": f2l_ok or solved,
        "oll": oll_ok or solved,
        "solved": solved,
    }


def make_scramble(mode: str, depth: int, seed: int | None, custom_alg: str = "") -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    if mode == "custom":
        moves = C.parse_alg(custom_alg) if custom_alg.strip() else []
        s = C.apply_alg(C.SOLVED.copy(), moves)
        return s, moves
    if mode == "solved":
        return C.SOLVED.copy(), []
    if mode == "pll":
        ps = gen.pll_states()
        s = ps[int(rng.integers(len(ps)))].copy()
        return s, ["(random PLL state)"]
    if mode == "oll_pll":
        scr = C.random_scramble(rng, 25)
        s = C.apply_alg(C.SOLVED.copy(), scr)
        s, plan = cfop.advance(s, ["cross", "f2l0", "f2l1", "f2l2", "f2l3"])
        return s, scr + [m for _, m in plan]
    if mode == "f2l_oll_pll":
        scr = C.random_scramble(rng, 25)
        s = C.apply_alg(C.SOLVED.copy(), scr)
        s, plan = cfop.advance(s, ["cross"])
        return s, scr + [m for _, m in plan]
    if mode == "f2l_pair":
        scr = C.random_scramble(rng, 25)
        s = C.apply_alg(C.SOLVED.copy(), scr)
        s, plan = cfop.advance(s, ["cross", "f2l0", "f2l1", "f2l2"])
        # Rotate y' 3 times so the 4th pair (BR) is in the front-right (FR) slot
        for _ in range(3):
            s = C.apply_move(s, "y'")
        return s, scr + [m for _, m in plan] + ["y'", "y'", "y'"]
    # Default: random scramble of given depth
    scr = C.random_scramble(rng, int(depth))
    s = C.apply_alg(C.SOLVED.copy(), scr)
    return s, scr


@torch.no_grad()
def solve_trace(ckpt_path: str, start_state: np.ndarray, max_steps: int = 160,
                reset_mem: bool = False, allow_scratch: bool = False, greedy: bool = True,
                temperature: float = 0.8) -> dict:
    model = get_model(ckpt_path)
    is_rnn = getattr(model, "is_recurrent", False)
    view_mode = getattr(model, "extra", {}).get("view", "ufr")
    V = T.view_len(view_mode)
    allowed = allowed_mask(DEVICE, allow_scratch and not is_rnn, True, True)

    cur = start_state.astype(np.uint8).copy()
    steps = []
    # Step 0 record (initial state before any move)
    steps.append({
        "step": 0,
        "action": "<start>",
        "state": cur.tolist(),
        "view_ufr": cur[:27].tolist(),
        "pred_next_view": None,
        "top_probs": [],
        "mem_summary": None,
        "badges": stage_badges(cur),
    })

    if is_rnn:
        H = model.cfg.window
        recent = torch.zeros(1, H, dtype=torch.long, device=DEVICE)
        mem = model.init_mem(1)
    else:
        hist_ids: list[int] = []

    solved_at = 0 if C.is_solved(cur) else -1

    for t in range(1, max_steps + 1):
        if C.is_solved(cur):
            if solved_at < 0:
                solved_at = t - 1
            break
        view_t = torch.as_tensor(cur[:V].astype(np.int64) + T.COLOR_BASE, device=DEVICE)[None, :]
        if is_rnn:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                logits, new_mem = model.step(view_t, recent, mem)
        else:
            if hist_ids:
                h_tensor = torch.as_tensor(hist_ids, dtype=torch.long, device=DEVICE)[None, :]
            else:
                h_tensor = torch.zeros(1, 0, dtype=torch.long, device=DEVICE)
            hlen = torch.tensor([len(hist_ids)], dtype=torch.long, device=DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                logits = model(view_t, h_tensor, hlen)
            new_mem = None

        lg = logits[0].float().masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(lg / (1.0 if greedy else max(temperature, 1e-3)), dim=-1)
        top_p, top_i = torch.topk(probs, k=5)
        top_list = [{"tok": T.NAMES[int(idx)], "p": round(float(p), 4)} for p, idx in zip(top_p, top_i)]

        if greedy:
            tok_id = int(lg.argmax(-1).item())
        else:
            tok_id = int(torch.multinomial(probs, 1).item())

        tok_name = T.NAMES[tok_id]

        # Predict next 3-face view if recurrent model has view prediction head
        pred_view = None
        mem_vis = None
        if is_rnn and new_mem is not None:
            if hasattr(model, "predict_next_view"):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                    vp = model.predict_next_view(new_mem, torch.tensor([tok_id], device=DEVICE))
                pred_view = vp[0].argmax(-1).cpu().tolist()
            # Summarize 16 memory slots x 16 first channels (or RMS per slot) for visualization
            m_np = new_mem[0].float().cpu().numpy()  # [16, 256]
            # Compress 256 channels into 16 blocks of 16 via mean so UI can show a 16x16 grid
            mem_vis = np.tanh(m_np.reshape(16, 16, 16).mean(axis=-1)).round(3).tolist()

        # Advance state and recurrent buffers
        if T.is_cube_token(tok_id):
            cur = cur[T.PERM_TABLE[tok_id]]

        if is_rnn:
            mem = model.init_mem(1) if reset_mem else new_mem
            recent = torch.cat([recent[:, 1:], torch.tensor([[tok_id]], device=DEVICE)], dim=1)
        else:
            hist_ids.append(tok_id)

        b = stage_badges(cur)
        if b["solved"] and solved_at < 0:
            solved_at = t

        steps.append({
            "step": t,
            "action": tok_name,
            "state": cur.tolist(),
            "view_ufr": cur[:27].tolist(),
            "pred_next_view": pred_view,
            "top_probs": top_list,
            "mem_summary": mem_vis,
            "badges": b,
        })

        if tok_id == T.DONE or b["solved"]:
            break

    return {
        "ckpt": ckpt_path,
        "is_recurrent": is_rnn,
        "reset_mem": reset_mem,
        "solved": solved_at >= 0,
        "solved_step": solved_at,
        "total_steps": len(steps) - 1,
        "steps": steps,
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>3-Face Partial-View Rubik's Cube AI Simulator</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #171b26;
    --panel2: #1f2535;
    --border: #2c354d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --good: #3fb950;
    --warn: #d29922;
    --bad: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
  }
  h1 { margin: 0 0 4px 0; font-size: 18px; font-weight: 600; }
  .sub { color: var(--muted); margin-bottom: 14px; font-size: 12px; }
  .top-bar {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    background: var(--panel); border: 1px solid var(--border);
    padding: 12px; border-radius: 8px; margin-bottom: 14px;
  }
  select, input[type=text], input[type=number], button {
    background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 12px;
  }
  button { cursor: pointer; font-weight: 500; transition: background 0.12s; }
  button:hover { background: #2b3348; }
  button.primary { background: #1f6feb; border-color: #388bfd; color: #fff; }
  button.primary:hover { background: #388bfd; }
  button.success { background: #238636; border-color: #2ea043; color: #fff; }
  button.success:hover { background: #2ea043; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 360px;
    gap: 14px;
  }
  @media (max-width: 1250px) {
    .grid { grid-template-columns: 1fr 1fr; }
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .card h2 {
    margin: 0 0 8px 0; font-size: 14px; font-weight: 600;
    display: flex; justify-content: space-between; align-items: center;
  }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; background: var(--panel2);
    border: 1px solid var(--border); color: var(--muted);
  }
  .badge.on { background: rgba(63,185,80,0.18); border-color: var(--good); color: var(--good); }
  .badge.warn { background: rgba(210,153,34,0.18); border-color: var(--warn); color: var(--warn); }
  .stage-bar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
  canvas.cube3d {
    width: 100%; height: 270px; background: #0b0e14;
    border-radius: 6px; border: 1px solid var(--border); display: block;
  }
  .net-wrap {
    display: flex; justify-content: center; align-items: center;
    margin-top: 10px;
  }
  .tape {
    display: flex; flex-wrap: wrap; gap: 4px;
    max-height: 135px; overflow-y: auto;
    background: #0b0e14; padding: 8px; border-radius: 6px;
    border: 1px solid var(--border);
  }
  .tok {
    padding: 3px 7px; border-radius: 4px; font-family: monospace;
    font-size: 12px; cursor: pointer; background: var(--panel2);
    border: 1px solid var(--border); user-select: none;
  }
  .tok.rot { color: #d2a8ff; border-color: #6e40c9; }
  .tok.active { background: var(--accent); color: #000; font-weight: 700; border-color: #fff; }
  .prob-row {
    display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
    font-family: monospace; font-size: 12px;
  }
  .prob-bar-bg {
    flex: 1; height: 12px; background: #0b0e14; border-radius: 4px; overflow: hidden;
  }
  .prob-bar { height: 100%; background: var(--accent); }
  .move-pad {
    display: grid; grid-template-columns: repeat(9, 1fr); gap: 4px; margin-top: 8px;
  }
  .move-pad button { padding: 4px 2px; font-family: monospace; font-size: 11px; }
  .status-banner {
    padding: 8px 12px; border-radius: 6px; font-weight: 600; margin-bottom: 10px;
    background: var(--panel2); border: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .status-banner.solved { background: rgba(63,185,80,0.16); border-color: var(--good); color: var(--good); }
</style>
</head>
<body>
  <h1>Partial-View (3-Face) Rubik's Cube AI Simulator</h1>
  <div class="sub">
    The AI only ever sees the <b>3 front-top-right faces (U, F, R = 27 stickers)</b>. Hidden faces (D, L, B) are blacked out and must be remembered in the AI's recurrent feedback memory across whole-cube rotations (<code>x, y, z</code>) and face turns.
  </div>

  <div class="top-bar">
    <label><b>Checkpoint:</b>
      <select id="ckptSelect" style="min-width:340px;"></select>
    </label>

    <label><b>Preset / Scramble:</b>
      <select id="scrambleMode" onchange="onModeChange()">
        <option value="random">Full Random Scramble (25 moves)</option>
        <option value="f2l_oll_pll">Post-Cross -> Solve F2L + OLL + PLL (97.3% rate)</option>
        <option value="f2l_pair">Single F2L Pair (95.3% rate)</option>
        <option value="oll_pll">Last Layer: OLL + PLL (100% rate)</option>
        <option value="pll">Last Layer: PLL Only (100% rate)</option>
        <option value="short">Short Random Scramble (depth N)</option>
        <option value="custom">Custom Move Sequence</option>
      </select>
    </label>

    <label id="depthWrap" style="display:none;">Depth:
      <input type="number" id="scrambleDepth" value="4" min="1" max="30" style="width:55px;"/>
    </label>

    <label id="customWrap" style="display:none;">Moves:
      <input type="text" id="customAlg" placeholder="e.g. R U R' U' F2" style="width:180px;"/>
    </label>

    <button class="primary" onclick="doScramble()">Scramble Cube</button>
    <button class="success" onclick="doSolve()">Let AI Solve</button>

    <label style="display:flex;align-items:center;gap:4px;margin-left:6px;" title="Ablation: wipe the 16 recurrent memory vectors after every step">
      <input type="checkbox" id="resetMem"/> Wipe Feedback Memory (Ablation)
    </label>
  </div>

  <div id="statusBanner" class="status-banner">
    <span id="statusText">Ready — pick a scramble or click "Let AI Solve".</span>
    <div class="stage-bar" id="stageBadges" style="margin:0;"></div>
  </div>

  <div class="grid">
    <!-- Column 1: What the AI sees (ONLY 3 FACES: U, F, R) -->
    <div class="card">
      <h2>
        <span>1. What the AI Sees (3 Faces Only: U, F, R)</span>
        <span class="badge warn">27 Visible / 27 Hidden</span>
      </h2>
      <canvas id="canvasAI" class="cube3d" width="460" height="270"></canvas>
      <div class="net-wrap">
        <canvas id="netAI" width="360" height="220"></canvas>
      </div>
    </div>

    <!-- Column 2: Full 6-Face Ground Truth + AI Mental Next-View Prediction -->
    <div class="card">
      <h2>
        <span>2. Ground-Truth Cube (All 6 Faces, incl. Hidden D/L/B)</span>
        <span class="badge">Drag 3D cube to orbit</span>
      </h2>
      <canvas id="canvasFull" class="cube3d" width="460" height="270"></canvas>
      <div class="net-wrap">
        <canvas id="netFull" width="360" height="220"></canvas>
      </div>
    </div>

    <!-- Column 3: Playback, Memory Heatmap, Top-5 Probs, Manual Controls -->
    <div class="card">
      <h2>
        <span>3. AI Brain, Feedback Memory & Controls</span>
        <span class="badge" id="stepCounter">Step 0 / 0</span>
      </h2>

      <div style="display:flex; gap:6px; align-items:center; margin-bottom:8px;">
        <button onclick="stepTo(0)">|&lt;</button>
        <button onclick="stepDelta(-1)">&lt; Prev</button>
        <button class="primary" id="playBtn" onclick="togglePlay()">Play</button>
        <button onclick="stepDelta(1)">Next &gt;</button>
        <button onclick="stepTo(trace.length-1)">&gt;|</button>
        <label style="margin-left:auto;font-size:11px;color:var(--muted);">Speed
          <input type="range" id="speedSlider" min="40" max="600" value="160" style="width:70px;vertical-align:middle;"/>
        </label>
      </div>

      <input type="range" id="scrubber" min="0" max="0" value="0" oninput="stepTo(parseInt(this.value))" style="width:100%;margin-bottom:8px;"/>

      <div style="font-weight:600; margin-bottom:4px;">Action & Rotation History Tape (click any step):</div>
      <div id="tape" class="tape"></div>

      <div style="display:flex; gap:10px; margin-top:10px;">
        <div style="flex:1;">
          <div style="font-weight:600; margin-bottom:6px;">AI Top-5 Next Token Probs:</div>
          <div id="probBox"></div>
        </div>
        <div style="width:130px;">
          <div style="font-weight:600; margin-bottom:4px;">Feedback Memory:</div>
          <canvas id="memCanvas" width="128" height="128" style="border:1px solid var(--border);border-radius:4px;background:#0b0e14;"></canvas>
        </div>
      </div>

      <div style="margin-top:10px; display:flex; align-items:center; justify-content:space-between;">
        <div>
          <div style="font-weight:600;">AI's Mental Next-View Prediction (U/F/R):</div>
          <div style="font-size:11px; color:var(--muted);" id="vpAccText">Predicted 27 stickers after chosen action</div>
        </div>
        <canvas id="vpCanvas" width="135" height="50" style="border:1px solid var(--border);border-radius:4px;background:#0b0e14;"></canvas>
      </div>

      <div style="margin-top:10px; font-weight:600;">Manual Move / Rotation Pad (apply to current state):</div>
      <div class="move-pad" id="movePad"></div>
    </div>
  </div>

<script>
// Color palette for faces U(0)=White, R(1)=Red, F(2)=Green, D(3)=Yellow, L(4)=Orange, B(5)=Blue
const COLORS = ["#f4f6f8", "#e53935", "#43a047", "#fdd835", "#fb8c00", "#1e88e5"];
const FACE_NAMES = ["U", "R", "F", "D", "L", "B"];
const MANUAL_MOVES = [
  "U","U'","U2","R","R'","R2","F","F'","F2",
  "D","D'","D2","L","L'","L2","B","B'","B2",
  "x","x'","x2","y","y'","y2","z","z'","z2"
];

let trace = [];
let curIdx = 0;
let playTimer = null;
let currentStartState = null;

// 3D camera angles for the two 3D canvases
let camAI = { yaw: -0.68, pitch: 0.52 };
let camFull = { yaw: -0.68, pitch: 0.52, dragging: false, lx: 0, ly: 0 };

function initMovePad() {
  const pad = document.getElementById("movePad");
  pad.innerHTML = "";
  MANUAL_MOVES.forEach(m => {
    const b = document.createElement("button");
    b.textContent = m;
    b.onclick = () => applyManualMove(m);
    pad.appendChild(b);
  });
}

function onModeChange() {
  const mode = document.getElementById("scrambleMode").value;
  document.getElementById("depthWrap").style.display = (mode === "short") ? "" : "none";
  document.getElementById("customWrap").style.display = (mode === "custom") ? "" : "none";
}

async function loadCheckpoints() {
  const res = await fetch("/api/checkpoints");
  const data = await res.json();
  const sel = document.getElementById("ckptSelect");
  sel.innerHTML = "";
  data.checkpoints.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = c.label;
    sel.appendChild(opt);
  });
}

async function doScramble() {
  stopPlay();
  const mode = document.getElementById("scrambleMode").value;
  const depth = (mode === "short") ? parseInt(document.getElementById("scrambleDepth").value) : 25;
  const customAlg = document.getElementById("customAlg").value;
  const res = await fetch("/api/scramble", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ mode: mode === "short" ? "random" : mode, depth, custom_alg: customAlg })
  });
  const data = await res.json();
  currentStartState = data.state;
  trace = [{
    step: 0,
    action: "<start>",
    state: data.state,
    view_ufr: data.state.slice(0, 27),
    pred_next_view: null,
    top_probs: [],
    mem_summary: null,
    badges: data.badges
  }];
  document.getElementById("statusBanner").className = "status-banner";
  document.getElementById("statusText").textContent =
    `Scrambled (${mode}, ${data.scramble.length} moves). Click "Let AI Solve" to run the model.`;
  renderTrace();
  stepTo(0);
}

async function applyManualMove(m) {
  stopPlay();
  const st = trace.length ? trace[curIdx].state : currentStartState;
  const res = await fetch("/api/move", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ state: st, move: m })
  });
  const data = await res.json();
  currentStartState = data.state;
  trace = trace.slice(0, curIdx + 1);
  trace.push({
    step: trace.length,
    action: m,
    state: data.state,
    view_ufr: data.state.slice(0, 27),
    pred_next_view: null,
    top_probs: [],
    mem_summary: null,
    badges: data.badges
  });
  renderTrace();
  stepTo(trace.length - 1);
}

async function doSolve() {
  stopPlay();
  if (!currentStartState && trace.length === 0) {
    await doScramble();
  }
  const startState = trace.length ? trace[curIdx].state : currentStartState;
  const ckpt = document.getElementById("ckptSelect").value;
  const resetMem = document.getElementById("resetMem").checked;
  document.getElementById("statusText").textContent = `Running AI solver (${ckpt})...`;

  const res = await fetch("/api/solve", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ ckpt, state: startState, reset_mem: resetMem, max_steps: 160 })
  });
  const data = await res.json();
  trace = data.steps;
  const banner = document.getElementById("statusBanner");
  if (data.solved) {
    banner.className = "status-banner solved";
    document.getElementById("statusText").textContent =
      `SOLVED in ${data.solved_step} steps! (${data.ckpt}${data.reset_mem ? " — MEMORY WIPED" : ""})`;
  } else {
    banner.className = "status-banner";
    document.getElementById("statusText").textContent =
      `Did not finish full solve within ${data.total_steps} steps (${data.ckpt}${data.reset_mem ? " — MEMORY WIPED" : ""}).`;
  }
  renderTrace();
  stepTo(0);
  togglePlay();
}

function renderTrace() {
  const tape = document.getElementById("tape");
  tape.innerHTML = "";
  trace.forEach((item, i) => {
    const sp = document.createElement("span");
    const isRot = /^[xyz]/.test(item.action);
    sp.className = "tok" + (isRot ? " rot" : "") + (i === curIdx ? " active" : "");
    sp.textContent = i === 0 ? "0:start" : `${i}:${item.action}`;
    sp.onclick = () => { stopPlay(); stepTo(i); };
    tape.appendChild(sp);
  });
  const sc = document.getElementById("scrubber");
  sc.max = Math.max(0, trace.length - 1);
}

function stepTo(idx, animate = true) {
  if (!trace.length) return;
  const prevIdx = curIdx;
  curIdx = Math.max(0, Math.min(trace.length - 1, idx));
  document.getElementById("scrubber").value = curIdx;
  document.getElementById("stepCounter").textContent = `Step ${curIdx} / ${trace.length - 1}`;

  const tape = document.getElementById("tape");
  Array.from(tape.children).forEach((el, i) => {
    el.classList.toggle("active", i === curIdx);
    if (i === curIdx) el.scrollIntoView({ block: "nearest" });
  });

  const item = trace[curIdx];
  renderBadges(item.badges);
  draw2DNet(document.getElementById("netAI"), item.state, true);
  draw2DNet(document.getElementById("netFull"), item.state, false);
  renderProbs(item);
  renderMem(item.mem_summary);
  renderVP(item);

  // Trigger smooth 3D rotation animation when advancing 1 step forward
  if (animate && curIdx === prevIdx + 1 && curIdx > 0 && item.action && item.action !== "<done>") {
    const prevState = trace[prevIdx].state;
    start3DMoveAnim(prevState, item.state, item.action);
  } else {
    cancel3DAnim();
    draw3DCube(document.getElementById("canvasAI"), item.state, camAI, true, null);
    draw3DCube(document.getElementById("canvasFull"), item.state, camFull, false, null);
  }
}

function stepDelta(d) {
  stopPlay();
  stepTo(curIdx + d);
}

function togglePlay() {
  if (playTimer) {
    stopPlay();
    return;
  }
  if (curIdx >= trace.length - 1) curIdx = 0;
  document.getElementById("playBtn").textContent = "Pause";
  const tick = () => {
    if (curIdx < trace.length - 1) {
      stepTo(curIdx + 1);
      const delay = 640 - parseInt(document.getElementById("speedSlider").value);
      playTimer = setTimeout(tick, delay);
    } else {
      stopPlay();
    }
  };
  tick();
}

function stopPlay() {
  if (playTimer) clearTimeout(playTimer);
  playTimer = null;
  document.getElementById("playBtn").textContent = "Play";
}

function renderBadges(b) {
  if (!b) return;
  const box = document.getElementById("stageBadges");
  const items = [
    ["Cross", b.cross],
    ["F2L-1", b.f2l_pairs && b.f2l_pairs[0]],
    ["F2L-2", b.f2l_pairs && b.f2l_pairs[1]],
    ["F2L-3", b.f2l_pairs && b.f2l_pairs[2]],
    ["F2L-4", b.f2l_pairs && b.f2l_pairs[3]],
    ["OLL", b.oll],
    ["SOLVED", b.solved],
  ];
  box.innerHTML = items.map(([name, ok]) =>
    `<span class="badge ${ok ? 'on' : ''}">${name}: ${ok ? '✓' : '·'}</span>`
  ).join("");
}

function renderProbs(item) {
  const box = document.getElementById("probBox");
  const nextItem = trace[curIdx + 1];
  const probs = nextItem ? nextItem.top_probs : item.top_probs;
  const chosen = nextItem ? nextItem.action : null;
  if (!probs || !probs.length) {
    box.innerHTML = `<div style="color:var(--muted);font-size:11px;">(Click "Let AI Solve" to view action probabilities)</div>`;
    return;
  }
  box.innerHTML = probs.map(row => {
    const pct = Math.round(row.p * 100);
    const isChosen = row.tok === chosen;
    return `<div class="prob-row">
      <span style="width:36px;font-weight:${isChosen ? '700' : '400'};color:${isChosen ? '#58a6ff' : 'var(--text)'}">${row.tok}</span>
      <div class="prob-bar-bg"><div class="prob-bar" style="width:${pct}%"></div></div>
      <span style="width:34px;text-align:right;">${pct}%</span>
    </div>`;
  }).join("");
}

function renderMem(mem) {
  const cv = document.getElementById("memCanvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!mem) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "10px sans-serif";
    ctx.fillText("No recurrent memory", 14, 66);
    return;
  }
  const cell = cv.width / 16;
  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const v = mem[r][c]; // in [-1, 1]
      const pos = Math.max(0, v), neg = Math.max(0, -v);
      const R = Math.round(30 + 210 * pos);
      const G = Math.round(40 + 120 * (1 - Math.abs(v)));
      const B = Math.round(45 + 210 * neg);
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(c * cell, r * cell, cell - 0.5, cell - 0.5);
    }
  }
}

function renderVP(item) {
  const cv = document.getElementById("vpCanvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const nextItem = trace[curIdx + 1] || item;
  const pv = nextItem.pred_next_view;
  const actual = nextItem.view_ufr;
  if (!pv) {
    document.getElementById("vpAccText").textContent = "Next-view prediction (available on Recurrent v2/v3)";
    return;
  }
  let match = 0;
  const s = 13;
  for (let f = 0; f < 3; f++) {
    const ox = 4 + f * (3 * s + 5), oy = 6;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const idx = f * 9 + r * 3 + c;
        if (pv[idx] === actual[idx]) match++;
        ctx.fillStyle = COLORS[pv[idx]];
        ctx.fillRect(ox + c * s, oy + r * s, s - 1, s - 1);
      }
    }
  }
  document.getElementById("vpAccText").textContent =
    `After ${nextItem.action}: ${match}/27 visible stickers matched`;
}

// 2D unfolded net: U on top, L F R B across middle, D on bottom
function draw2DNet(cv, state, hideDLB) {
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const s = 21;
  const pad = 4;
  const F = 3 * s;
  // Face offsets in grid (col, row) for U(0), R(1), F(2), D(3), L(4), B(5)
  const layout = [
    [1, 0], // U
    [2, 1], // R
    [1, 1], // F
    [1, 2], // D
    [0, 1], // L
    [3, 1], // B
  ];
  const ox0 = 36, oy0 = 8;
  for (let fi = 0; fi < 6; fi++) {
    const [gx, gy] = layout[fi];
    const ox = ox0 + gx * (F + pad);
    const oy = oy0 + gy * (F + pad);
    const hidden = hideDLB && (fi >= 3);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const colorIdx = state[fi * 9 + r * 3 + c];
        ctx.fillStyle = hidden ? "#161a24" : COLORS[colorIdx];
        ctx.fillRect(ox + c * s, oy + r * s, s - 1.5, s - 1.5);
        if (hidden && r === 1 && c === 1) {
          ctx.fillStyle = "#556078";
          ctx.font = "bold 10px monospace";
          ctx.fillText("?" + FACE_NAMES[fi], ox + c * s + 2, oy + r * s + 14);
        }
      }
    }
    ctx.strokeStyle = hidden ? "#283044" : "#44506b";
    ctx.strokeRect(ox - 1, oy - 1, F + 1, F + 1);
  }
}

// 3D cube renderer using matches to rubik/cube.py face frames
const NORMALS = [
  [0, 1, 0],  // U
  [1, 0, 0],  // R
  [0, 0, 1],  // F
  [0, -1, 0], // D
  [-1, 0, 0], // L
  [0, 0, -1], // B
];
const FRAMES = [
  [[0, 0, -1], [1, 0, 0]],  // U
  [[0, 1, 0], [0, 0, -1]],  // R
  [[0, 1, 0], [1, 0, 0]],   // F
  [[0, 0, 1], [1, 0, 0]],   // D
  [[0, 1, 0], [0, 0, 1]],   // L
  [[0, 1, 0], [-1, 0, 0]],  // B
];

function rot3D(p, yaw, pitch) {
  // Yaw around Y, then pitch around X
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = p[0] * cy + p[2] * sy;
  const y1 = p[1];
  const z1 = -p[0] * sy + p[2] * cy;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x2 = x1;
  const y2 = y1 * cp - z1 * sp;
  const z2 = y1 * sp + z1 * cp;
  return [x2, y2, z2];
}

// Parse a move or whole-cube rotation into {axis: 'x'|'y'|'z', angle: radians, layer: -1|0|1|null, isWholeCube: bool}
function parseMoveSpec(action) {
  if (!action || action === "<start>" || action === "<done>" || action.startsWith("<s")) return null;
  const base = action[0];
  const mod = action.slice(1);
  const mult = (mod === "'") ? -1 : (mod === "2" ? 2 : 1);
  const halfPi = (Math.PI / 2) * mult;
  // Whole-cube rotations (x follows R, y follows U, z follows F)
  if (base === "x") return { axis: "x", angle: halfPi, layer: null, isWholeCube: true, name: action };
  if (base === "y") return { axis: "y", angle: halfPi, layer: null, isWholeCube: true, name: action };
  if (base === "z") return { axis: "z", angle: halfPi, layer: null, isWholeCube: true, name: action };
  // Face layer turns
  if (base === "R") return { axis: "x", angle: halfPi, layer: +1, isWholeCube: false, name: action };
  if (base === "L") return { axis: "x", angle: -halfPi, layer: -1, isWholeCube: false, name: action };
  if (base === "U") return { axis: "y", angle: halfPi, layer: +1, isWholeCube: false, name: action };
  if (base === "D") return { axis: "y", angle: -halfPi, layer: -1, isWholeCube: false, name: action };
  if (base === "F") return { axis: "z", angle: halfPi, layer: +1, isWholeCube: false, name: action };
  if (base === "B") return { axis: "z", angle: -halfPi, layer: -1, isWholeCube: false, name: action };
  return null;
}

function rotateAxis3D(p, axis, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const [x, y, z] = p;
  if (axis === "x") return [x, y * c + z * s, -y * s + z * c];
  if (axis === "y") return [x * c - z * s, y, x * s + z * c];
  if (axis === "z") return [x * c + y * s, -x * s + y * c, z];
  return p;
}

function stickerInMoveLayer(center3D, spec) {
  if (!spec) return false;
  if (spec.isWholeCube) return true;
  const idx = spec.axis === "x" ? 0 : (spec.axis === "y" ? 1 : 2);
  return (spec.layer > 0) ? (center3D[idx] > 0.35) : (center3D[idx] < -0.35);
}

let animRAF = null;

function cancel3DAnim() {
  if (animRAF) {
    cancelAnimationFrame(animRAF);
    animRAF = null;
  }
}

function start3DMoveAnim(prevState, targetState, action) {
  cancel3DAnim();
  const spec = parseMoveSpec(action);
  if (!spec) {
    draw3DCube(document.getElementById("canvasAI"), targetState, camAI, true, null);
    draw3DCube(document.getElementById("canvasFull"), targetState, camFull, false, null);
    return;
  }
  const sliderVal = parseInt(document.getElementById("speedSlider").value || "160");
  const dur = Math.max(90, Math.min(340, (640 - sliderVal) * 0.78));
  const t0 = performance.now();

  function frame(now) {
    const rawP = Math.min(1.0, (now - t0) / dur);
    // Smooth ease-in-out cubic
    const p = rawP < 0.5 ? 4 * rawP * rawP * rawP : 1 - Math.pow(-2 * rawP + 2, 3) / 2;
    if (rawP < 1.0) {
      const animState = { spec, theta: spec.angle * p, progress: p };
      draw3DCube(document.getElementById("canvasAI"), prevState, camAI, true, animState);
      draw3DCube(document.getElementById("canvasFull"), prevState, camFull, false, animState);
      animRAF = requestAnimationFrame(frame);
    } else {
      animRAF = null;
      const doneOverlay = { spec, theta: 0, progress: 1.0 };
      draw3DCube(document.getElementById("canvasAI"), targetState, camAI, true, doneOverlay);
      draw3DCube(document.getElementById("canvasFull"), targetState, camFull, false, doneOverlay);
    }
  }
  animRAF = requestAnimationFrame(frame);
}

function draw3DCube(cv, state, cam, hideDLB, anim) {
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const cx = cv.width / 2, cy = cv.height / 2 + 6;
  const scale = 68;
  const dist = 5.0;
  const quads = [];

  const activeSpec = (anim && anim.progress < 1.0) ? anim.spec : null;
  const theta = activeSpec ? anim.theta : 0;

  for (let fi = 0; fi < 6; fi++) {
    const n = NORMALS[fi];
    const [up, right] = FRAMES[fi];
    const hidden = hideDLB && (fi >= 3);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const idx = fi * 9 + r * 3 + c;
        const uc = (1 - r) * (2/3), vc = (c - 1) * (2/3);
        const center3D = [
          n[0] + up[0] * uc + right[0] * vc,
          n[1] + up[1] * uc + right[1] * vc,
          n[2] + up[2] * uc + right[2] * vc,
        ];
        const inLayer = activeSpec && stickerInMoveLayer(center3D, activeSpec);

        const u0 = uc - 1/3, u1 = uc + 1/3;
        const v0 = vc - 1/3, v1 = vc + 1/3;
        const pad = 0.035;
        let corners = [
          [u1 - pad, v0 + pad],
          [u1 - pad, v1 - pad],
          [u0 + pad, v1 - pad],
          [u0 + pad, v0 + pad],
        ].map(([uu, vv]) => [
          n[0] + up[0] * uu + right[0] * vv,
          n[1] + up[1] * uu + right[1] * vv,
          n[2] + up[2] * uu + right[2] * vv,
        ]);
        let curNorm = n;
        if (inLayer && theta !== 0) {
          corners = corners.map(p => rotateAxis3D(p, activeSpec.axis, theta));
          curNorm = rotateAxis3D(n, activeSpec.axis, theta);
        }

        const rCorners = corners.map(p => rot3D(p, cam.yaw, cam.pitch));
        const rNorm = rot3D(curNorm, cam.yaw, cam.pitch);
        const zAvg = rCorners.reduce((s, p) => s + p[2], 0) / 4;
        // Backface cull slightly
        if (rNorm[2] < -0.15) continue;
        quads.push({
          z: zAvg,
          pts: rCorners.map(([x, y, z]) => {
            const k = scale * dist / (dist - z * 0.55);
            return [cx + x * k, cy - y * k];
          }),
          color: hidden ? "#141822" : COLORS[state[idx]],
          hidden,
          isCenter: (r === 1 && c === 1),
          faceName: FACE_NAMES[fi],
        });
      }
    }
  }

  quads.sort((a, b) => a.z - b.z);
  quads.forEach(q => {
    ctx.beginPath();
    q.pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fillStyle = q.color;
    ctx.fill();
    ctx.strokeStyle = "#0b0e14";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (q.isCenter) {
      const mx = q.pts.reduce((s, p) => s + p[0], 0) / 4;
      const my = q.pts.reduce((s, p) => s + p[1], 0) / 4;
      ctx.fillStyle = q.hidden ? "#6e7681" : "rgba(0,0,0,0.65)";
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(q.hidden ? "?" + q.faceName : q.faceName, mx, my);
    }
  });

  // Draw live action / whole-cube rotation HUD overlay on top-left of 3D canvas
  const curItem = trace[curIdx];
  const curAct = curItem ? curItem.action : null;
  if (curAct && curAct !== "<start>") {
    const spec = parseMoveSpec(curAct);
    const isRot = spec && spec.isWholeCube;
    const badgeText = isRot
      ? `WHOLE-CUBE ROTATION: ${curAct} (Re-orienting View)`
      : (curAct === "<done>" ? "SOLVED (<done>)" : `FACE MOVE: ${curAct}`);
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(badgeText).width;
    ctx.fillStyle = isRot ? "rgba(110, 64, 201, 0.28)" : "rgba(31, 111, 235, 0.22)";
    ctx.strokeStyle = isRot ? "#d2a8ff" : "#58a6ff";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(10, 10, tw + 16, 22, 5);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = isRot ? "#e2c5ff" : "#a5d6ff";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, 18, 21);
  }
}

// Orbit drag on the Full 3D canvas
const cf = document.getElementById("canvasFull");
cf.addEventListener("mousedown", e => {
  camFull.dragging = true; camFull.lx = e.clientX; camFull.ly = e.clientY;
});
window.addEventListener("mouseup", () => camFull.dragging = false);
window.addEventListener("mousemove", e => {
  if (!camFull.dragging || !trace.length) return;
  camFull.yaw += (e.clientX - camFull.lx) * 0.01;
  camFull.pitch = Math.max(-1.3, Math.min(1.3, camFull.pitch + (e.clientY - camFull.ly) * 0.01));
  camFull.lx = e.clientX; camFull.ly = e.clientY;
  draw3DCube(cf, trace[curIdx].state, camFull, false);
});

window.addEventListener("DOMContentLoaded", async () => {
  initMovePad();
  await loadCheckpoints();
  await doScramble();
});
</script>
</body>
</html>
"""


class SimHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, obj: dict, code: int = 200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif parsed.path == "/api/checkpoints":
            self._send_json({"checkpoints": list_checkpoints()})
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        try:
            if parsed.path == "/api/scramble":
                mode = body.get("mode", "random")
                depth = int(body.get("depth", 25))
                seed = body.get("seed", None)
                custom_alg = body.get("custom_alg", "")
                st, scr = make_scramble(mode, depth, seed, custom_alg)
                self._send_json({
                    "state": st.tolist(),
                    "scramble": scr,
                    "badges": stage_badges(st),
                })
            elif parsed.path == "/api/move":
                st = np.array(body["state"], dtype=np.uint8)
                mv = body["move"]
                st2 = C.apply_alg(st, [mv])
                self._send_json({
                    "state": st2.tolist(),
                    "badges": stage_badges(st2),
                })
            elif parsed.path == "/api/solve":
                ckpt = body["ckpt"]
                st = np.array(body["state"], dtype=np.uint8)
                reset_mem = bool(body.get("reset_mem", False))
                max_steps = int(body.get("max_steps", 160))
                res = solve_trace(ckpt, st, max_steps=max_steps, reset_mem=reset_mem)
                self._send_json(res)
            else:
                self.send_error(404)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, code=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    # Warm default checkpoint
    if os.path.exists("runs/rnn_v3/final.pt"):
        get_model("runs/rnn_v3/final.pt")
    srv = HTTPServer((a.host, a.port), SimHandler)
    print(f"Rubik 3-Face AI Web Simulator running at http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
