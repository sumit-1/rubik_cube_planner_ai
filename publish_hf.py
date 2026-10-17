r"""Upload the 3-Face Partial-View Rubik's Cube AI model, white paper, checkpoints,
and a 100% FREE Static Hugging Face Space (interactive 3D/2D Web Simulator).

Usage:
    .\.venv\Scripts\python.exe publish_hf.py --space
    .\.venv\Scripts\python.exe publish_hf.py --space_only
"""
from __future__ import annotations

import argparse
import json
import os
from huggingface_hub import HfApi, login

import web_sim


UPLOAD_FILES = [
    # Documentation & Code
    "README.md",
    "WHITEPAPER.md",
    "requirements.txt",
    "eval.py",
    "web_sim.py",
    "pretrain_wm.py",
    "train_sft.py",
    "train_rl.py",
    "train_rnn.py",
    "train_rnn_v2.py",
    # Small teacher caches
    "cache/oll.json",
    "cache/pll.json",
    # Evaluation summaries & training logs
    "runs/eval_rnn_v3.json",
    "runs/eval_rnn_v3_resetmem.json",
    "runs/eval_rnn_mem.json",
    "runs/eval_ff2.json",
    "runs/eval_full.json",
    "runs/eval_pll.json",
    "runs/rnn_v3/log.jsonl",
    "runs/rnn_v2/log.jsonl",
    "runs/rnn_mem/log.jsonl",
    "runs/rnn_nobptt/log.jsonl",
    "runs/wm_pretrain/log.jsonl",
    "runs/sft_ufr3/log.jsonl",
    "runs/rl_scratch/log.jsonl",
    "runs/rl_noscratch/log.jsonl",
    # Featured Model Checkpoints
    "runs/rnn_v3/final.pt",
    "runs/rnn_v2/final.pt",
    "runs/rnn_mem/final.pt",
    "runs/rnn_nobptt/final.pt",
    "runs/wm_pretrain/final.pt",
    "runs/sft_ufr3/full.pt",
    "runs/rl_scratch/last.pt",
    "runs/rl_noscratch/last.pt",
]

SPACE_README = """---
title: 3-Face Partial-View Rubik's Cube AI Simulator
emoji: 🧊
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# 3-Face Partial-View Rubik's Cube AI Simulator (Static Web App)

Interactive 3D & 2D simulator for `RecurrentCubeLM` (5.35M parameters), which solves a $3\\times 3\\times 3$ Rubik's cube seeing **only 3 faces (`U, F, R` = 27 stickers)** at a time using a 16-slot gated recurrent memory feedback loop.

- **Model & Checkpoints:** [sumitk/rubik-cube-3face-recurrent-ai](https://huggingface.co/sumitk/rubik-cube-3face-recurrent-ai)
- **GitHub Repository:** [sumit-1/rubik_cube_planner_ai](https://github.com/sumit-1/rubik_cube_planner_ai)
"""


def build_static_space_bundle() -> tuple[bytes, bytes]:
    """Generate precomputed closed-loop AI traces across presets and build a standalone static index.html."""
    print("Generating real closed-loop AI solve traces for the free Static Space...", flush=True)
    ckpt = "runs/rnn_v3/final.pt"
    presets = [
        ("random", 25, [101, 102, 105, 108, 112]),
        ("f2l_oll_pll", 25, [201, 202, 203, 204]),
        ("f2l_pair", 25, [301, 302, 303, 304]),
        ("oll_pll", 25, [401, 402, 403, 404]),
        ("pll", 25, [501, 502, 503, 504]),
        ("short", 4, [601, 602, 603, 604]),
    ]
    bundle: dict[str, list[dict]] = {}
    for mode, depth, seeds in presets:
        mode_list = []
        actual_mode = "random" if mode == "short" else mode
        for sd in seeds:
            st, scr = web_sim.make_scramble(actual_mode, depth, seed=sd)
            normal_res = web_sim.solve_trace(ckpt, st, max_steps=140, reset_mem=False)
            wiped_res = web_sim.solve_trace(ckpt, st, max_steps=70, reset_mem=True)
            mode_list.append({
                "seed": sd,
                "scramble": scr,
                "start_state": st.tolist(),
                "start_badges": web_sim.stage_badges(st),
                "normal": normal_res,
                "wiped": wiped_res,
            })
        bundle[mode] = mode_list

    traces_json = json.dumps(bundle, separators=(",", ":")).encode("utf-8")

    # Convert web_sim.HTML_PAGE to use preloaded static traces.json when /api/* is unavailable
    html = web_sim.HTML_PAGE
    static_shim = r"""
// Static Hugging Face Space Adapter (uses precomputed real GPU traces from traces.json)
let STATIC_TRACES = null;
let staticCursor = {};
let activeSample = null;

async function ensureStaticTraces() {
  if (!STATIC_TRACES) {
    const r = await fetch("traces.json");
    STATIC_TRACES = await r.json();
  }
  return STATIC_TRACES;
}

window.addEventListener("DOMContentLoaded", async () => {
  // Replace backend fetch calls with instant static trace lookup
  const sel = document.getElementById("ckptSelect");
  sel.innerHTML = `<option value="runs/rnn_v3/final.pt">★ Recurrent v3 (5.35M: Active-Slot + W=64 + ViewPred + DAgger)</option>`;
  document.getElementById("customWrap").style.display = "none";
  const modeSel = document.getElementById("scrambleMode");
  Array.from(modeSel.options).forEach(o => { if (o.value === "custom") o.remove(); });
  await ensureStaticTraces();
  await doScramble();
  await doSolve();
}, { once: true });

doScramble = async function() {
  stopPlay();
  const db = await ensureStaticTraces();
  const mode = document.getElementById("scrambleMode").value;
  const list = db[mode] || db["random"];
  const idx = (staticCursor[mode] || 0) % list.length;
  staticCursor[mode] = idx + 1;
  activeSample = list[idx];
  currentStartState = activeSample.start_state;
  trace = [{
    step: 0,
    action: "<start>",
    state: activeSample.start_state,
    view_ufr: activeSample.start_state.slice(0, 27),
    pred_next_view: null,
    top_probs: [],
    mem_summary: null,
    badges: activeSample.start_badges
  }];
  document.getElementById("statusBanner").className = "status-banner";
  document.getElementById("statusText").textContent =
    `Scrambled (${mode}, sample #${idx + 1}). Click "Let AI Solve" to run RecurrentCubeLM.`;
  renderTrace();
  stepTo(0, false);
};

doSolve = async function() {
  stopPlay();
  await ensureStaticTraces();
  if (!activeSample) await doScramble();
  const resetMem = document.getElementById("resetMem").checked;
  const data = resetMem ? activeSample.wiped : activeSample.normal;
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
  stepTo(0, false);
  togglePlay();
};
</script>
"""
    html = html.replace("</script>", static_shim)
    return html.encode("utf-8"), traces_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="Hugging Face model repo ID (default: <username>/rubik-cube-3face-recurrent-ai)")
    ap.add_argument("--space", action="store_true", help="Also create & deploy a 100%% FREE Static Hugging Face Space")
    ap.add_argument("--space_only", action="store_true", help="Only create & deploy the free Static Hugging Face Space (skip model re-upload)")
    ap.add_argument("--space_repo", default=None, help="Hugging Face Space repo ID (default: <username>/rubik-cube-3face-sim)")
    ap.add_argument("--private", action="store_true", help="Create private HF repository")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", None), help="Hugging Face write token")
    args = ap.parse_args()

    if args.token:
        login(token=args.token, add_to_git_credential=False)

    api = HfApi(token=args.token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = args.repo or f"{username}/rubik-cube-3face-recurrent-ai"

    if not args.space_only:
        print(f"Creating / verifying Hugging Face model repo: https://huggingface.co/{repo_id}", flush=True)
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

        for folder in ["rubik", "tests"]:
            print(f"Uploading folder {folder}/ ...", flush=True)
            api.upload_folder(
                folder_path=folder,
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["__pycache__", "*.pyc"],
            )

        for rel_path in UPLOAD_FILES:
            if os.path.exists(rel_path):
                size_mb = os.path.getsize(rel_path) / (1024 * 1024)
                print(f"Uploading {rel_path} ({size_mb:.2f} MB) ...", flush=True)
                api.upload_file(
                    path_or_fileobj=rel_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="model",
                )

        print(f"\nModel & Whitepaper published to: https://huggingface.co/{repo_id}", flush=True)

    if args.space or args.space_only:
        space_id = args.space_repo or f"{username}/rubik-cube-3face-sim"
        print(f"\nCreating / verifying FREE Static Hugging Face Space: https://huggingface.co/spaces/{space_id}", flush=True)
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="static",
            private=args.private,
            exist_ok=True,
        )
        index_bytes, traces_bytes = build_static_space_bundle()
        api.upload_file(
            path_or_fileobj=SPACE_README.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )
        api.upload_file(
            path_or_fileobj=index_bytes,
            path_in_repo="index.html",
            repo_id=space_id,
            repo_type="space",
        )
        api.upload_file(
            path_or_fileobj=traces_bytes,
            path_in_repo="traces.json",
            repo_id=space_id,
            repo_type="space",
        )
        print(f"FREE Live 3D Simulator Static Space published to: https://huggingface.co/spaces/{space_id}", flush=True)


if __name__ == "__main__":
    main()
