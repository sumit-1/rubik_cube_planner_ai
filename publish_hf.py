"""Upload the 3-Face Partial-View Rubik's Cube AI model, white paper, checkpoints,
and optionally a live interactive 3D Web Simulator Space to Hugging Face Hub.

Usage:
    # 1. Log in once (or pass --token hf_xxx):
    .\.venv\Scripts\hf.exe auth login

    # 2. Publish Model + Checkpoints + Whitepaper + Live 3D Space:
    .\.venv\Scripts\python.exe publish_hf.py --space
"""
from __future__ import annotations

import argparse
import os
from huggingface_hub import HfApi, login


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
    "runs/wm_pretrain/wm_final.pt",
    "runs/sft_ufr3/full.pt",
    "runs/rl_scratch/last.pt",
    "runs/rl_noscratch/last.pt",
]

SPACE_README = """---
title: 3-Face Partial-View Rubik's Cube AI Simulator
emoji: 🧊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 3-Face Partial-View Rubik's Cube AI Simulator

Interactive 3D & 2D simulator for `RecurrentCubeLM` (5.35M parameters), which solves a $3\\times 3\\times 3$ Rubik's cube seeing **only 3 faces (`U, F, R` = 27 stickers)** at a time using a 16-slot gated recurrent memory feedback loop.
"""

SPACE_DOCKERFILE = """FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch numpy numba

COPY rubik /app/rubik
COPY cache /app/cache
COPY runs/rnn_v3/final.pt /app/runs/rnn_v3/final.pt
COPY runs/rnn_nobptt/final.pt /app/runs/rnn_nobptt/final.pt
COPY web_sim.py /app/web_sim.py

EXPOSE 7860
CMD ["python", "web_sim.py", "--host", "0.0.0.0", "--port", "7860"]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="Hugging Face model repo ID (default: <username>/rubik-cube-3face-recurrent-ai)")
    ap.add_argument("--space", action="store_true", help="Also create & deploy a live Docker Hugging Face Space for web_sim.py")
    ap.add_argument("--space_repo", default=None, help="Hugging Face Space repo ID (default: <username>/rubik-cube-3face-sim)")
    ap.add_argument("--private", action="store_true", help="Create private HF repository")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", None), help="Hugging Face write token (or run `hf auth login` first)")
    args = ap.parse_args()

    if args.token:
        login(token=args.token, add_to_git_credential=False)

    api = HfApi(token=args.token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = args.repo or f"{username}/rubik-cube-3face-recurrent-ai"

    print(f"Creating / verifying Hugging Face model repo: https://huggingface.co/{repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Upload rubik/ and tests/ source folders
    for folder in ["rubik", "tests"]:
        print(f"Uploading folder {folder}/ ...", flush=True)
        api.upload_folder(
            folder_path=folder,
            path_in_repo=folder,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["__pycache__", "*.pyc"],
        )

    # Upload individual files and checkpoints
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

    if args.space:
        space_id = args.space_repo or f"{username}/rubik-cube-3face-sim"
        print(f"\nCreating / verifying Hugging Face Space: https://huggingface.co/spaces/{space_id}", flush=True)
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=SPACE_README.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )
        api.upload_file(
            path_or_fileobj=SPACE_DOCKERFILE.encode("utf-8"),
            path_in_repo="Dockerfile",
            repo_id=space_id,
            repo_type="space",
        )
        api.upload_folder(
            folder_path="rubik",
            path_in_repo="rubik",
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=["__pycache__", "*.pyc"],
        )
        for sp_file in [
            "requirements.txt",
            "web_sim.py",
            "cache/oll.json",
            "cache/pll.json",
            "runs/rnn_v3/final.pt",
            "runs/rnn_nobptt/final.pt",
        ]:
            if os.path.exists(sp_file):
                api.upload_file(
                    path_or_fileobj=sp_file,
                    path_in_repo=sp_file,
                    repo_id=space_id,
                    repo_type="space",
                )
        print(f"Live 3D Simulator Space published to: https://huggingface.co/spaces/{space_id}", flush=True)


if __name__ == "__main__":
    main()
