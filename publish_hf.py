"""Upload the 3-Face Partial-View Rubik's Cube AI model, white paper, and checkpoints to Hugging Face Hub.

Usage:
    .\.venv\Scripts\python.exe publish_hf.py --repo <hf-username>/rubik-cube-3face-recurrent-ai
"""
from __future__ import annotations

import argparse
import os
from huggingface_hub import HfApi


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="Hugging Face repo ID (e.g. username/rubik-cube-3face-recurrent-ai)")
    ap.add_argument("--private", action="store_true", help="Create private HF repository")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", None), help="Optional HF write token")
    args = ap.parse_args()

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
        else:
            print(f"Skipping missing optional file: {rel_path}", flush=True)

    print(f"\nSuccessfully published to: https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
