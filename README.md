---
license: mit
tags:
  - rubiks-cube
  - partial-observability
  - pomdp
  - recurrent-transformer
  - world-model
  - reinforcement-learning
  - dagger
  - pytorch
pipeline_tag: reinforcement-learning
---

# Solving the Rubik's Cube Under Strict 3-Face Partial Observability via Recurrent Attention Feedback

[![Tests](https://img.shields.io/badge/pytest-57%20passed-brightgreen)](#quick-start)
[![Model](https://img.shields.io/badge/params-5.35M%20RecurrentCubeLM-blue)](#architecture-recurrentcubelm)
[![Observability](https://img.shields.io/badge/observation-3%20Faces%20Only%20(U%2FF%2FR%20%3D%2027%20stickers)-orange)](#key-benchmark-results-256-episodes--task-3-face-view-only)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Live%203D%20Simulator-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/sumitk/rubik-cube-3face-sim)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20%26%20Checkpoints-Hugging%20Face%20Hub-purple)](https://huggingface.co/sumitk/rubik-cube-3face-recurrent-ai)

- 🧊 **Live Interactive 3D & 2D Simulator (Hugging Face Space):** **[https://huggingface.co/spaces/sumitk/rubik-cube-3face-sim](https://huggingface.co/spaces/sumitk/rubik-cube-3face-sim)**
- 🤗 **Trained Checkpoints & Model Repository (Hugging Face Hub):** **[https://huggingface.co/sumitk/rubik-cube-3face-recurrent-ai](https://huggingface.co/sumitk/rubik-cube-3face-recurrent-ai)**
- 📄 **Full Technical White Paper:** **[`WHITEPAPER.md`](WHITEPAPER.md)**

An attention-based neural network trained from scratch to solve a $3\times 3\times 3$ Rubik's cube in closed loop under **strict 3-face partial observability**:
- **3-Face Partial View (`U, F, R` = 27 stickers):** At every step $t$, the agent only sees the 27 stickers of the Top (`U`), Front (`F`), and Right (`R`) faces. The Back (`B`), Left (`L`), and Down (`D`) faces are completely occluded.
- **View Replacement (POMDP):** Each step's 27-sticker view replaces the previous view. Whole-cube rotations (`x, y, z`) bring occluded faces into view while pushing previously visible faces out of view.
- **Recurrent Continuous Attention Feedback (`RecurrentCubeLM`, 5.35M params):** Maintains 16 gated continuous memory vectors across time, paired with a cross-attention next-view world-model head (`predict_next_view`).
- **Zero Search & Zero Kociemba at Inference:** The neural network outputs raw face turns (`U, R, F, D, L, B`), whole-cube rotations (`x, y, z`), and `<done>` in a greedy closed loop with zero external search or state scaffolding.

---

## Key Benchmark Results (256 Episodes / Task, 3-Face View Only)

| Task | Feed-Forward SFT (`sft_ufr3`, 14.2M) | Feed-Forward + GRPO RL (`rl_scratch`) | Recurrent No-BPTT (`rnn_nobptt`, $W=1$) | Recurrent $W=16$ (`rnn_mem`) | **Recurrent v3 (`rnn_v3/final.pt`, 5.35M)** | **Ablation: Memory Wiped (`--reset_mem`)** |
|---|---|---|---|---|---|---|
| `pll` (Last-Layer Permutation) | 60.2% | — | 93.4% | 91.8% | **100.0%** | 0.8% |
| `oll` (Last-Layer Orientation) | 44.5% | — | 57.8% | 59.4% | **100.0%** | 5.5% |
| `oll_pll` (Full Last Layer to `<done>`) | 20.7% | — | 53.1% | 53.5% | **100.0%** | **0.0%** |
| `f2l_pair` (Single F2L Pair) | 11.7% | — | 13.3% | 15.6% | **95.3%** | **0.0%** |
| `f2l` (All 4 F2L Pairs Chained) | 0.0% | — | 0.0% | 0.0% | **97.7%** | **0.0%** |
| `f2l_oll_pll` (4 F2L Pairs + OLL + PLL) | 0.0% | — | 0.0% | 0.0% | **97.3%** | **0.0%** |
| **`full` (Full 25-Move Scramble to `<done>`)** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **76.6%** | **0.0%** |
| `rev` Depth 1 / 2 / 3 / 4 | 100 / 100 / 97.7 / 72.3% | 100 / 100 / 98.4 / 75.4% | — / — / — / 46.1% | — / — / 92.6 / 67.2% | **100 / 100 / 99.6 / 83.2%** | — / — / — / 5.9% |

---

## Architecture (`RecurrentCubeLM`)

```mermaid
flowchart TB
  subgraph inputs [76 Input Tokens at Step t - Bidirectional Full Self-Attention]
    direction LR
    M_in["[0..15] Memory Input<br/>RMSNorm(M_{t-1}) + mem_pos_emb<br/>(16 x 256)"]
    V_in["[16..42] Visible 3-Face View<br/>tok_emb(o_t) + view_slot_emb<br/>(27 x 256)"]
    A_in["[43..58] Recent Action Window<br/>tok_emb(a_{t-16:t-1}) + act_pos_emb<br/>(16 x 256)"]
    Q_act["[59] Action Query<br/>act_query<br/>(1 x 256)"]
    Q_wr["[60..75] Memory Write Queries<br/>write_queries<br/>(16 x 256)"]
  end

  inputs --> Trunk["6x Bidirectional Transformer Blocks<br/>(d_model = 256, n_heads = 4, d_ff = 688, SwiGLU, RMSNorm)"]

  Trunk --> H_act["h_act = h[59]"]
  Trunk --> H_wr["h_write = h[60..75]"]

  H_act --> HeadAct["Action Head: Linear(256 -> 64)<br/>Logits over 18 Moves, 9 Rotations, <done>"]
  H_wr --> MemUpdate["Gated Residual Memory Update<br/>M_t = g * M_{t-1} + (1 - g) * cand<br/>(16 x 256 Continuous State)"]
  MemUpdate --> NextViewHead["Cross-Attention Next-View Head<br/>Predicts Next 27 Visible Stickers o_{t+1} (27 x 6)"]
```

---

## Quick Start

### 1. Install Dependencies & Run Verification Suite
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

### 2. Launch the Interactive 3D & 2D Web Simulator
Visualizes the 3-face partial view (`U/F/R`) alongside the 6-face ground truth, with smooth 3D whole-cube rotation (`x, y, z`) and face-twist animations, live 16-slot memory heatmap, mental next-view prediction, and memory-wiping ablation toggle:
```powershell
.\.venv\Scripts\python.exe web_sim.py --port 8765
# Open http://127.0.0.1:8765 in your browser
```

### 3. Evaluate Checkpoints in Closed Loop
```powershell
# Standard closed-loop evaluation
.\.venv\Scripts\python.exe eval.py --ckpt runs/rnn_v3/final.pt --n 256

# Causal memory-wiping ablation (--reset_mem)
.\.venv\Scripts\python.exe eval.py --ckpt runs/rnn_v3/final.pt --n 256 --reset_mem
```

### 4. Generate Trajectories & Train from Scratch
```powershell
# 1. Generate active-slot CFOP trajectories (builds PDB caches automatically on first run)
.\.venv\Scripts\python.exe -m rubik.data.gen --out data_rnn3 --stage all --n 60000 --workers 8 --p_scratch 0.0 --p_rot 0.0 --inspect x_z2 --stage_glance --f2l_rotate --canonical_start

# 2. Pretrain 3-face next-view world model on random move+rotation walks
.\.venv\Scripts\python.exe pretrain_wm.py --out runs/wm_pretrain --iters 1500 --batch 256 --seq_len 24 --lr 5e-4

# 3. Train Recurrent v3 solver with W=64 full BPTT + next-view loss + online DAgger
.\.venv\Scripts\python.exe train_rnn_v2.py --data data_rnn3 --out runs/rnn_v3 --init runs/wm_pretrain/wm_final.pt --bptt 64 --batch 192 --lr 4e-4 --vp_weight 0.25 --short_iters 400 --all_iters 1600 --dagger_every 4 --dagger_batch 64 --f2l_rotate --canonical_start
```

See [`WHITEPAPER.md`](WHITEPAPER.md) for the complete mathematical formulation, all six experimental iterations, root-cause analyses of negative results, and full command-line reference.
