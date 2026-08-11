#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Week-6 refinement hyperparameter sweep (tau, K) — G-Retriever, WebQSP
# Tunes Contribution-3 on the VALIDATION split (245 samples), never test.
#
# Loads the config #4 checkpoint ONCE and evaluates each (tau, K) combo by
# mutating the live model (see inference_sweep.py COMBOS). No training; each
# combo is generation-only over val.
#
# Usage on EC2:
#   tmux new -s sweep
#   bash ~/run_sweep.sh
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever

CKPT="output/webqsp_attn/model_name_adaptive_graph_llm_llm_model_name_8b_llm_frozen_False_max_txt_len_512_max_new_tokens_32_gnn_model_name_graphgps_patience_2_num_epochs_10_seed0_checkpoint_best.pth"

# --- Guards (same invariants as run_refine.sh) -------------------------------
if ! grep -q '_refine_batch' src/model/adaptive_graph_llm.py; then
  echo "ERROR: refinement not deployed in adaptive_graph_llm.py."; exit 1; fi
if [ ! -f src/dataset/utils/subgraph_refine.py ]; then
  echo "ERROR: subgraph_refine.py missing."; exit 1; fi
if [ ! -f inference_sweep.py ]; then
  echo "ERROR: inference_sweep.py missing (deploy week6_sweep/inference_sweep.py)."; exit 1; fi
if [ ! -f "$CKPT" ]; then
  echo "ERROR: config #4 checkpoint not found: $CKPT"; exit 1; fi
echo "Checkpoint OK ($(du -h "$CKPT" | cut -f1))."

n=$(ls dataset/webqsp/cached_graphs_attn/ 2>/dev/null | wc -l)
if [ "$n" -lt 4699 ]; then
  echo "ERROR: attention cache not ready ($n/4699)."; exit 1; fi
echo "Attention cache OK ($n graphs)."

python patch_refine_flags.py
if ! python inference_sweep.py --help 2>/dev/null | grep -q -- '--refine'; then
  echo "ERROR: --refine not recognised after patching config.py."; exit 1; fi
echo "config flags OK."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--dataset webqsp_attn --model_name adaptive_graph_llm \
  --gnn_model_name graphgps --confidence_head --refine \
  --llm_model_name 8b --llm_frozen False --eval_batch_size 4"

# --- GPU smoke: 1 combo, 1 batch on val — must load ckpt + write a CSV -------
echo "=== GPU SMOKE (SMOKE_BATCHES=1, val) ==="
SMOKE_BATCHES=1 SWEEP_SPLIT=val python inference_sweep.py $COMMON
echo "Smoke OK."

# --- Full sweep over the validation split ------------------------------------
echo "=== FULL SWEEP (val) ==="
SWEEP_SPLIT=val python inference_sweep.py $COMMON 2>&1 | tee ~/sweep_val.log

echo "=== SWEEP COMPLETE ==="
echo "Per-combo CSVs: output/webqsp_attn/..._seed0_val_<tag>.csv"
echo "Summary JSON:   output/webqsp_attn/sweep_summary_val_seed0.json"
echo "Log:            ~/sweep_val.log"
echo "Pull CSVs + summary, then run week6_sweep/aggregate_sweep.py locally."
echo "Remember: STOP the EC2 instance when done (do NOT terminate)."
