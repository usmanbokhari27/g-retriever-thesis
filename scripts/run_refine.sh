#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Iterative-refinement eval (ablation config #5) — G-Retriever, WebQSP
# Contribution 3: test-time subgraph refinement driven by the confidence head.
#
# Loads the ALREADY-TRAINED config #4 checkpoint (LoRA + GraphGPS + projector +
# confidence head) and runs the test set twice:
#   1) --refine OFF  -> sanity re-eval, should reproduce config #4 (Hit@1 72.54)
#   2) --refine ON   -> config #5, the Contribution-3 result
# No training happens, so each pass is ~1 h (eval only), not ~24 h.
#
# Self-contained: verifies the Week-5 files are deployed, the checkpoint exists,
# the attention+RWSE cache is ready, and the argparse flags are present.
#
# Usage on EC2:
#   tmux new -s refine
#   bash ~/run_refine.sh
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever

CKPT="output/webqsp_attn/model_name_adaptive_graph_llm_llm_model_name_8b_llm_frozen_False_max_txt_len_512_max_new_tokens_32_gnn_model_name_graphgps_patience_2_num_epochs_10_seed0_checkpoint_best.pth"

# --- Guard 0: Week-5 code must be deployed -----------------------------------
if ! grep -q '_refine_batch' src/model/adaptive_graph_llm.py; then
  echo "ERROR: src/model/adaptive_graph_llm.py has no refinement (_refine_batch)."
  echo "Deploy week5_port/adaptive_graph_llm.py -> src/model/adaptive_graph_llm.py first."
  exit 1
fi
if [ ! -f src/dataset/utils/subgraph_refine.py ]; then
  echo "ERROR: src/dataset/utils/subgraph_refine.py missing (deploy week5_port/subgraph_refine.py)."
  exit 1
fi
if [ ! -f inference_refine.py ]; then
  echo "ERROR: inference_refine.py missing (deploy week5_port/inference_refine.py)."
  exit 1
fi

# --- Guard 1: the config #4 checkpoint must exist ----------------------------
if [ ! -f "$CKPT" ]; then
  echo "ERROR: config #4 checkpoint not found:"
  echo "  $CKPT"
  echo "Refinement needs the trained confidence head — re-run config #4 first."
  exit 1
fi
echo "Checkpoint OK ($(du -h "$CKPT" | cut -f1))."

# --- Guard 2: attention cache + RWSE must be present -------------------------
n=$(ls dataset/webqsp/cached_graphs_attn/ 2>/dev/null | wc -l)
if [ "$n" -lt 4699 ]; then
  echo "ERROR: attention cache not ready ($n/4699). Build: PYTHONPATH=. python -m src.dataset.webqsp_attn"
  exit 1
fi
has_pe=$(PYTHONPATH=. python -c "import torch,glob; g=torch.load(sorted(glob.glob('dataset/webqsp/cached_graphs_attn/*.pt'))[0], weights_only=False); print(1 if (getattr(g,'pe',None) is not None and g.pe.shape[1]==20) else 0)")
if [ "$has_pe" != "1" ]; then
  echo "ERROR: cached attention graphs have no RWSE (pe). Add: PYTHONPATH=. python -m add_rwse_attn"
  exit 1
fi
echo "Attention cache OK ($n graphs, RWSE present)."

# --- Guard 3: ensure the Week-5 argparse flags exist -------------------------
python patch_refine_flags.py
if ! python inference_refine.py --help 2>/dev/null | grep -q -- '--refine'; then
  echo "ERROR: --refine not recognised after patching config.py."
  exit 1
fi
echo "config flags OK (--refine recognised)."

# Sanity: the CPU-only subgraph ops reproduce the cached level-0 subgraph.
PYTHONPATH=. python -m src.dataset.utils.subgraph_refine

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--dataset webqsp_attn --model_name adaptive_graph_llm \
  --gnn_model_name graphgps --confidence_head \
  --llm_model_name 8b --llm_frozen False --eval_batch_size 4"

echo "=== PASS 1/2: refinement OFF (sanity — expect Hit@1 ~= 72.54) ==="
python inference_refine.py $COMMON 2>&1 | tee ~/refine_off_webqsp.log

echo "=== PASS 2/2: refinement ON (config #5, Contribution 3) ==="
python inference_refine.py $COMMON --refine --refine_tau 0.5 \
  --refine_max_iters 2 --refine_max_nodes 60 2>&1 | tee ~/refine_on_webqsp.log

echo "=== REFINEMENT RUN COMPLETE ==="
echo "OFF (config #4 sanity):  ~/refine_off_webqsp.log"
echo "ON  (config #5):         ~/refine_on_webqsp.log"
echo "Compare Test Hit@1: config #5 vs config #4 (72.54), #2 (72.17), anchor (61.61)."
echo "Predictions carry 'confidence' + 'num_refine_iters' columns for analysis."
echo "Remember: STOP the EC2 instance when done (do NOT terminate)."
