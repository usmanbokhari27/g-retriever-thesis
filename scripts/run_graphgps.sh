#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GraphGPS-encoder run (ablation config #3) — G-Retriever, WebQSP
# Backbone: LLaMA 3.1 8B + QLoRA (4-bit NF4) on a T4 (16 GB)
# Same model/optim/retrieval as config #2 (attention, --dataset webqsp_attn);
# ONLY the graph encoder changes: gt -> GraphGPS (Contribution 1) via
# --model_name adaptive_graph_llm --gnn_model_name graphgps.
# Compare its Hit@1 against config #2 (78.32) and the 68.98 PCST anchor.
#
# Usage on EC2:
#   tmux new -s train
#   bash ~/run_graphgps.sh
# (the script self-activates conda — no need to activate or cd first)
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever

# Guard 1: the attention cache must be built (CPU pass):
#   PYTHONPATH=. python -m src.dataset.webqsp_attn
n=$(ls dataset/webqsp/cached_graphs_attn/ 2>/dev/null | wc -l)
if [ "$n" -lt 4699 ]; then
  echo "ERROR: attention cache not ready — found $n graphs in cached_graphs_attn/ (need 4699)."
  echo "Build it first:  PYTHONPATH=. python -m src.dataset.webqsp_attn"
  exit 1
fi

# Guard 2: RWSE (pe) must have been added to the cache (GraphGPS needs it):
#   PYTHONPATH=. python -m add_rwse_attn
has_pe=$(PYTHONPATH=. python -c "import torch,glob; g=torch.load(sorted(glob.glob('dataset/webqsp/cached_graphs_attn/*.pt'))[0], weights_only=False); print(1 if (getattr(g,'pe',None) is not None and g.pe.shape[1]==20) else 0)")
if [ "$has_pe" != "1" ]; then
  echo "ERROR: cached attention graphs have no RWSE (pe). GraphGPS needs it."
  echo "Add it first:  PYTHONPATH=. python -m add_rwse_attn"
  exit 1
fi
echo "Attention cache OK ($n cached graphs, RWSE present)."

# Avoid VRAM fragmentation OOM on the T4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size 2 * grad_steps 8 = effective batch 16 (defaults 8/16 OOM an 8B model on a T4).
# No CPU smoke test of the full model is possible — bitsandbytes 4-bit weights require CUDA.
# Watch the first ~10 steps for OOM / tokenizer / shape errors before walking away.
python train.py \
  --dataset webqsp_attn \
  --model_name adaptive_graph_llm \
  --gnn_model_name graphgps \
  --llm_model_name 8b \
  --llm_frozen False \
  --batch_size 2 \
  --grad_steps 8 \
  --eval_batch_size 4 \
  2>&1 | tee ~/graphgps_webqsp.log

echo "=== GRAPHGPS RUN COMPLETE ==="
echo "Results logged to WandB and ~/graphgps_webqsp.log"
echo "Compare the final Test Hit@1 against config #2 (78.32) and the 68.98 anchor."
echo "Remember: STOP the EC2 instance now (do NOT terminate)."
