#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Attention-retrieval run (ablation config #2) — G-Retriever, WebQSP
# Backbone: LLaMA 3.1 8B + QLoRA (4-bit NF4) on a T4 (16 GB)
# Same model/optim as the baseline; ONLY the retrieval changes (EGR mod #1,
# attention top-k) via --dataset webqsp_attn. Compare its Hit@1 against the
# 68.98% PCST baseline anchor.
#
# Usage on EC2:
#   tmux new -s train
#   bash ~/run_attention.sh
# (the script self-activates conda — no need to activate or cd first)
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever

# Guard: the attention cache must be built first (CPU pass):
#   PYTHONPATH=. python -m src.dataset.webqsp_attn
n=$(ls dataset/webqsp/cached_graphs_attn/ 2>/dev/null | wc -l)
if [ "$n" -lt 4699 ]; then
  echo "ERROR: attention cache not ready — found $n graphs in cached_graphs_attn/ (need 4699)."
  echo "Build it first:  PYTHONPATH=. python -m src.dataset.webqsp_attn"
  exit 1
fi
echo "Attention cache OK ($n cached graphs)."

# Avoid VRAM fragmentation OOM on the T4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size 2 * grad_steps 8 = effective batch 16 (defaults 8/16 OOM an 8B model on a T4).
# No CPU smoke test is possible — bitsandbytes 4-bit weights require CUDA.
# Watch the first ~10 steps for OOM / tokenizer errors before walking away.
python train.py \
  --dataset webqsp_attn \
  --model_name graph_llm \
  --llm_model_name 8b \
  --llm_frozen False \
  --batch_size 2 \
  --grad_steps 8 \
  --eval_batch_size 4 \
  2>&1 | tee ~/attention_webqsp.log

echo "=== ATTENTION RUN COMPLETE ==="
echo "Results logged to WandB and ~/attention_webqsp.log"
echo "Compare the final Test Hit@1 against the 68.98% PCST baseline."
echo "Remember: STOP the EC2 instance now (do NOT terminate)."
