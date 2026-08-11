#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Baseline run (ablation config #1) — G-Retriever, WebQSP
# Backbone: LLaMA 3.1 8B + QLoRA (4-bit NF4) on a T4 (16 GB)
# This is OUR reproduced baseline — the anchor all contributions are measured against.
#
# Usage on EC2:
#   tmux new -s train
#   conda activate gretriever
#   cd ~/G-Retriever
#   bash ~/run_baseline.sh
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever

# Avoid VRAM fragmentation OOM on the T4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size 2 * grad_steps 8 = effective batch 16 (defaults 8/16 OOM an 8B model on a T4).
# No CPU smoke test is possible — bitsandbytes 4-bit weights require CUDA.
# Watch the first ~10 steps for OOM / tokenizer errors before walking away.
python train.py \
  --dataset webqsp \
  --model_name graph_llm \
  --llm_model_name 8b \
  --llm_frozen False \
  --batch_size 2 \
  --grad_steps 8 \
  --eval_batch_size 4 \
  2>&1 | tee ~/baseline_webqsp.log

echo "=== BASELINE RUN COMPLETE ==="
echo "Results logged to WandB and ~/baseline_webqsp.log"
echo "Remember: STOP the EC2 instance now (do NOT terminate)."
