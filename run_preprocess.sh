#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== preprocessing started $(date) ==="
python -m src.dataset.preprocess.webqsp 2>&1 | tee preprocess_webqsp.log
echo "=== preprocessing finished $(date) with exit ${PIPESTATUS[0]} ==="
