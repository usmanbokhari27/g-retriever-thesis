#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Confidence-head run (ablation config #4) — G-Retriever, WebQSP
# Backbone: LLaMA 3.1 8B + QLoRA (4-bit NF4) on a T4 (16 GB)
# Same model/optim/retrieval/encoder as config #3 (attention + GraphGPS);
# ONLY adds Contribution 2: a confidence head trained with a BCE auxiliary loss
# (subgraph-contains-gold-answer) via --confidence_head.
# Compare its Hit@1 against config #3 (79.79), config #2 (78.32) and the 68.98
# PCST anchor. Precision/F1 should also read cleaner here — this class truncates
# predictions at the EOS marker (the Session-10 post-</s> cleanup).
#
# Self-contained: verifies the Week-4 model file is deployed, auto-adds the two
# argparse flags to src/config.py (idempotent, backed up), then launches.
#
# Usage on EC2:
#   tmux new -s train
#   bash ~/run_confidence.sh
# (the script self-activates conda — no need to activate or cd first)
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gretriever
cd ~/G-Retriever

# --- Guard 0: the Week-4 model file must actually be deployed -----------------
# Without this, --confidence_head is silently ignored (getattr default False) and
# you'd unknowingly re-run config #3 instead of #4.
if ! grep -q 'confidence_head' src/model/adaptive_graph_llm.py; then
  echo "ERROR: src/model/adaptive_graph_llm.py has no confidence head."
  echo "Deploy the Week-4 version first (scp week4_port/adaptive_graph_llm.py"
  echo "  -> ~/G-Retriever/src/model/adaptive_graph_llm.py)."
  exit 1
fi

# --- Guard 1: attention cache must be built ----------------------------------
#   PYTHONPATH=. python -m src.dataset.webqsp_attn
n=$(ls dataset/webqsp/cached_graphs_attn/ 2>/dev/null | wc -l)
if [ "$n" -lt 4699 ]; then
  echo "ERROR: attention cache not ready — found $n graphs in cached_graphs_attn/ (need 4699)."
  echo "Build it first:  PYTHONPATH=. python -m src.dataset.webqsp_attn"
  exit 1
fi

# --- Guard 2: RWSE (pe) must be present (GraphGPS encoder needs it) -----------
#   PYTHONPATH=. python -m add_rwse_attn
has_pe=$(PYTHONPATH=. python -c "import torch,glob; g=torch.load(sorted(glob.glob('dataset/webqsp/cached_graphs_attn/*.pt'))[0], weights_only=False); print(1 if (getattr(g,'pe',None) is not None and g.pe.shape[1]==20) else 0)")
if [ "$has_pe" != "1" ]; then
  echo "ERROR: cached attention graphs have no RWSE (pe). GraphGPS needs it."
  echo "Add it first:  PYTHONPATH=. python -m add_rwse_attn"
  exit 1
fi
echo "Attention cache OK ($n cached graphs, RWSE present)."

# --- Guard 3: ensure the two Week-4 argparse flags exist in src/config.py -----
# Adding argparse flags is the sanctioned extension pattern (handover §4.5).
# Idempotent + backed up; inserts right before parser.parse_args(), matching indent.
python - <<'PY'
p = 'src/config.py'
src = open(p).read()
if '--confidence_head' in src:
    print("config.py: Week-4 flags already present.")
else:
    import shutil
    shutil.copyfile(p, p + '.bak_confidence')
    out, inserted = [], False
    for ln in src.splitlines(keepends=True):
        if not inserted and 'parser.parse_args(' in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(f'{indent}parser.add_argument("--confidence_head", action="store_true")\n')
            out.append(f'{indent}parser.add_argument("--confidence_weight", type=float, default=0.1)\n')
            inserted = True
        out.append(ln)
    if not inserted:
        raise SystemExit("ERROR: could not find parser.parse_args() in config.py — "
                         "add the two flags manually (see week4_port/CONFIG_ADDITIONS.md).")
    open(p, 'w').write(''.join(out))
    print("config.py: Week-4 flags inserted (backup at src/config.py.bak_confidence).")
PY

# Verify argparse actually recognises them before spending GPU time.
if ! python train.py --help 2>/dev/null | grep -q -- '--confidence_head'; then
  echo "ERROR: --confidence_head not recognised by train.py after patching config.py."
  echo "Inspect src/config.py and add the flags manually (week4_port/CONFIG_ADDITIONS.md)."
  exit 1
fi
echo "config flags OK (--confidence_head recognised)."

# Avoid VRAM fragmentation OOM on the T4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size 2 * grad_steps 8 = effective batch 16 (defaults 8/16 OOM an 8B model on a T4).
# --confidence_head adds a tiny MLP (~0.5M params) + a BCE term to the loss; negligible
# extra VRAM/time over config #3. Watch the first ~10 steps for OOM/tokenizer/shape errors.
python train.py \
  --dataset webqsp_attn \
  --model_name adaptive_graph_llm \
  --gnn_model_name graphgps \
  --confidence_head \
  --confidence_weight 0.1 \
  --llm_model_name 8b \
  --llm_frozen False \
  --batch_size 2 \
  --grad_steps 8 \
  --eval_batch_size 4 \
  2>&1 | tee ~/confidence_webqsp.log

echo "=== CONFIDENCE RUN COMPLETE ==="
echo "Results logged to WandB and ~/confidence_webqsp.log"
echo "Compare Test Hit@1 vs config #3 (79.79), #2 (78.32), anchor (68.98)."
echo "The saved prediction CSV now also carries a 'confidence' column per sample"
echo "(score the head's AUC/calibration offline; Week-5 refinement consumes it)."
echo "Remember: STOP the EC2 instance now (do NOT terminate)."
