# Adaptive Graph RAG with Structure-Aware Encoding, Confidence Estimation, and Iterative Refinement

MSc dissertation project — **Syed Usman Ali Shah** (`ec25079@qmul.ac.uk`)
School of Electronic Engineering and Computer Science, Queen Mary University of London, August 2026

This repository extends **G-Retriever** (He et al., NeurIPS 2024) with three components that make its retrieval adaptive and aware of its own reliability. It is a fork of [`XiaoxinHe/G-Retriever`](https://github.com/XiaoxinHe/G-Retriever); the upstream commit history is preserved beneath this project's commits, so `git log` shows exactly which code is inherited and which is mine.

---

## Contributions

| | Component | Selected by |
|---|---|---|
| — | Attention-based retrieval — *adopted from [Efficient-G-Retriever](https://arxiv.org/abs/2504.14955), cited, not claimed* | `--dataset webqsp_attn` |
| **1** | **GraphGPS encoder** with random-walk structural encodings | `--gnn_model_name graphgps` |
| **2** | **Confidence head** — MLP estimating whether the retrieved subgraph contains the answer | `--confidence_head` |
| **3** | **Iterative refinement** — expands the subgraph one hop when confidence < τ, at test time | `--refine` |

Everything runs on **Llama-3.1-8B with 4-bit QLoRA on a single NVIDIA T4 (16 GB)**, evaluated on **WebQSP**.

## Results

WebQSP test set, single seed, scored after end-of-sequence truncation (see [Evaluation note](#evaluation-note-important)):

| # | Configuration | Hit@1 | Acc. | Prec. | F1 |
|---|---|---|---|---|---|
| 1 | PCST + graph transformer *(reproduced baseline)* | 61.61 | 41.85 | 54.87 | 42.16 |
| 2 | Attention + graph transformer | 72.17 | 52.25 | 68.43 | 54.02 |
| 3 | Attention + GraphGPS *(Contribution 1)* | 71.99 | 52.51 | 67.12 | 53.25 |
| 4 | + confidence head *(Contribution 2)* | 72.54 | 52.55 | 67.64 | **54.11** |
| 5 | + refinement *(Contribution 3)* | **72.85** | 52.95 | **69.05** | 54.09 |

Refinement is selective: it fires on **153 / 1628 questions (9.4 %)**, and on that subset improves Hit@1 by **5.23** points (62.75 → 67.97) while leaving the other 90.6 % within 0.6 points. Recall is omitted above because the evaluation script's accuracy coincides with it exactly.

Each row changes **exactly one component** from the row above, so any metric change is attributable to that component. Row 5 is a test-time procedure applied to the row-4 checkpoint with no retraining.

---

## Repository layout

Files marked ★ are this project's; everything else is upstream G-Retriever.

```
src/
├── dataset/
│   ├── webqsp.py                     PCST retrieval cache (upstream)
│   ├── webqsp_attn.py              ★ attention-retrieval dataset variant
│   └── utils/
│       ├── retrieval.py              PCST retriever (upstream)
│       ├── attention_retrieval.py  ★ attention retriever (adopted, cited)
│       └── subgraph_refine.py      ★ 1-hop expansion + subgraph rebuild
├── model/
│   ├── graph_llm.py                  baseline; QLoRA + device patch applied
│   ├── graphgps.py                 ★ GraphGPS encoder (Contribution 1)
│   └── adaptive_graph_llm.py       ★ GraphLLM subclass: encoder hook,
│                                     confidence head, refinement loop
└── config.py                         + flags added by the patch scripts

add_rwse_attn.py                    ★ one-time offline RWSE precomputation
inference_refine.py                 ★ test-set eval with/without refinement
inference_sweep.py                  ★ validation sweep over τ and K
truncate_eval.py                    ★ offline EOS-truncation re-scoring
patch_config_flags.py               ★ adds --confidence_head/--confidence_weight
patch_refine_flags.py               ★ adds --refine/--refine_tau/...
scripts/                            ★ one runnable script per configuration
analysis/                           ★ offline analysis — no GPU needed
results/                            ★ predictions, logs and figures behind the paper
```

**Design rule.** The original G-Retriever files are not modified. Every contribution is a new file, a subclass, or a command-line flag registered through the codebase's existing lookup tables, so each configuration differs from the next by one switch. There is [one documented exception](#the-one-modified-upstream-file).

---

## Setup

```bash
git clone https://github.com/usmanbokhari27/g-retriever-thesis.git
cd g-retriever-thesis

conda create -n gretriever python=3.10 -y
conda activate gretriever
pip install -r requirements.txt
pip install bitsandbytes accelerate        # required for 4-bit QLoRA

python patch_config_flags.py               # idempotent; backs up src/config.py
python patch_refine_flags.py

huggingface-cli login                      # Llama-3.1-8B weights are gated
wandb login                                # optional
```

**PyTorch ≥ 2.6 note.** `torch.load` now defaults to `weights_only=True`, which cannot unpickle the PyTorch Geometric `Data` objects this codebase saves (`UnpicklingError: Unsupported global ... DataEdgeAttr`). Add a `sitecustomize.py` to the environment's `site-packages`:

```python
import torch
_orig = torch.load
def _load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _load
```

This changes the environment, not the repository; deleting the file reverts it. It is safe here because every file loaded is self-generated.

## Data preparation

Preprocessing runs in **two stages** — the second is not called by the first, and training fails with `FileNotFoundError: cached_graphs/0.pt` if it is skipped.

```bash
bash scripts/run_preprocess.sh     # 1. Sentence-BERT embedding      GPU, ~10 h
python -m src.dataset.webqsp       # 2. PCST retrieval cache         CPU, ~30 min
python -m src.dataset.webqsp_attn  # 2b. attention cache (configs 2-5) CPU, ~5 min
python add_rwse_attn.py            # 3. RWSE encodings for GraphGPS  ~16 s
```

Expected end state: **4,699** graphs in `dataset/webqsp/graphs/`, `q_embs.pt`, and `dataset/webqsp/split/` populated 2,826 / 245 / 1,628. The count is 4,699 not 4,700 — index 2937 is an empty graph and is skipped by design.

> **Disk.** Preprocessing produces ~100 GB. A 100 GB volume is *not* enough and will fail with `ENOSPC` partway through; use 200 GB. If stage 1 is interrupted, `resume_webqsp.py` restarts it without recomputing completed graphs and `monitor_finalize.sh` runs the split generation afterwards. Delete the highest-index `.pt` before resuming — it may have been truncated mid-write.

## Running the experiments

Run in order; each script checks its own preconditions before launching.

```bash
bash scripts/run_baseline.sh     # config 1: PCST + graph transformer (the anchor)
bash scripts/run_attention.sh    # config 2: + attention retrieval
bash scripts/run_graphgps.sh     # config 3: + GraphGPS            (Contribution 1)
bash scripts/run_confidence.sh   # config 4: + confidence head     (Contribution 2)
bash scripts/run_refine.sh       # config 5: + refinement          (Contribution 3)
bash scripts/run_sweep.sh        # validation sweep over τ and K
```

Use `tmux` for anything long-running (`tmux new -s train`, detach with `Ctrl-B` `D`).

Configurations 1–3 must be re-scored before their numbers are comparable:

```bash
python truncate_eval.py output/webqsp/<predictions>.csv
```

Configurations 4–5 have truncation built into inference.

> ⚠️ Output filenames encode the model and encoder but **not** `--confidence_head`, so running config 4 silently overwrites config 3's predictions and checkpoint. Back up `output/` between runs.

**Approximate cost on a T4:** preprocessing ~10 h, each training run 14–20 h, refinement eval ~5 h, sweep ~3 h. Full reproduction is roughly 80 GPU-hours.

## Verifying the results without a GPU

Every number in the paper can be re-derived from the committed predictions in about a minute. Requires only `pandas numpy matplotlib scipy`.

```bash
cd analysis
python make_figures.py        # regenerates both paper figures + calibration table
python analyze_fired.py       # fired vs not-fired breakdown
python significance_test.py   # McNemar exact test on the fired-subset flips
python aggregate_sweep.py     # aggregates the validation sweep
```

`make_figures.py` prints exactly the values quoted in the paper:

```
n=1628  positive rate=0.6677  AUC=0.6622  ECE=0.0646
mean conf | positive=0.7064  negative=0.6326
```

## Why there is no executable

The system fine-tunes an 8-billion-parameter model and needs a CUDA GPU with ≥16 GB VRAM; the preprocessed WebQSP data is ~100 GB and cannot be distributed; and the Llama-3.1-8B weights are gated and cannot be redistributed. The reproduction steps above are complete, and the no-GPU verification path covers every reported number.

---

## Notes

### Evaluation note (important)

Upstream G-Retriever runs on Llama-2, where `</s>` is the end-of-sequence token, so generation halts there. Under Llama-3.1, `eos_token_id` is `None` and `</s>` is ordinary text — generation runs the full 32 tokens and emits degenerate repetition after the real answer, which the evaluation script (splitting on delimiters) scores as extra predicted entities. **89–91 % of predictions were affected.** Truncating at the first end-of-sequence marker reproduces what correct EOS-terminated generation would have produced, and **lowers** every headline number by 6–8 Hit@1 points. All numbers in this repository and in the paper are the truncated ones.

### The one modified upstream file

`src/model/graph_llm.py` was edited directly, with supervisor approval, because it *is* the reproduced baseline rather than a layered contribution: loading the backbone in 4-bit NF4 via `BitsAndBytesConfig` (plain LoRA does not fit a T4), and adding `.to(self.model.device)` on the BOS and pad-token tensors in `forward` and `inference` to fix a device-placement failure that appears only under 4-bit quantisation, where bitsandbytes bypasses accelerate's device-move hooks for the embedding lookup.

### Retrieval flag naming

Attention retrieval is selected by `--dataset webqsp_attn` rather than a `--retriever` flag, because `train.py` instantiates datasets by name with no constructor arguments (`load_dataset[args.dataset]()`) and `eval_funcs` is keyed on the same name. A dataset subclass was far less invasive, and it keeps output paths distinct between runs.

### Limitations

All results except the validation sweep come from a **single random seed**; the sweep shows decoding nondeterminism alone is worth about ±1.5 Hit@1 points on the 245-question validation split. Evaluation is on **WebQSP only**. Both are discussed in the dissertation.

---

## Attribution and licence

| Component | Source | Status |
|---|---|---|
| Base pipeline, training loop, PCST retrieval, metrics | [G-Retriever](https://github.com/XiaoxinHe/G-Retriever) (He et al., 2024), MIT | Unmodified except as noted above |
| Attention-based retrieval | [Efficient-G-Retriever](https://arxiv.org/abs/2504.14955) (Solanki, 2025) | Re-implemented from the paper; cited, not claimed |
| `GPSConv`, `GINEConv`, `AddRandomWalkPE` | PyTorch Geometric | Library components |
| Llama-3.1-8B | Meta | Gated weights, used under Meta's licence |
| Sentence-BERT `all-roberta-large-v1` | Reimers & Gurevych (2019) | Frozen, inference only |

`src/model/graphgps.py`, the confidence-head and refinement paths in `src/model/adaptive_graph_llm.py`, `src/dataset/utils/subgraph_refine.py`, and all of `analysis/` are my own work. Upstream's MIT `LICENSE` is retained.

Use of generative AI tools during this project is declared in Appendix D of the dissertation.
