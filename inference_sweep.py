"""Week-6 validation sweep of refinement hyperparameters (tau, K).

Tunes the Contribution-3 refinement loop on the WebQSP VALIDATION split so the
final tau / K are chosen without ever touching the test set. The config #4
checkpoint (LoRA + GraphGPS + projector + confidence head) is loaded ONCE, and
because AdaptiveGraphLLM reads ``refine_tau`` / ``refine_max_iters`` as live
instance attributes, each (tau, K) combo is evaluated by mutating those
attributes and re-running only the generation loop -- no model reload, no
retraining. One CSV is written per combo; authoritative Hit@1/F1 come from
``aggregate_sweep.py`` run over those CSVs.

Companion to ``inference_refine.py`` (same machinery); leaves it and all
original files untouched, per the project's minimal-surgery rule.

Usage on EC2 (repo root, conda env active):
    # GPU smoke first (1 combo, 1 batch) -- verify ckpt load + val split + CSV:
    SMOKE_BATCHES=1 python inference_sweep.py \
      --dataset webqsp_attn --model_name adaptive_graph_llm \
      --gnn_model_name graphgps --confidence_head --refine \
      --llm_model_name 8b --llm_frozen False --eval_batch_size 4 \
      --checkpoint_path <path-to-config4-ckpt>.pth

    # full sweep (drop SMOKE_BATCHES; run in tmux):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inference_sweep.py \
      --dataset webqsp_attn --model_name adaptive_graph_llm \
      --gnn_model_name graphgps --confidence_head --refine \
      --llm_model_name 8b --llm_frozen False --eval_batch_size 4 \
      --checkpoint_path <path-to-config4-ckpt>.pth

Env knobs:
    SWEEP_SPLIT   which split to evaluate (default 'val'; never use 'test' to tune)
    SMOKE_BATCHES if >0, run only the first combo for N batches and skip metrics
"""
import os
import gc
import json

import torch
import wandb
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
from src.utils.ckpt import _reload_best_model, _reload_model

# --- Sweep grid -------------------------------------------------------------
# A cross design centred on the current default (tau=0.5, K=2): sweep tau at
# fixed K=2, then vary K at tau=0.5. Plus a refinement-off anchor on the same
# split. Edit this list to widen/narrow the sweep. Each combo generates over
# the whole validation split (~245 samples), so keep it lean.
COMBOS = [
    {"tag": "off",           "refine": False, "tau": 0.0, "K": 0},
    {"tag": "tau0.4_K2",     "refine": True,  "tau": 0.4, "K": 2},
    {"tag": "tau0.5_K2",     "refine": True,  "tau": 0.5, "K": 2},
    {"tag": "tau0.6_K2",     "refine": True,  "tau": 0.6, "K": 2},
    {"tag": "tau0.7_K2",     "refine": True,  "tau": 0.7, "K": 2},
    {"tag": "tau0.5_K1",     "refine": True,  "tau": 0.5, "K": 1},
    {"tag": "tau0.5_K3",     "refine": True,  "tau": 0.5, "K": 3},
]


def eval_combo(model, loader, path, smoke):
    """Run the generation loop over `loader`, writing one JSON line per sample."""
    model.eval()
    bar = tqdm(range(len(loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(loader):
            if smoke and step >= smoke:
                break
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            bar.update(1)


def fire_stats(path):
    iters = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "num_refine_iters" in rec:
                iters.append(rec["num_refine_iters"])
    if not iters:
        return None
    fired = sum(1 for x in iters if x > 0)
    return dict(n=len(iters), fired=fired,
                pct=100 * fired / len(iters),
                mean=sum(iters) / len(iters), max=max(iters))


def main(args):
    seed = args.seed
    split = os.environ.get("SWEEP_SPLIT", "val")
    smoke = int(os.environ.get("SMOKE_BATCHES", "0"))

    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_sweep_{split}_seed{seed}",
               config=args)
    seed_everything(seed=seed)
    print(args)
    print(f"[sweep] split={split}  smoke={smoke}  combos={len(COMBOS)}")

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    if split not in idx_split:
        raise KeyError(f"split '{split}' not in idx_split keys {list(idx_split)}")
    eval_dataset = [dataset[i] for i in idx_split[split]]
    print(f"[sweep] {split} set: {len(eval_dataset)} samples")
    loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                        drop_last=False, pin_memory=True, shuffle=False,
                        collate_fn=collate_fn)

    # Build model + load config #4 weights ONCE.
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args,
                                        init_prompt=dataset.prompt)
    if getattr(args, "checkpoint_path", ""):
        model = _reload_model(model, args.checkpoint_path)
    else:
        model = _reload_best_model(model, args)

    out_dir = f"{args.output_dir}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    combos = COMBOS[:1] if smoke else COMBOS
    summary = []
    for combo in combos:
        # Mutate the live model -- no reload.
        model.refine = combo["refine"]
        model.refine_tau = float(combo["tau"])
        model.refine_max_iters = int(combo["K"])
        path = (f"{out_dir}/model_name_{args.model_name}"
                f"_llm_model_name_{args.llm_model_name}"
                f"_gnn_model_name_{args.gnn_model_name}"
                f"_seed{seed}_{split}_{combo['tag']}.csv")
        print(f"\n[sweep] === {combo['tag']}  refine={combo['refine']} "
              f"tau={combo['tau']} K={combo['K']} ===\n  -> {path}")
        eval_combo(model, loader, path, smoke)

        if smoke:
            print(f"[sweep] SMOKE: wrote {smoke} batch(es) for {combo['tag']}; "
                  f"skipping metrics.")
            continue

        hit = eval_funcs[args.dataset](path)  # prints Acc/Hit/Prec/Rec/F1
        fs = fire_stats(path) if combo["refine"] else None
        row = dict(tag=combo["tag"], tau=combo["tau"], K=combo["K"], hit=hit)
        if fs:
            row.update(fired=fs["fired"], n=fs["n"], fired_pct=round(fs["pct"], 2),
                       mean_iters=round(fs["mean"], 3), max_iters=fs["max"])
            print(f"[sweep] {combo['tag']}: Hit {hit:.4f}  fired "
                  f"{fs['fired']}/{fs['n']} ({fs['pct']:.1f}%)  "
                  f"mean iters {fs['mean']:.3f}")
        else:
            print(f"[sweep] {combo['tag']}: Hit {hit:.4f}")
        summary.append(row)
        wandb.log({f"hit/{combo['tag']}": hit})

    if not smoke:
        print("\n[sweep] ===== SUMMARY (val) =====")
        for r in summary:
            print("  " + json.dumps(r))
        summ_path = f"{out_dir}/sweep_summary_{split}_seed{seed}.json"
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sweep] wrote {summ_path}")


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
