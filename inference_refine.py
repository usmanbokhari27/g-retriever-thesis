"""Test-time evaluation of the config #4 checkpoint, with or without Week-5
iterative refinement (Contribution 3).

Unlike the stock ``inference.py`` (which never loads a checkpoint), this loads the
trained config #4 weights — LoRA adapter + GraphGPS encoder + projector +
confidence head — and runs the test set through ``AdaptiveGraphLLM.inference``.
With ``--refine`` the subgraph is expanded on low-confidence samples before
generation; without it, this simply re-evaluates config #4 (a sanity check that
should reproduce Hit@1 72.54).

No training happens here, so a full pass is only the ~1 h eval, not the ~24 h run.

Usage on EC2 (from the repo root, conda env active):
    # config #5 (refinement ON):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inference_refine.py \
      --dataset webqsp_attn --model_name adaptive_graph_llm \
      --gnn_model_name graphgps --confidence_head --refine \
      --llm_model_name 8b --llm_frozen False --eval_batch_size 4

    # sanity re-eval (refinement OFF) — should reproduce config #4 (72.54):
    (drop --refine)

The output CSV name encodes the refine flag so config #4's CSV is never clobbered.
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


def main(args):
    seed = args.seed
    tag = 'refine' if getattr(args, 'refine', False) else 'norefine'
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_{tag}_seed{seed}",
               config=args)

    seed_everything(seed=seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    test_dataset = [dataset[i] for i in idx_split['test']]
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False,
                             collate_fn=collate_fn)

    # Build the model, then load the trained config #4 weights into it.
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args,
                                        init_prompt=dataset.prompt)
    if getattr(args, 'checkpoint_path', ''):
        model = _reload_model(model, args.checkpoint_path)
    else:
        model = _reload_best_model(model, args)

    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = (f'{args.output_dir}/{args.dataset}/'
            f'model_name_{args.model_name}_llm_model_name_{args.llm_model_name}'
            f'_gnn_model_name_{args.gnn_model_name}_seed{seed}_{tag}.csv')
    print(f'path: {path}')

    # Optional smoke cap: SMOKE_BATCHES=2 evaluates only the first N batches so a
    # GPU dry-run can verify checkpoint load + refinement before the full pass.
    smoke = int(os.environ.get('SMOKE_BATCHES', '0'))

    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            if smoke and step >= smoke:
                break
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)
    if smoke:
        print(f'SMOKE: evaluated {smoke} batch(es); skipping full metrics.')
        return

    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    wandb.log({'Test Acc': acc})

    # Refinement diagnostics: how often expansion fired, and how deep.
    if getattr(args, 'refine', False):
        iters = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if 'num_refine_iters' in rec:
                    iters.append(rec['num_refine_iters'])
        if iters:
            fired = sum(1 for x in iters if x > 0)
            print(f'Refinement fired on {fired}/{len(iters)} '
                  f'({100 * fired / len(iters):.1f}%); '
                  f'mean iters {sum(iters) / len(iters):.3f}, max {max(iters)}')


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
