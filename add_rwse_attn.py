"""One-time offline pass: add RWSE (random-walk structural encodings) as a
``pe`` attribute to the cached attention subgraphs, for the GraphGPS encoder
(Contribution 1). RWSE MUST be precomputed offline and cached — never computed
inside the training loop.

- Pure CPU, fast: attention subgraphs are tiny (~11 nodes / 5 edges), so
  AddRandomWalkPE(walk_length=20) over all 4699 graphs takes ~15 s.
- Non-destructive: only adds ``data.pe`` of shape [num_nodes, 20]; the existing
  x / edge_index / edge_attr are untouched. The gt/gcn/gat encoders ignore ``pe``,
  so config #2 (attention + gt) is unaffected by this pass.
- Idempotent: skips graphs that already carry a correctly-shaped ``pe``.

Run once from the repo root before launching --gnn_model_name graphgps:
    PYTHONPATH=. python -m add_rwse_attn
"""
import glob
import os
import torch
from tqdm import tqdm
from torch_geometric.transforms import AddRandomWalkPE

CACHE_DIR = 'dataset/webqsp/cached_graphs_attn'
WALK_LENGTH = 20
ATTR = 'pe'


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.pt')))
    if not files:
        raise SystemExit(f'No cached graphs found in {CACHE_DIR}/ — build the '
                         f'attention cache first (python -m src.dataset.webqsp_attn).')

    transform = AddRandomWalkPE(walk_length=WALK_LENGTH, attr_name=ATTR)
    added = skipped = 0
    for f in tqdm(files, desc='RWSE'):
        g = torch.load(f, weights_only=False)
        pe = getattr(g, ATTR, None)
        if pe is not None and tuple(pe.shape) == (g.num_nodes, WALK_LENGTH):
            skipped += 1
            continue
        g = transform(g)
        torch.save(g, f)
        added += 1

    print(f'done: {added} augmented with pe, {skipped} already had pe, '
          f'total {len(files)} (walk_length={WALK_LENGTH}).')


if __name__ == '__main__':
    main()
