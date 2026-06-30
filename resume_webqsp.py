"""
Resume helper for WebQSP step_two preprocessing.

Standalone script (does NOT modify the original src/dataset/preprocess/webqsp.py).
Replicates step_two() exactly, but:
  - skips q_embs.pt if it already exists
  - skips any graph index whose <index>.pt already exists on disk

This lets us resume after the disk-full crash without recomputing the
~2889 graphs already embedded. Output is byte-for-byte equivalent to the
original step_two for the remaining graphs (same SBERT model, same Data layout),
so baseline comparability is preserved.

Run from the repo root:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m resume_webqsp
or
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python resume_webqsp.py
"""
import os
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def resume_step_two():
    print('Loading dataset...')
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    questions = [i['question'] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # questions: only encode if missing
    q_path = f'{path}/q_embs.pt'
    if os.path.exists(q_path):
        print(f'q_embs.pt already exists, skipping question encoding.')
    else:
        print('Encoding questions...')
        q_embs = text2embedding(model, tokenizer, device, questions)
        torch.save(q_embs, q_path)

    print('Encoding graphs (skipping existing)...')
    os.makedirs(path_graphs, exist_ok=True)

    skipped = 0
    for index in tqdm(range(len(dataset))):
        out_file = f'{path_graphs}/{index}.pt'
        if os.path.exists(out_file):
            skipped += 1
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes['node_attr'] = nodes['node_attr'].fillna("")
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, out_file)

    print(f'Done. Skipped {skipped} already-existing graphs.')


if __name__ == '__main__':
    resume_step_two()
