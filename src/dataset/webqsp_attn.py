import os
import torch
import pandas as pd
import datasets
from tqdm import tqdm

from src.dataset.webqsp import WebQSPDataset, path, path_nodes, path_edges, path_graphs
from src.dataset.utils.attention_retrieval import retrieval_via_attention

# Separate cache dirs from the PCST baseline so the two retrievers never collide.
cached_graph = f'{path}/cached_graphs_attn'
cached_desc = f'{path}/cached_desc_attn'


class WebQSPAttnDataset(WebQSPDataset):
    """WebQSP with attention-based retrieval (EGR mod #1).

    Identical to WebQSPDataset (same questions, labels, split, q_embs) except it
    reads the attention-retrieval cache (cached_graphs_attn / cached_desc_attn)
    instead of the PCST cache. The single variable changed vs the 68.98% PCST
    baseline is the retrieval method.
    """

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }


def preprocess():
    """Build the attention-retrieval cache. Pure CPU (cosine + top-k + pandas),
    no SBERT and no LLM -- reuses the already-built graphs/nodes/edges/q_embs.pt.
    Skips the empty graph at index 2937 (no cached file), so the expected cache
    count is 4699, matching the PCST baseline."""
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_attention(graph, q_emb, nodes, edges, topk=3, topk_e=5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    preprocess()

    dataset = WebQSPAttnDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
