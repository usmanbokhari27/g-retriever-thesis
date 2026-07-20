"""Iterative-refinement subgraph operations (Contribution 3, Week 5).

Pure-CPU graph manipulation used by ``AdaptiveGraphLLM`` at inference time to
grow a retrieved subgraph when the confidence head is not yet confident. Kept in
its own module (no model / no LLM) so it is unit-testable on CPU and so it does
not touch any original G-Retriever file.

Design (matches the Week-5 plan):
  * Level 0 reproduces the *exact* attention-retrieval subgraph the model was
    trained on (same top-k node / top-k_e edge selection, same reindexing, same
    RWSE), so refinement with 0 iterations == config #4 byte-for-byte.
  * Each refinement step expands one hop *along edges* from the current node set
    into the full per-question KG (``graphs/<i>.pt``), adding the incident edges
    and their far endpoints. Growth is capped at ``max_nodes`` by ranking new
    candidate nodes by cosine similarity to the question embedding — the SAME
    relevance signal attention retrieval and PCST use — so the expansion stays
    focused and the textual description / VRAM stay bounded.
  * RWSE (``pe``) is recomputed on every produced subgraph exactly as the Week-3
    offline pass did (``AddRandomWalkPE(walk_length=20, attr_name='pe')``), so the
    GraphGPS encoder receives the positional encodings it expects.

The full graphs / textual node & edge tables and the question embeddings are the
same artifacts attention retrieval was built from, so nothing new is preprocessed.
"""
import functools

import torch
import pandas as pd
from torch_geometric.data.data import Data
from torch_geometric.transforms import AddRandomWalkPE

from src.dataset.webqsp import path, path_nodes, path_edges, path_graphs

WALK_LENGTH = 20
_pe_transform = AddRandomWalkPE(walk_length=WALK_LENGTH, attr_name='pe')
_cos = torch.nn.CosineSimilarity(dim=-1)


# --------------------------------------------------------------------------- IO
@functools.lru_cache(maxsize=16)
def _load_full(index):
    """Load (and LRU-cache) the full KG + textual tables for one question.

    Cached so that the K expansion iterations of a single sample re-use one load
    instead of re-reading the ~20 MB graph each step.
    """
    graph = torch.load(f'{path_graphs}/{index}.pt', weights_only=False)
    nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
    edges = pd.read_csv(f'{path_edges}/{index}.csv')
    return graph, nodes, edges


@functools.lru_cache(maxsize=1)
def _q_embs():
    return torch.load(f'{path}/q_embs.pt')


# ------------------------------------------------------------------ selection
def attention_node_edge_ids(graph, q_emb, topk=3, topk_e=5):
    """Reproduce the attention-retrieval level-0 selection.

    Returns ``(sorted_node_ids, edge_ids)`` in the ORIGINAL id space of the full
    graph. Mirrors ``retrieval_via_attention`` exactly: top-k_e edges + top-k
    nodes + the nodes incident to the selected edges. ``edge_ids`` keeps the
    top-k_e edges (not the induced set), matching the cached subgraph the model
    was trained on.
    """
    selected_edges = []
    if topk_e > 0 and graph.num_edges > 0:
        e_prizes = _cos(q_emb, graph.edge_attr)
        k_e = min(topk_e, graph.num_edges)
        _, idx = torch.topk(e_prizes, k_e, largest=True)
        selected_edges = idx.tolist()

    selected_nodes = set()
    if topk > 0 and graph.num_nodes > 0:
        n_prizes = _cos(q_emb, graph.x)
        k_n = min(topk, graph.num_nodes)
        _, idx = torch.topk(n_prizes, k_n, largest=True)
        selected_nodes.update(idx.tolist())

    if len(selected_edges) > 0:
        incident = graph.edge_index[:, selected_edges]
        selected_nodes.update(incident[0].tolist())
        selected_nodes.update(incident[1].tolist())

    return sorted(selected_nodes), list(selected_edges)


# ------------------------------------------------------------------- expansion
def expand_one_hop(graph, node_ids, edge_ids, q_emb, max_nodes):
    """Grow the subgraph one hop along edges from the current node set.

    Adds every full-graph edge incident to a current node, plus the far
    endpoints of those edges. If that would exceed ``max_nodes``, the new
    candidate nodes are ranked by cosine similarity to the question and only the
    top ones (and the edges reaching them) are kept.

    Returns ``(new_node_ids_sorted, new_edge_ids, num_added_nodes)``. When
    ``num_added_nodes == 0`` the expansion is exhausted (no reachable new nodes,
    or the cap is already saturated) and the caller should stop.
    """
    node_set = set(node_ids)
    edge_set = set(edge_ids)
    if graph.num_edges == 0:
        return sorted(node_set), list(edge_set), 0

    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    node_tensor = torch.tensor(sorted(node_set))
    # Edges with at least one endpoint already inside the subgraph.
    incident_mask = torch.isin(src, node_tensor) | torch.isin(dst, node_tensor)
    incident_edges = incident_mask.nonzero(as_tuple=True)[0].tolist()

    # Candidate new nodes = far endpoints not already present.
    candidates = {}
    for j in incident_edges:
        for endpoint in (int(src[j]), int(dst[j])):
            if endpoint not in node_set:
                candidates.setdefault(endpoint, j)
    if not candidates:
        return sorted(node_set), list(edge_set), 0

    budget = max_nodes - len(node_set)
    if budget <= 0:
        return sorted(node_set), list(edge_set), 0

    cand_nodes = list(candidates.keys())
    if len(cand_nodes) > budget:
        # Relevance-rank: keep the candidates most similar to the question.
        scores = _cos(q_emb, graph.x[torch.tensor(cand_nodes)])
        keep_idx = torch.topk(scores, budget, largest=True).indices.tolist()
        cand_nodes = [cand_nodes[i] for i in keep_idx]

    kept = set(cand_nodes)
    new_node_set = node_set | kept
    # Keep every incident edge whose BOTH endpoints are now in the subgraph.
    for j in incident_edges:
        if int(src[j]) in new_node_set and int(dst[j]) in new_node_set:
            edge_set.add(j)

    return sorted(new_node_set), sorted(edge_set), len(kept)


# ------------------------------------------------------------- subgraph build
def build_subgraph(index, node_ids, edge_ids):
    """Build the ``(Data, desc)`` for an explicit node/edge selection.

    Reindexes nodes to ``0..len(node_ids)-1`` and remaps ``edge_index`` exactly
    as ``retrieval_via_attention`` does, so ``desc`` keeps original node ids and
    the tensors line up with the encoder. Adds the RWSE ``pe`` attribute.
    """
    graph, textual_nodes, textual_edges = _load_full(index)
    node_ids = sorted(node_ids)
    mapping = {n: i for i, n in enumerate(node_ids)}

    x = graph.x[node_ids]
    if len(edge_ids) > 0:
        edge_ids = sorted(edge_ids)
        ei = graph.edge_index[:, edge_ids]
        edge_attr = graph.edge_attr[edge_ids]
        src = [mapping[i] for i in ei[0].tolist()]
        dst = [mapping[i] for i in ei[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = graph.edge_attr[:0]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=len(node_ids))
    data = _add_pe(data)

    n = textual_nodes.iloc[node_ids]
    e = textual_edges.iloc[edge_ids] if len(edge_ids) > 0 else textual_edges.iloc[[]]
    desc = n.to_csv(index=False) + '\n' + \
        e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    return data, desc


def _add_pe(data):
    """Add RWSE identically to the Week-3 offline pass; degenerate 0-edge graphs
    (which AddRandomWalkPE cannot handle) get a zero pe of the right shape."""
    if data.num_edges == 0:
        data.pe = torch.zeros((data.num_nodes, WALK_LENGTH))
        return data
    return _pe_transform(data)


# ------------------------------------------------------- top-level convenience
def initial_state(index, topk=3, topk_e=5):
    """Level-0 selection for a question. Returns (node_ids, edge_ids, q_emb)."""
    graph, _, _ = _load_full(index)
    q_emb = _q_embs()[index]
    node_ids, edge_ids = attention_node_edge_ids(graph, q_emb, topk, topk_e)
    return node_ids, edge_ids, q_emb


if __name__ == '__main__':
    # CPU self-test on a real question: level 0 must match the cached attention
    # subgraph, and one expansion must strictly grow (or exhaust) the node set.
    import glob
    idx = 1
    node_ids, edge_ids, q_emb = initial_state(idx)
    data0, desc0 = build_subgraph(idx, node_ids, edge_ids)
    print(f'[level 0] nodes={data0.num_nodes} edges={data0.num_edges} '
          f'pe={tuple(data0.pe.shape)}')

    cached = torch.load(f'{path}/cached_graphs_attn/{idx}.pt', weights_only=False)
    assert data0.num_nodes == cached.num_nodes, (data0.num_nodes, cached.num_nodes)
    assert data0.num_edges == cached.num_edges, (data0.num_edges, cached.num_edges)
    assert torch.allclose(data0.x, cached.x), 'level-0 x mismatch vs cache'
    print('[level 0] MATCHES cached attention subgraph ✓')

    graph, _, _ = _load_full(idx)
    n1, e1, added = expand_one_hop(graph, node_ids, edge_ids, q_emb, max_nodes=60)
    data1, desc1 = build_subgraph(idx, n1, e1)
    print(f'[level 1] nodes={data1.num_nodes} edges={data1.num_edges} '
          f'added={added}')
    assert data1.num_nodes >= data0.num_nodes
    print('CPU self-test passed.')
