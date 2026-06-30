import torch
from torch_geometric.data.data import Data


def retrieval_via_attention(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=5):
    """Attention-based subgraph retrieval (ported from Efficient-G-Retriever,
    Solanki 2025, arXiv:2504.14955).

    Scores nodes and edges by cosine similarity to the question embedding -- the
    SAME relevance signal that PCST uses -- but selects the subgraph by a plain
    top-k of nodes plus top-k_e of edges (and the nodes incident to those edges),
    with NO Steiner-tree / PCST optimisation and no connectivity guarantee. The
    contribution is compute efficiency: no pcst_fast solver, cheaper selection.

    Returns the same ``(Data, desc)`` contract as ``retrieval_via_pcst`` so it is a
    drop-in replacement. The returned ``Data`` uses reindexed node ids; ``desc``
    keeps the original node ids/names (matching the PCST path exactly).
    """
    # Degenerate case (mirrors retrieval_via_pcst): nothing to retrieve over ->
    # return the full graph and the full textual description.
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    cos = torch.nn.CosineSimilarity(dim=-1)

    # --- top-k_e edges by cosine similarity to the question ---
    selected_edges = []
    if topk_e > 0 and graph.num_edges > 0:
        e_prizes = cos(q_emb, graph.edge_attr)
        k_e = min(topk_e, graph.num_edges)
        _, topk_e_indices = torch.topk(e_prizes, k_e, largest=True)
        selected_edges = topk_e_indices.tolist()

    # --- top-k nodes by cosine similarity to the question ---
    selected_nodes = set()
    if topk > 0 and graph.num_nodes > 0:
        n_prizes = cos(q_emb, graph.x)
        k_n = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, k_n, largest=True)
        selected_nodes.update(topk_n_indices.tolist())

    # Every node incident to a selected edge must be kept so the edge stays valid.
    if len(selected_edges) > 0:
        incident = graph.edge_index[:, selected_edges]
        selected_nodes.update(incident[0].tolist())
        selected_nodes.update(incident[1].tolist())

    selected_nodes = sorted(selected_nodes)
    mapping = {n: i for i, n in enumerate(selected_nodes)}

    # --- build the induced subgraph with reindexed node ids (mirror PCST output) ---
    x = graph.x[selected_nodes]
    if len(selected_edges) > 0:
        edge_index_orig = graph.edge_index[:, selected_edges]
        edge_attr = graph.edge_attr[selected_edges]
        src = [mapping[i] for i in edge_index_orig[0].tolist()]
        dst = [mapping[i] for i in edge_index_orig[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = graph.edge_attr[:0]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    # desc keeps ORIGINAL node ids/names (same as retrieval_via_pcst).
    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    return data, desc
