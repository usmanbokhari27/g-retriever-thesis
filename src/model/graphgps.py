import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv
from src.model.gnn import load_gnn_model


class GraphGPS(torch.nn.Module):
    """GraphGPS encoder (Rampasek et al., 2022, arXiv:2205.12454) — Contribution 1.

    Each layer fuses, in parallel, a *local* message-passing model (GINEConv,
    which consumes the 1024-dim edge features) with *global* self-attention over
    all nodes of the (retrieved) subgraph. Random-walk structural encodings
    (RWSE), precomputed offline as ``data.pe`` (walk_length=20), are injected
    into the node features before the GPS stack so the position-blind attention
    knows each node's structural role.

    Constructor mirrors the other encoders in ``load_gnn_model`` so the parent
    ``GraphLLM.__init__`` can build it unchanged. It differs only in ``forward``,
    which additionally consumes ``pe`` and ``batch`` — both supplied by
    ``AdaptiveGraphLLM.encode_graphs``. Returns ``(node_embeds, edge_attr)`` to
    match the ``n_embeds, _ = self.graph_encoder(...)`` unpacking at the call site.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_heads=4, pe_dim=20):
        super().__init__()
        self.node_lin = nn.Linear(in_channels, hidden_channels)
        self.edge_lin = nn.Linear(in_channels, hidden_channels)
        self.pe_lin = nn.Linear(pe_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            # GINEConv adds the (projected) edge feature to each neighbour before
            # the MLP, so edge_attr must match the node hidden dim (edge_dim=None).
            conv = GPSConv(hidden_channels, GINEConv(mlp), heads=num_heads,
                           dropout=dropout)
            self.convs.append(conv)

        self.out_lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pe, batch):
        # Inject RWSE into the node features (added, not concatenated).
        x = self.node_lin(x) + self.pe_lin(pe)
        e = self.edge_lin(edge_attr)
        for conv in self.convs:
            # `batch` scopes the global attention to each subgraph in the batch.
            x = conv(x, edge_index, batch=batch, edge_attr=e)
        x = self.out_lin(x)
        return x, edge_attr


# Register so GraphLLM.__init__ can build it via load_gnn_model[args.gnn_model_name]
# (keeps the original gnn.py untouched — same extension pattern as webqsp_attn).
load_gnn_model['graphgps'] = GraphGPS
