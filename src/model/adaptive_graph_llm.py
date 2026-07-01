import torch  # noqa: F401  (kept for parity / downstream extension)
from torch_scatter import scatter
from src.model.graph_llm import GraphLLM
import src.model.graphgps  # noqa: F401  (registers 'graphgps' in load_gnn_model on import)


class AdaptiveGraphLLM(GraphLLM):
    """Unified backbone for the thesis contributions.

    Week 3 (Contribution 1): swap the subgraph encoder to GraphGPS. GraphGPS
    needs two inputs the base encoders don't — the precomputed RWSE (``graph.pe``)
    and the ``batch`` vector (to scope global attention per subgraph) — so the
    only method that changes is ``encode_graphs``. Everything else (LLM loading,
    LoRA/QLoRA, projector, ``forward``, ``inference``) is inherited unchanged,
    which keeps this a clean single-variable step over config #2 (attention + gt).

    Select with ``--model_name adaptive_graph_llm --gnn_model_name graphgps``.
    (Weeks 4-5 will extend this class with the confidence head + iterative
    refinement; those are gated off here.)
    """

    def encode_graphs(self, samples):
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(
            graphs.x,
            graphs.edge_index.long(),
            graphs.edge_attr,
            graphs.pe,
            graphs.batch,
        )
        # mean pooling (identical to GraphLLM)
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        return g_embeds
