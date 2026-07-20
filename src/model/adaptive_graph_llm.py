import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Batch
from src.model.graph_llm import GraphLLM
import src.model.graphgps  # noqa: F401  (registers 'graphgps' in load_gnn_model on import)
from src.dataset.utils import subgraph_refine as sr


class AdaptiveGraphLLM(GraphLLM):
    """Unified backbone for the thesis contributions.

    Week 3 (Contribution 1 — GraphGPS encoder): swap the subgraph encoder to
    GraphGPS, which needs two inputs the base encoders don't — the precomputed
    RWSE (``graph.pe``) and the ``batch`` vector — so ``encode_graphs`` is
    overridden. Select with ``--gnn_model_name graphgps``.

    Week 4 (Contribution 2 — Confidence head): a lightweight MLP on the pooled
    GraphGPS embedding produces a scalar in [0, 1] estimating whether the
    retrieved subgraph actually contains a gold answer entity. It is trained with
    a BCE auxiliary loss added to the LM loss and is gated behind
    ``--confidence_head`` (off by default, so config #3 is reproduced exactly when
    the flag is absent). The confidence scalar is also emitted at inference time.

    Week 5 (Contribution 3 — Iterative refinement): a *test-time* procedure gated
    behind ``--refine`` that consumes the trained confidence head. Before
    generating, each sample's retrieved subgraph is scored; while the confidence
    is below ``--refine_tau`` (and under ``--refine_max_iters`` steps), the
    subgraph is expanded one hop along edges into the full per-question KG
    (relevance-capped at ``--refine_max_nodes``), re-encoded and re-scored. The
    final, possibly-expanded subgraph is then used to generate the answer. This
    needs NO retraining — it runs on the config #4 checkpoint — so ``--refine``
    with the confidence head loaded turns config #4 into config #5.

    Select the full config #5 (refinement) with:
        --model_name adaptive_graph_llm --gnn_model_name graphgps \
        --confidence_head --refine

    Cleanup (Session-10 TODO, applied here for every run through this class):
    ``inference`` truncates each decoded prediction at the first end-of-sequence
    marker (``</s>`` etc.). The EOS string is literal text under the LLaMA-3.1
    tokenizer, so ``skip_special_tokens`` never removed it — the trailing,
    post-EOS hallucination was inflating entity counts and depressing precision/F1
    for every config. Truncating restores a clean answer span.
    """

    # End-of-sequence markers that may appear as *literal text* in a decoded
    # prediction (LLaMA-2 chat EOS + LLaMA-3.1 special-token strings). Ordered
    # by how they show up; we cut at whichever appears earliest.
    EOS_MARKERS = ('</s>', '<|eot_id|>', '<|end_of_text|>')

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        # Gated so an absent flag reproduces config #3 (GraphGPS) byte-for-byte.
        self.use_confidence = bool(getattr(args, 'confidence_head', False))
        self.confidence_weight = float(getattr(args, 'confidence_weight', 0.1))
        self._last_pooled = None  # cache of the most recent pooled graph embedding

        # Week-5 iterative refinement (test-time; consumes the confidence head).
        self.refine = bool(getattr(args, 'refine', False))
        self.refine_tau = float(getattr(args, 'refine_tau', 0.5))
        self.refine_max_iters = int(getattr(args, 'refine_max_iters', 2))
        self.refine_max_nodes = int(getattr(args, 'refine_max_nodes', 60))
        self._refine_iters = None  # per-sample expansion counts from the last batch

        if self.use_confidence:
            hidden = int(args.gnn_hidden_dim)  # pooled GraphGPS embed dim (=1024)
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(float(getattr(args, 'gnn_dropout', 0.0))),
                nn.Linear(hidden // 2, 1),
            ).to(self.model.device)

    # ------------------------------------------------------------------ encode
    def encode_graphs(self, samples):
        """GraphGPS encoding (Contribution 1) + cache the pooled embedding.

        The cache lets ``forward``/``inference`` reuse the exact pooled vector the
        LLM was conditioned on for the confidence head, without duplicating any of
        the parent's tokenization/LLM logic. ``encode_graphs`` is called exactly
        once per forward pass, so a single-slot cache is safe.
        """
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
        self._last_pooled = g_embeds
        return g_embeds

    # ----------------------------------------------------------------- forward
    def forward(self, samples):
        # Parent forward runs encode_graphs (populating self._last_pooled), builds
        # the multimodal prompt, and returns the LM cross-entropy loss.
        lm_loss = super().forward(samples)
        if not self.use_confidence:
            return lm_loss

        targets = self._confidence_targets(samples)              # [B] in {0,1}
        logits = self.confidence_head(self._last_pooled).squeeze(-1)  # [B]
        conf_loss = F.binary_cross_entropy_with_logits(logits, targets)
        return lm_loss + self.confidence_weight * conf_loss

    # --------------------------------------------------------------- inference
    def inference(self, samples):
        # Week-5: refine the retrieved subgraph before generating (test-time).
        if self.refine:
            if not self.use_confidence:
                raise RuntimeError(
                    '--refine needs the confidence head (--confidence_head) — it '
                    'is the stopping signal. Load the config #4 checkpoint.')
            samples = self._refine_batch(samples)

        out = super().inference(samples)  # also refreshes self._last_pooled

        # Cleanup: cut each prediction at the first EOS marker (helps ALL configs).
        if 'pred' in out:
            out['pred'] = [self._truncate(p) for p in out['pred']]

        # Emit the confidence estimate per sample (for offline AUC/calibration;
        # under --refine this is the FINAL, post-expansion subgraph's confidence).
        if self.use_confidence and self._last_pooled is not None:
            with torch.no_grad():
                conf = torch.sigmoid(self.confidence_head(self._last_pooled).squeeze(-1))
            out['confidence'] = conf.detach().cpu().tolist()

        # Emit how many expansion steps each sample took (0 == answered as config #4).
        if self.refine and self._refine_iters is not None:
            out['num_refine_iters'] = list(self._refine_iters)
        return out

    # --------------------------------------------------------------- refinement
    def _refine_batch(self, samples):
        """Expand each sample's subgraph until the confidence head is confident.

        Returns a shallow copy of ``samples`` with ``graph`` (re-batched) and
        ``desc`` replaced by the final expanded subgraph. Per-sample expansion
        counts are stashed in ``self._refine_iters``. Only the tiny GraphGPS
        encoder + MLP run in the loop — the expensive LLM generation happens once,
        afterwards, on the final subgraph.
        """
        ids = samples['id']
        graphs_out, descs_out, iters_out = [], [], []

        for index in ids:
            index = int(index)
            node_ids, edge_ids, q_emb = sr.initial_state(index)
            full_graph, _, _ = sr._load_full(index)

            data, desc = sr.build_subgraph(index, node_ids, edge_ids)
            conf = self._confidence_of(data)

            it = 0
            while conf < self.refine_tau and it < self.refine_max_iters:
                node_ids, edge_ids, added = sr.expand_one_hop(
                    full_graph, node_ids, edge_ids, q_emb, self.refine_max_nodes)
                if added == 0:
                    break  # nothing left to add (or cap saturated) — stop early
                data, desc = sr.build_subgraph(index, node_ids, edge_ids)
                conf = self._confidence_of(data)
                it += 1

            graphs_out.append(data)
            descs_out.append(desc)
            iters_out.append(it)

        samples = dict(samples)
        samples['graph'] = Batch.from_data_list(graphs_out)
        samples['desc'] = descs_out
        self._refine_iters = iters_out
        return samples

    def _confidence_of(self, data):
        """Confidence of a single candidate subgraph (encoder + head, no LLM)."""
        batch = Batch.from_data_list([data]).to(self.model.device)
        with torch.no_grad():
            n_embeds, _ = self.graph_encoder(
                batch.x, batch.edge_index.long(), batch.edge_attr,
                batch.pe, batch.batch)
            pooled = scatter(n_embeds, batch.batch, dim=0, reduce='mean')
            conf = torch.sigmoid(self.confidence_head(pooled).squeeze(-1))
        return float(conf.item())

    # ------------------------------------------------------------- confidence
    def _confidence_targets(self, samples):
        """BCE target: 1 if the retrieved subgraph text contains a gold answer.

        A retrieval-recall proxy — the confidence head learns to predict whether
        the (attention-)retrieved subgraph is answer-bearing, which is exactly the
        signal Week-5 refinement needs (low confidence -> expand the subgraph).
        WebQSP labels are '|'-separated, already lowercased; ``desc`` is the
        subgraph's textual node/edge dump. Substring match is a deliberately
        simple, deterministic proxy (documented as a design choice, not a bug).
        """
        descs = samples['desc']
        labels = samples['label']
        targets = []
        for desc, label in zip(descs, labels):
            desc_l = desc.lower()
            golds = [a.strip().lower() for a in str(label).split('|') if a.strip()]
            hit = any(g in desc_l for g in golds)
            targets.append(1.0 if hit else 0.0)
        return torch.tensor(targets, dtype=torch.float, device=self.model.device)

    @classmethod
    def _truncate(cls, text):
        cut = len(text)
        for marker in cls.EOS_MARKERS:
            i = text.find(marker)
            if i != -1:
                cut = min(cut, i)
        return text[:cut].strip()
