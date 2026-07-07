import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from src.model.graph_llm import GraphLLM
import src.model.graphgps  # noqa: F401  (registers 'graphgps' in load_gnn_model on import)


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
    the flag is absent). The confidence scalar is also emitted at inference time,
    which is the signal Week 5's iterative-refinement loop will consume.

    Select the full config #4 with:
        --model_name adaptive_graph_llm --gnn_model_name graphgps --confidence_head

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
        out = super().inference(samples)  # also refreshes self._last_pooled

        # Cleanup: cut each prediction at the first EOS marker (helps ALL configs).
        if 'pred' in out:
            out['pred'] = [self._truncate(p) for p in out['pred']]

        # Emit the confidence estimate per sample (consumed by Week-5 refinement;
        # also lets us score the head's calibration/AUC offline from the CSV).
        if self.use_confidence and self._last_pooled is not None:
            with torch.no_grad():
                conf = torch.sigmoid(self.confidence_head(self._last_pooled).squeeze(-1))
            out['confidence'] = conf.detach().cpu().tolist()
        return out

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
