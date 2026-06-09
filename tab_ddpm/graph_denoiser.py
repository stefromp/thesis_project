"""
Graph-aware denoiser for TabDDPM.

GraphAwareDenoiser is a drop-in replacement for MLPDiffusion that treats each
tabular feature as a node in a dependency graph.  The graph is an inductive
bias inside the denoising network — the model output is still a synthetic
tabular row, not a graph.

Two adjacency modes are supported:
  static  — precomputed from training-data statistics (see graph_builder.py),
             stored as a registered buffer and frozen during training.
  dynamic — learned end-to-end via DynamicAdjacency at every forward pass.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules import timestep_embedding
from .graph_builder import DynamicAdjacency
from .gnn_layers import (
    GCNLayer,
    GATLayer,
    GATv2Layer,
    GINLayer,
    TransformerGraphLayer,
    GraphAttentionLayer,   # alias of TransformerGraphLayer — kept for users
                           # who import it from graph_denoiser.
)


# ---------------------------------------------------------------------------
# GNN-layer dispatch
# ---------------------------------------------------------------------------
#
# Five GNN backbones are supported via the `gnn_type` argument of
# GraphAwareDenoiser. They all share the same I/O contract:
#
#     forward(x: (B, N, d_model), adj: (N, N) or (B, N, N)) -> (B, N, d_model)
#
# `graphmha`  — graph-masked multi-head attention with FFN (the original
#               GraphAttentionLayer below). Highest per-layer capacity.
# `gcn`       — Kipf & Welling 2017, symmetric-normalised aggregation.
#               Cheapest; respects the data-derived adjacency strictly.
# `gat`       — Velickovic et al. 2018, canonical GAT. Static attention
#               (gat_v2 Theorem 1): the per-key ranking is shared across
#               queries.
# `gatv2`     — Brody, Alon & Yahav 2022. Dynamic attention; strictly more
#               expressive than GAT and more robust to structural noise
#               in the adjacency (relevant since our static adjacency is
#               built from finite-sample statistics).
# `gin`       — Xu et al. 2019. Sum + MLP aggregation with (1+eps) self-term;
#               maximally expressive aggregator. Useful when categorical
#               nodes form effectively discrete multisets.
# ---------------------------------------------------------------------------

_GNN_TYPES = ("graphmha", "gcn", "gat", "gatv2", "gin")


def _build_gnn_layer(
    gnn_type: str,
    d_model: int,
    n_heads: int,
    dropout: float,
) -> nn.Module:
    """Instantiate one GNN layer of the requested type."""
    if gnn_type == "graphmha":
        return TransformerGraphLayer(d_model, n_heads, dropout=dropout)
    if gnn_type == "gcn":
        return GCNLayer(d_model, dropout=dropout)
    if gnn_type == "gat":
        return GATLayer(d_model, n_heads, dropout=dropout)
    if gnn_type == "gatv2":
        return GATv2Layer(d_model, n_heads, dropout=dropout)
    if gnn_type == "gin":
        return GINLayer(d_model, dropout=dropout)
    raise ValueError(
        f"unknown gnn_type {gnn_type!r}; expected one of {_GNN_TYPES}"
    )


# ---------------------------------------------------------------------------
# Graph-aware denoiser
# ---------------------------------------------------------------------------

class GraphAwareDenoiser(nn.Module):
    """Graph-aware denoiser; drop-in replacement for MLPDiffusion.

    Each feature is mapped to a graph node.  N = d_num + len(cat_sizes) nodes
    in total.  Numerical features produce scalar-input nodes; each categorical
    feature produces one node whose initial embedding comes from its one-hot
    slice.  After L graph attention layers, every node is projected back to its
    original feature space and the outputs are concatenated to reconstruct the
    full input vector.

    Accepts the same forward signature as MLPDiffusion:
        forward(x, timesteps, y=None) → Tensor of shape (B, d_in)

    Args:
        d_in:       total input / output dimension (d_num + sum(cat_sizes)).
        num_classes: > 0 for classification, 0 for regression (label cond.).
        is_y_cond:  whether to condition on labels.
        d_num:      number of numerical features.
        cat_sizes:  cardinalities of categorical features (empty list = none).
        d_model:    hidden dimension per node.
        n_layers:   number of graph attention layers.
        n_heads:    number of attention heads (must divide d_model).
        graph_mode: 'static' or 'dynamic'.
        adjacency:  (N, N) binary tensor for static mode; None for dynamic.
        top_k:      top-k sparsification for DynamicAdjacency (0 = dense).
        gnn_type:   which GNN backbone to stack. One of
                    {"graphmha", "gcn", "gat", "gatv2", "gin"}.
                    Default "graphmha" preserves the original behaviour
                    (graph-masked multi-head attention + FFN).
        dropout:    dropout rate inside each GNN layer (attention / FFN).
        dim_t:      sinusoidal timestep embedding dimension.
    """

    def __init__(
        self,
        d_in: int,
        num_classes: int,
        is_y_cond: bool,
        d_num: int,
        cat_sizes: List[int],
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        graph_mode: str = "static",
        adjacency: Optional[Tensor] = None,
        top_k: int = 0,
        gnn_type: str = "graphmha",
        dropout: float = 0.0,
        dim_t: int = 128,
    ) -> None:
        super().__init__()

        self.d_num = d_num
        self.cat_sizes = list(cat_sizes)
        self.n_cat = len(cat_sizes)
        self.N = d_num + self.n_cat
        self.d_model = d_model
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond
        self.graph_mode = graph_mode

        gnn_type = gnn_type.lower()
        if gnn_type not in _GNN_TYPES:
            raise ValueError(
                f"unknown gnn_type {gnn_type!r}; expected one of {_GNN_TYPES}"
            )
        self.gnn_type = gnn_type
        self.dropout = dropout

        if self.N == 0:
            raise ValueError("GraphAwareDenoiser requires at least one feature (node).")

        # ------------------------------------------------------------------
        # Input projections — one per node
        # ------------------------------------------------------------------
        # Numerical: x[:, i] (scalar) → d_model via per-feature affine.
        # Stored as weight (d_num, d_model) and bias (d_num, d_model).
        if d_num > 0:
            self.num_in_weight = nn.Parameter(torch.empty(d_num, d_model))
            self.num_in_bias = nn.Parameter(torch.zeros(d_num, d_model))
            nn.init.kaiming_uniform_(self.num_in_weight, a=math.sqrt(5))

        # Categorical: x[:, offset:offset+K_i] (one-hot) → d_model via Linear.
        if self.n_cat > 0:
            self.cat_in_projs = nn.ModuleList(
                [nn.Linear(K, d_model) for K in cat_sizes]
            )

        # ------------------------------------------------------------------
        # Timestep conditioning
        # ------------------------------------------------------------------
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_model),
        )

        # ------------------------------------------------------------------
        # Label conditioning
        # ------------------------------------------------------------------
        if is_y_cond:
            if num_classes > 0:
                self.label_emb: nn.Module = nn.Embedding(num_classes, d_model)
            else:
                self.label_emb = nn.Linear(1, d_model)

        # ------------------------------------------------------------------
        # Adjacency
        # ------------------------------------------------------------------
        if graph_mode == "static":
            if adjacency is not None:
                self.register_buffer("static_adj", adjacency.float())
            else:
                # Placeholder replaced by load_state_dict when loading a
                # trained checkpoint (identity = self-loops only).
                self.register_buffer("static_adj", torch.eye(self.N))
        else:
            self.dynamic_adj = DynamicAdjacency(self.N, d_model, top_k)

        # ------------------------------------------------------------------
        # GNN backbone layers — chosen via gnn_type:
        #   graphmha : graph-masked multi-head attention (the original block).
        #   gcn      : Kipf & Welling (2017), symmetric-normalised aggregation.
        #   gat      : Velickovic et al. (2018), static masked attention.
        #   gatv2    : Brody, Alon & Yahav (2022), dynamic masked attention.
        #   gin      : Xu et al. (2019), injective sum + MLP (WL-expressive).
        # Every layer shares the (x, adj) -> x interface defined in
        # gnn_layers.py, so the forward pass below is identical for all.
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                _build_gnn_layer(
                    self.gnn_type,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # ------------------------------------------------------------------
        # Output projections — one per node
        # ------------------------------------------------------------------
        if d_num > 0:
            self.num_out_weight = nn.Parameter(torch.empty(d_num, d_model))
            self.num_out_bias = nn.Parameter(torch.zeros(d_num))
            nn.init.kaiming_uniform_(self.num_out_weight, a=math.sqrt(5))

        if self.n_cat > 0:
            self.cat_out_projs = nn.ModuleList(
                [nn.Linear(d_model, K) for K in cat_sizes]
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, timesteps: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Denoise one batch.

        Args:
            x:          (B, d_in) noisy input (d_num numericals + one-hot cats).
            timesteps:  (B,) integer diffusion timesteps.
            y:          (B,) or (B, 1) class labels / regression targets, or None.

        Returns:
            (B, d_in) predicted denoised output.
        """
        B = x.shape[0]

        # ------------------------------------------------------------------
        # 1. Build node embeddings  (B, N, d_model)
        # ------------------------------------------------------------------
        parts: List[Tensor] = []

        if self.d_num > 0:
            x_num = x[:, : self.d_num].float()  # (B, d_num)
            # x_num[:, i] * weight[i] + bias[i]  →  (B, d_num, d_model)
            num_nodes = (
                x_num.unsqueeze(-1) * self.num_in_weight.unsqueeze(0)
                + self.num_in_bias.unsqueeze(0)
            )
            parts.append(num_nodes)

        if self.n_cat > 0:
            offset = self.d_num
            cat_nodes: List[Tensor] = []
            for i, K in enumerate(self.cat_sizes):
                slice_i = x[:, offset : offset + K].float()  # (B, K)
                cat_nodes.append(self.cat_in_projs[i](slice_i))  # (B, d_model)
                offset += K
            parts.append(torch.stack(cat_nodes, dim=1))  # (B, n_cat, d_model)

        node_emb = torch.cat(parts, dim=1)  # (B, N, d_model)

        # ------------------------------------------------------------------
        # 2. Timestep and label conditioning
        # ------------------------------------------------------------------
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))  # (B, d_model)
        node_emb = node_emb + t_emb.unsqueeze(1)  # broadcast over N

        if self.is_y_cond and y is not None:
            if self.num_classes > 0:
                y_emb = F.silu(self.label_emb(y.squeeze()))       # (B, d_model)
            else:
                y_emb = F.silu(self.label_emb(y.view(B, 1).float()))
            node_emb = node_emb + y_emb.unsqueeze(1)

        # ------------------------------------------------------------------
        # 3 & 4. Adjacency + Graph attention layers
        # ------------------------------------------------------------------
        if self.graph_mode == "static":
            adj = self.static_adj  # (N, N) — frozen, computed once
            for layer in self.layers:
                node_emb = layer(node_emb, adj)
        else:
            for layer in self.layers:
                adj = self.dynamic_adj(node_emb)  # (B, N, N) — recomputed from current embeddings
                node_emb = layer(node_emb, adj)

        # ------------------------------------------------------------------
        # 5. Readout — project each node back to its original feature space
        # ------------------------------------------------------------------
        output_parts: List[Tensor] = []

        if self.d_num > 0:
            # (B, d_num, d_model) ⊙ (d_num, d_model) → sum → (B, d_num)
            num_out = (
                (node_emb[:, : self.d_num, :] * self.num_out_weight.unsqueeze(0))
                .sum(-1)
                + self.num_out_bias.unsqueeze(0)
            )
            output_parts.append(num_out)

        if self.n_cat > 0:
            for i in range(self.n_cat):
                output_parts.append(
                    self.cat_out_projs[i](node_emb[:, self.d_num + i, :])
                )  # (B, K_i)

        return torch.cat(output_parts, dim=1)  # (B, d_in)
