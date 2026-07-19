"""
GNN backbone layers for GraphAwareDenoiser.

Every layer is a drop-in unit operating on a feature-graph with N = d_num +
n_cat nodes, one node per tabular feature.  All layers share the same API

    forward(x, adj) -> x_new

where
    x   : (B, N, d_model)  node embeddings,
    adj : (N, N)           static binary adjacency,  OR
          (B, N, N)        dynamic soft adjacency in [0, 1],
    out : (B, N, d_model)  updated node embeddings.

This uniform API lets GraphAwareDenoiser swap GNN backbones without touching
the rest of the forward pass.  Static adjacency = hard mask; dynamic = soft
gate.  Both are handled by the same code path inside each layer.

Implemented backbones
---------------------
GCNLayer   : Kipf & Welling (2017)  -- symmetric-normalised mean aggregation.
GATLayer   : Velickovic et al. (2018) -- static masked attention.
GATv2Layer : Brody et al. (2022) -- dynamic masked attention.
GINLayer   : Xu et al. (2019) -- sum aggregation + MLP, WL-expressive.
TransformerGraphLayer (legacy) : multi-head transformer-style attention
             gated by adjacency. Kept for backwards compatibility with the
             existing GraphAwareDenoiser checkpoints.

Design choices (per gnn_imp.pdf -- Luo et al., "Classic GNNs are Strong
Baselines"):
  * Pre-norm LayerNorm before message passing.
  * Residual connection around the aggregation block.
  * Dropout on activations (default 0.0 to preserve existing behaviour).
  * Post-aggregation FFN (4x expansion, GELU) for capacity, mirroring the
    transformer-style block already used in the legacy attention layer so
    all backbones have comparable parameter budgets.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def _expand_adj(adj: Tensor, B: int) -> Tensor:
    """Return adjacency as a (B, N, N) tensor regardless of input rank."""
    if adj.dim() == 2:
        return adj.unsqueeze(0).expand(B, -1, -1)
    return adj


def _sym_normalize(adj: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric normalisation D^{-1/2} (A + I) D^{-1/2}.

    Works for either binary or soft adjacency.  Self-loops are (re)added with
    weight exactly 1 before normalisation: the incoming adjacency already has
    a diagonal set to 1 (static builder / DynamicAdjacency), so we first zero
    the diagonal and then add I.  This gives self-weight 1 (matching textbook
    GCN / PyG GCNConv) rather than 2 from double-counting.  Operates on the
    last two dims of any (..., N, N) tensor.
    """
    N = adj.size(-1)
    eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
    if adj.dim() == 2:
        a = adj * (1.0 - eye) + eye
    else:
        eye_b = eye.unsqueeze(0)
        a = adj * (1.0 - eye_b) + eye_b
    deg = a.sum(dim=-1).clamp(min=eps)        # (..., N)
    d_inv_sqrt = deg.pow(-0.5)                # (..., N)
    if adj.dim() == 2:
        return d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    return d_inv_sqrt.unsqueeze(-1) * a * d_inv_sqrt.unsqueeze(-2)


# ---------------------------------------------------------------------------
# Shared FFN block (post-aggregation)
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    """4x-expansion GELU FFN, pre-norm, with residual.  Identical across all
    backbones so a fair comparison is possible at matched parameter budget."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


# ---------------------------------------------------------------------------
# GCN -- Kipf & Welling, ICLR 2017
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Graph Convolutional Network layer.

    Aggregation:
        H' = sigma( D^{-1/2} (A + I) D^{-1/2} H W )

    For dynamic adjacency the normalisation is recomputed each forward pass
    from the soft adjacency, so edge weights affect both *which* neighbours
    contribute and *how much*.

    Pre-norm, residual, plus a post-aggregation FFN for capacity parity with
    the attention backbones.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.W = nn.Linear(d_model, d_model, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ffn = _FFN(d_model, dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, N, _ = x.shape
        h = self.norm(x)

        a_hat = _sym_normalize(adj)            # (N, N) or (B, N, N)
        msg = self.W(h)                        # (B, N, d_model)
        if a_hat.dim() == 2:
            out = torch.einsum("ij,bjd->bid", a_hat, msg)
        else:
            out = torch.bmm(a_hat, msg)        # (B, N, d_model)
        out = self.drop(self.act(out))

        x = x + out                            # residual
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# GAT -- Velickovic et al., ICLR 2018
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Graph Attention Network layer (original GAT).

    Multi-head masked attention with the *original* GAT scoring function:

        e_{ij} = LeakyReLU( a^T [W h_i || W h_j] )
               = LeakyReLU( a_src^T (W h_i) + a_dst^T (W h_j) )

    Note (Brody et al., 2022, Theorem 1): this scoring computes only **static**
    attention -- the argmax-key is shared across all queries.  Provided as a
    baseline, *use GATv2Layer for dynamic edge weighting*.

    Edges outside the adjacency are masked (softmax over neighbours only).
    For soft dynamic adjacency, adjacency values gate the attention scores
    additively in log-space (log(adj + eps)), so a near-zero edge weight is
    treated as an effectively-masked edge while staying differentiable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.leaky_slope = leaky_slope

        self.norm = nn.LayerNorm(d_model)
        self.W = nn.Linear(d_model, d_model, bias=False)
        # Attention vector a = [a_src || a_dst], one per head.
        self.a_src = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = _FFN(d_model, dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, N, D = x.shape
        h_in = x
        h = self.norm(x)
        Wh = self.W(h).view(B, N, self.n_heads, self.d_head)       # (B, N, H, dh)

        # Per-head src/dst contributions (decomposed concat -- see Brody Eq. 5).
        e_src = (Wh * self.a_src).sum(dim=-1)                       # (B, N, H)
        e_dst = (Wh * self.a_dst).sum(dim=-1)                       # (B, N, H)
        # e_{ij} = LeakyReLU( e_src_i + e_dst_j ), broadcast to (B, H, N, N)
        scores = e_src.permute(0, 2, 1).unsqueeze(-1) + \
                 e_dst.permute(0, 2, 1).unsqueeze(-2)
        scores = F.leaky_relu(scores, negative_slope=self.leaky_slope)

        # Adjacency mask: hard binary -> additive -inf; soft -> additive log.
        adj_b = _expand_adj(adj, B).unsqueeze(1)                    # (B, 1, N, N)
        scores = scores + torch.log(adj_b.clamp(min=1e-12))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate: (B, H, N, N) x (B, H, N, dh) -> (B, H, N, dh)
        V = Wh.permute(0, 2, 1, 3)                                  # (B, H, N, dh)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)

        x = h_in + out
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# GATv2 -- Brody, Alon & Yahav, ICLR 2022
# ---------------------------------------------------------------------------

class GATv2Layer(nn.Module):
    """Graph Attention Network v2 layer.

    Fix to GAT (Brody et al., 2022): apply the linear scoring vector *after*
    the nonlinearity, so attention becomes a universal approximator:

        e_{ij} = a^T LeakyReLU( W [h_i || h_j] )

    Equivalently, with W = [W_src || W_dst]:

        e_{ij} = a^T LeakyReLU( W_src h_i + W_dst h_j )

    This is strictly more expressive than GAT -- it can express *dynamic*
    attention where different queries pick different keys (Theorem 2 in the
    paper).  Highly relevant for our setting: as t varies along the diffusion
    schedule, the "important" inter-feature dependencies shift, so the model
    benefits from per-(t, x_t) edge weighting that monotonic GAT cannot give.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.leaky_slope = leaky_slope

        self.norm = nn.LayerNorm(d_model)
        # Separate src / dst linear maps so we add features *before* the
        # nonlinearity -- this is the crucial difference vs. GAT.
        self.W_src = nn.Linear(d_model, d_model, bias=False)
        self.W_dst = nn.Linear(d_model, d_model, bias=False)
        self.W_val = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.a)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = _FFN(d_model, dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, N, D = x.shape
        h_in = x
        h = self.norm(x)

        Wh_src = self.W_src(h).view(B, N, self.n_heads, self.d_head)   # (B,N,H,dh)
        Wh_dst = self.W_dst(h).view(B, N, self.n_heads, self.d_head)
        V      = self.W_val(h).view(B, N, self.n_heads, self.d_head)

        # Broadcast-add to form (B, H, N, N, dh): pre-activation pairs.
        src = Wh_src.permute(0, 2, 1, 3).unsqueeze(-2)   # (B, H, N, 1, dh)
        dst = Wh_dst.permute(0, 2, 1, 3).unsqueeze(-3)   # (B, H, 1, N, dh)
        pair = F.leaky_relu(src + dst, negative_slope=self.leaky_slope)

        # Dynamic scoring: a^T applied AFTER the nonlinearity.
        scores = (pair * self.a.view(1, self.n_heads, 1, 1, self.d_head)) \
                    .sum(dim=-1)                          # (B, H, N, N)

        # Adjacency gating in log-space (handles hard 0 and soft values).
        adj_b = _expand_adj(adj, B).unsqueeze(1)          # (B, 1, N, N)
        scores = scores + torch.log(adj_b.clamp(min=1e-12))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        Vh = V.permute(0, 2, 1, 3)                        # (B, H, N, dh)
        out = torch.matmul(attn, Vh)                      # (B, H, N, dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)

        x = h_in + out
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# GIN -- Xu et al., ICLR 2019
# ---------------------------------------------------------------------------

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer.

    Aggregation (Eq. 4.1 of Xu et al., 2019):

        h_v^{(k)} = MLP( (1 + eps) * h_v^{(k-1)} + sum_{u in N(v)} h_u^{(k-1)} )

    Sum aggregation makes GIN's update injective on multisets, giving it the
    same discriminative power as the Weisfeiler-Lehman test -- the strongest
    among standard message-passing GNNs.  In our setting this means GIN can,
    in principle, distinguish feature-value multisets that GCN/GAT collapse,
    which is useful when many features carry similar marginal statistics but
    differ jointly.

    eps is learnable (`GIN-eps`).  Soft adjacency is used as edge weights in
    the weighted sum, so dynamic mode still works.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        learn_eps: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))
        # GIN's MLP: 2 linear layers with GELU + dropout (matches gnn_imp.pdf
        # advice on dropout being essential).
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn = _FFN(d_model, dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, N, _ = x.shape
        h_in = x
        h = self.norm(x)

        # Zero diagonal to compute neighbour sum (we add (1+eps) * h_v separately).
        # adj has self-loops baked in by graph_builder; subtract them off here.
        eye = torch.eye(N, device=h.device, dtype=h.dtype)
        if adj.dim() == 2:
            a_off = adj * (1.0 - eye)
            neigh_sum = torch.einsum("ij,bjd->bid", a_off, h)
        else:
            a_off = adj * (1.0 - eye.unsqueeze(0))
            neigh_sum = torch.bmm(a_off, h)

        out = self.mlp((1.0 + self.eps) * h + neigh_sum)

        x = h_in + out                              # residual
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Legacy transformer-style attention
# ---------------------------------------------------------------------------

class TransformerGraphLayer(nn.Module):
    """Transformer-style multi-head attention gated by adjacency.

    Identical to the original `GraphAttentionLayer` in graph_denoiser.py --
    kept here under a clearer name so all backbones live in one file.
    Acts like a soft GATv2 with a different scoring function (scaled
    dot-product instead of additive).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, N, D = x.shape
        h, dh = self.n_heads, self.d_head

        residual = x
        x_ln = self.norm1(x)
        Q = self.q_proj(x_ln).view(B, N, h, dh).transpose(1, 2)
        K = self.k_proj(x_ln).view(B, N, h, dh).transpose(1, 2)
        V = self.v_proj(x_ln).view(B, N, h, dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        if adj.dim() == 2:
            adj_gate = adj.unsqueeze(0).unsqueeze(0)
        else:
            adj_gate = adj.unsqueeze(1)
        attn = attn * adj_gate
        row_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn = attn / row_sum
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.ff(self.norm2(x))
        return x


# Backwards-compatibility alias for existing import sites.
GraphAttentionLayer = TransformerGraphLayer


# ---------------------------------------------------------------------------
# Dense self-attention (global channel, adjacency-free)
# ---------------------------------------------------------------------------

class SelfAttentionLayer(nn.Module):
    """Dense multi-head self-attention over feature-nodes, pre-norm + FFN.

    Unlike the graph layers above, this sublayer does **not** use the
    adjacency: every node attends to every other node.  It provides the
    "global" mixing channel in an attention->GNN block, while the GNN
    sublayers that follow provide the adjacency-constrained "local" channel.

    Each instance owns its own Q/K/V/out projections, so when several blocks
    are stacked the attention parameters differ per block and the attention
    pattern is recomputed from each block's (already updated) embeddings --
    attention adapts with depth rather than being shared across layers.

    Signature mirrors the graph layers, `forward(x, adj=None)`, with `adj`
    accepted and ignored so blocks can call every sublayer uniformly.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = _FFN(d_model, dropout)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        B, N, D = x.shape
        h, dh = self.n_heads, self.d_head

        residual = x
        x_ln = self.norm1(x)
        Q = self.q_proj(x_ln).view(B, N, h, dh).transpose(1, 2)
        K = self.k_proj(x_ln).view(B, N, h, dh).transpose(1, 2)
        V = self.v_proj(x_ln).view(B, N, h, dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        x = residual + out
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

GNN_REGISTRY = {
    "gcn":         GCNLayer,
    "gat":         GATLayer,
    "gatv2":       GATv2Layer,
    "gin":         GINLayer,
    "transformer": TransformerGraphLayer,   # = original behaviour
}


def build_gnn_layer(
    gnn_type: str,
    d_model: int,
    n_heads: int = 4,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct one GNN layer by name.

    Args:
        gnn_type: one of {'gcn', 'gat', 'gatv2', 'gin', 'transformer'}.
        d_model:  hidden dimension per node.
        n_heads:  number of attention heads (ignored by GCN/GIN).
        dropout:  dropout probability inside attention/FFN.

    Returns:
        nn.Module with signature  forward(x, adj) -> x_new.
    """
    key = gnn_type.lower()
    if key not in GNN_REGISTRY:
        raise ValueError(
            f"Unknown gnn_type='{gnn_type}'. "
            f"Choose from {list(GNN_REGISTRY)}."
        )
    cls = GNN_REGISTRY[key]
    # GCN and GIN ignore n_heads -- keep their constructors clean.
    if key in ("gcn", "gin"):
        return cls(d_model=d_model, dropout=dropout)
    return cls(d_model=d_model, n_heads=n_heads, dropout=dropout)
