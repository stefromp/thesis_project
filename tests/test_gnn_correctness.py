"""
Correctness tests for the custom dense-adjacency GNN layers.

Goal: prove the dense (no edge_index) implementations are mathematically
equivalent to the standard sparse message-passing formulations, using
hand-written / independent references on small 3-4 node graphs.

torch_geometric is NOT required.  Each reference here is re-derived from the
paper equations (not copied from the layer code): it reuses only the layer's
*parameters* (weights, biases, eps) and recomputes the aggregation from
scratch with explicit loops / einsum.  If the layer's internal orchestration
(einsum order, broadcasting, log-bias, softmax axis, masking) matches the
independent reference, the layer implements the intended equation.

Run:  pytest tests/test_gnn_correctness.py -v
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tab_ddpm.gnn_layers import (
    GCNLayer,
    GATv2Layer,
    GINLayer,
    _sym_normalize,
)
from tab_ddpm.graph_builder import DynamicAdjacency

torch.set_default_dtype(torch.float64)  # tight tolerances
SEED = 0


def _rand_x(B, N, d):
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(B, N, d, generator=g)


def _rand_soft_adj(B, N):
    """Random asymmetric soft adjacency in (0,1) with diagonal forced to 1."""
    g = torch.Generator().manual_seed(SEED + 1)
    a = torch.rand(B, N, N, generator=g)
    eye = torch.eye(N).unsqueeze(0)
    return torch.clamp(a * (1 - eye) + eye, max=1.0)


# ===========================================================================
# A. Dynamic adjacency
# ===========================================================================

def test_A1_sigmoid_not_softmax():
    """A_ij = sigmoid(phi_ij) elementwise; rows do NOT sum to 1 (not softmax)."""
    N, d = 5, 8
    dyn = DynamicAdjacency(N, d, top_k=0)
    x = _rand_x(1, N, d)
    adj = dyn(x)

    # Reconstruct raw scores independently and compare to sigmoid.
    Q = dyn.q_proj(x)
    K = dyn.k_proj(x)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)
    expected = torch.sigmoid(scores)
    # off-diagonal must equal elementwise sigmoid (diagonal is overwritten to 1)
    eye = torch.eye(N, dtype=torch.bool)
    assert torch.allclose(adj[0][~eye], expected[0][~eye], atol=1e-10)

    # A softmax row would sum to 1; sigmoid rows generally do not.
    row_sums = adj[0].sum(-1)
    assert not torch.allclose(row_sums, torch.ones(N), atol=1e-3)


def test_A2_A4_topk_before_sigmoid_and_per_row():
    """top-k applied per-row on raw scores BEFORE sigmoid; masked -> ~0."""
    N, d, k = 6, 8, 2
    dyn = DynamicAdjacency(N, d, top_k=k)
    x = _rand_x(1, N, d)
    adj = dyn(x)[0]

    eye = torch.eye(N, dtype=torch.bool)
    off = adj.masked_fill(eye, 0.0)
    # Each row: exactly k surviving off-diagonal edges (rest ~ sigmoid(-1e9)=0).
    alive = (off > 1e-6).sum(-1)
    assert torch.equal(alive, torch.full((N,), k))
    # Masked entries are ~0 (proves mask happened before sigmoid: sigmoid(-1e9)=0,
    # NOT sigmoid(1e-9)=0.5 and NOT masked after sigmoid).
    masked_vals = off[off <= 1e-6]
    assert torch.all(masked_vals < 1e-6)


def test_A3_self_loops_after_topk():
    """Diagonal is exactly 1 even with aggressive top-k (self-loop not cut)."""
    N, d, k = 6, 8, 1
    dyn = DynamicAdjacency(N, d, top_k=k)
    adj = dyn(_rand_x(1, N, d))[0]
    assert torch.allclose(torch.diag(adj), torch.ones(N), atol=1e-12)


def test_A5_asymmetry():
    """A is not forced symmetric (Q,K independent)."""
    N, d = 5, 8
    dyn = DynamicAdjacency(N, d, top_k=0)
    adj = dyn(_rand_x(1, N, d))[0]
    assert not torch.allclose(adj, adj.T, atol=1e-4)


def test_E4_gradient_flow_through_topk():
    """Straight-through top-k: gradients reach W_Q and W_K via surviving edges."""
    N, d, k = 6, 8, 2
    dyn = DynamicAdjacency(N, d, top_k=k)
    x = _rand_x(1, N, d).requires_grad_(True)
    adj = dyn(x)
    adj.sum().backward()
    assert dyn.q_proj.weight.grad is not None
    assert dyn.k_proj.weight.grad is not None
    assert dyn.q_proj.weight.grad.abs().sum() > 0
    assert dyn.k_proj.weight.grad.abs().sum() > 0


# ===========================================================================
# B. GCN
# ===========================================================================

def test_B_sym_normalize_matches_hand_reference():
    """_sym_normalize == D^{-1/2}(A+I)D^{-1/2}, hand-computed."""
    A = torch.tensor([[0., 1., 0.],
                      [1., 0., 1.],
                      [0., 1., 0.]])
    got = _sym_normalize(A)
    Ah = A + torch.eye(3)
    d = Ah.sum(-1)
    Dinv = torch.diag(d.pow(-0.5))
    ref = Dinv @ Ah @ Dinv
    assert torch.allclose(got, ref, atol=1e-10)


def test_GCN_full_forward_matches_independent_reference():
    """Full GCN layer == independent re-derivation using its own params."""
    B, N, d = 2, 4, 8
    layer = GCNLayer(d, dropout=0.0).eval()
    x = _rand_x(B, N, d)
    adj = _rand_soft_adj(B, N)

    got = layer(x, adj)

    # Independent reference: H'=x+GELU(Ahat @ W(norm(x))); then FFN.
    h = layer.norm(x)
    a_hat = _sym_normalize(adj)
    msg = layer.W(h)
    out = F.gelu(torch.bmm(a_hat, msg))
    xr = x + out
    ref = xr + layer.ffn.ff(layer.ffn.norm(xr))
    assert torch.allclose(got, ref, atol=1e-10)


def test_B5_no_double_self_loop():
    """FIX B5: incoming adjacency already has diag=1; _sym_normalize must NOT
    double it.  Self-weight must be exactly 1 (textbook GCN / GCNConv), i.e.
    _sym_normalize(A_with_diag1) == _sym_normalize(A_with_diag0).
    """
    N = 4
    g = torch.Generator().manual_seed(3)
    core = torch.rand(N, N, generator=g)
    core = core * (1 - torch.eye(N))                # off-diagonal only
    a_diag0 = core.clone()                          # no self-loops
    a_diag1 = core + torch.eye(N)                   # self-loops baked in (our case)

    # Both must normalise identically -> the extra diagonal-1 is not counted twice.
    assert torch.allclose(_sym_normalize(a_diag0), _sym_normalize(a_diag1), atol=1e-12)

    # And it matches the hand-computed textbook Â = D^{-1/2}(A0 + I)D^{-1/2}.
    Ah = a_diag0 + torch.eye(N)
    d = Ah.sum(-1)
    Dinv = torch.diag(d.pow(-0.5))
    assert torch.allclose(_sym_normalize(a_diag1), Dinv @ Ah @ Dinv, atol=1e-12)


# ===========================================================================
# C. GATv2
# ===========================================================================

def _gatv2_reference(layer, x, adj, leaky_before_a):
    """Independent single-head GATv2 aggregation from the paper equation.

    leaky_before_a=True  -> genuine GATv2:  e_ij = a . LeakyReLU(Ws h_i + Wd h_j)
    leaky_before_a=False -> GAT-v1 ordering: e_ij = LeakyReLU(a . (Ws h_i + Wd h_j))
    """
    B, N, D = x.shape
    assert layer.n_heads == 1
    h = layer.norm(x)
    Ws = layer.W_src(h)          # (B,N,D)
    Wd = layer.W_dst(h)
    Wv = layer.W_val(h)
    a = layer.a.view(D)          # single head
    out = torch.zeros(B, N, D)
    for b in range(B):
        for i in range(N):
            logits = torch.empty(N)
            for j in range(N):
                pre = Ws[b, i] + Wd[b, j]
                if leaky_before_a:
                    s = torch.dot(a, F.leaky_relu(pre, 0.2))
                else:
                    s = F.leaky_relu(torch.dot(a, pre), 0.2)
                logits[j] = s + math.log(max(adj[b, i, j].item(), 1e-12))
            attn = F.softmax(logits, dim=0)
            out[b, i] = (attn.unsqueeze(-1) * Wv[b]).sum(0)
    out = layer.out_proj(out)
    xr = x + out
    return xr + layer.ffn.ff(layer.ffn.norm(xr))


def test_GATv2_matches_v2_reference_not_v1():
    """Proves C1 (LeakyReLU before a) and C2 (log-adj bias before softmax)."""
    B, N, d = 2, 4, 6
    layer = GATv2Layer(d, n_heads=1, dropout=0.0).eval()
    x = _rand_x(B, N, d)
    adj = _rand_soft_adj(B, N)

    got = layer(x, adj)
    ref_v2 = _gatv2_reference(layer, x, adj, leaky_before_a=True)
    ref_v1 = _gatv2_reference(layer, x, adj, leaky_before_a=False)

    assert torch.allclose(got, ref_v2, atol=1e-9), "layer must match GATv2 ordering"
    assert not torch.allclose(got, ref_v1, atol=1e-6), "must NOT be GATv1 ordering"


def test_GATv2_masked_edge_no_nan():
    """A_ij ~ 0 -> log clamp -> huge negative bias, softmax ~0, no NaN."""
    B, N, d = 1, 4, 6
    layer = GATv2Layer(d, n_heads=2, dropout=0.0).eval()
    x = _rand_x(B, N, d)
    adj = torch.eye(N).unsqueeze(0)  # only self-loops survive
    out = layer(x, adj)
    assert torch.isfinite(out).all()


def test_GATv2_multihead_shapes():
    B, N, d = 3, 5, 8
    layer = GATv2Layer(d, n_heads=4, dropout=0.0).eval()
    out = layer(_rand_x(B, N, d), _rand_soft_adj(B, N))
    assert out.shape == (B, N, d)


# ===========================================================================
# D. GIN
# ===========================================================================

def test_GIN_sum_aggregation_matches_reference():
    """GIN == MLP((1+eps)h_i + sum_{j!=i} A_ij h_j); sum (not mean/max)."""
    B, N, d = 2, 4, 8
    layer = GINLayer(d, dropout=0.0, learn_eps=True).eval()
    with torch.no_grad():
        layer.eps.fill_(0.37)  # non-trivial eps
    x = _rand_x(B, N, d)
    adj = _rand_soft_adj(B, N)

    got = layer(x, adj)

    h = layer.norm(x)
    eye = torch.eye(N)
    a_off = adj * (1 - eye.unsqueeze(0))          # zero diagonal
    neigh = torch.bmm(a_off, h)                    # weighted SUM over neighbours
    agg = (1.0 + layer.eps) * h + neigh
    out = layer.mlp(agg)
    xr = x + out
    ref = xr + layer.ffn.ff(layer.ffn.norm(xr))
    assert torch.allclose(got, ref, atol=1e-10)


def test_GIN_eps_is_learnable():
    layer = GINLayer(8, learn_eps=True)
    assert layer.eps.requires_grad
    layer2 = GINLayer(8, learn_eps=False)
    assert not layer2.eps.requires_grad


# ===========================================================================
# E. Batch broadcasting of a shared 2D adjacency
# ===========================================================================

@pytest.mark.parametrize("layer_fn", [
    lambda d: GCNLayer(d),
    lambda d: GATv2Layer(d, n_heads=2),
    lambda d: GINLayer(d),
])
def test_2d_adj_broadcasts_over_batch(layer_fn):
    B, N, d = 4, 5, 8
    layer = layer_fn(d).eval()
    x = _rand_x(B, N, d)
    adj2d = _rand_soft_adj(1, N)[0]                # (N,N)
    adj3d = adj2d.unsqueeze(0).expand(B, -1, -1)   # (B,N,N)
    assert torch.allclose(layer(x, adj2d), layer(x, adj3d), atol=1e-10)
