"""
Equivalence tests: custom dense-adjacency message passing == torch_geometric.

Proves that the dense (no edge_index) aggregation each custom layer performs is
numerically identical to the canonical sparse PyG conv on the same graph with
the same weights.  We convert the dense binary adjacency A to an edge_index for
the PyG reference (A[i,j]=1 means j->i, i.e. src=j, dst=i).

We compare the *message-passing core* -- the part that replaces edge_index.
The surrounding pre-norm / residual / FFN / out_proj wrappers are
adjacency-independent and are covered by test_gnn_correctness.py
(full layer == independent hand reference).

Skipped automatically if torch_geometric is not installed.

Run:  pytest tests/test_gnn_pyg_equivalence.py -v
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pyg = pytest.importorskip("torch_geometric")
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv  # noqa: E402

from tab_ddpm.gnn_layers import GCNLayer, GATv2Layer, GINLayer, _sym_normalize  # noqa: E402

torch.set_default_dtype(torch.float64)  # tight tolerances for equivalence


def _edge_index_from_dense(A: torch.Tensor) -> torch.Tensor:
    """Dense (N,N) binary A -> edge_index. A[i,j]=1 means j->i (src=j, dst=i)."""
    rows, cols = A.nonzero(as_tuple=True)      # (i, j) pairs where A[i,j]=1
    return torch.stack([cols, rows], dim=0)    # [src=j, dst=i]


def _small_graph(N=4, self_loops=False, seed=2):
    """Random undirected binary adjacency on N nodes."""
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(N, N, generator=g) > 0.4).double()
    A = ((A + A.T) > 0).double()               # symmetrise
    A.fill_diagonal_(1.0 if self_loops else 0.0)
    return A


# ===========================================================================
# GCN  ==  GCNConv
# ===========================================================================

def test_gcn_core_equals_pyg_gcnconv():
    D, N = 8, 5
    layer = GCNLayer(D)                        # provides the real weight matrix W
    A = _small_graph(N, self_loops=False)      # GCNConv adds self-loops itself
    ei = _edge_index_from_dense(A)

    g = torch.Generator().manual_seed(7)
    x = torch.randn(N, D, generator=g)

    # PyG reference: symmetric-normalised GCN, self-loops added internally.
    conv = GCNConv(D, D, bias=False, add_self_loops=True, normalize=True).double()
    with torch.no_grad():
        conv.lin.weight.copy_(layer.W.weight)
    ref = conv(x, ei)

    # Custom dense core:  D^{-1/2}(A+I)D^{-1/2} @ (X W^T).  _sym_normalize is the
    # REAL module function; it re-adds self-loops with weight 1 (post B5 fix),
    # matching GCNConv's add_self_loops.
    a_hat = _sym_normalize(A)                   # (N, N)
    msg = F.linear(x, layer.W.weight)           # X W^T, no bias (bias=False in ref)
    mine = a_hat @ msg

    assert torch.allclose(mine, ref, atol=1e-10), (mine - ref).abs().max()


# ===========================================================================
# GIN  ==  GINConv
# ===========================================================================

def test_gin_core_equals_pyg_ginconv():
    D, N = 8, 5
    layer = GINLayer(D, learn_eps=True)         # provides real mlp + eps
    with torch.no_grad():
        layer.eps.fill_(0.53)
    A = _small_graph(N, self_loops=False)       # GINConv adds NO self-loops
    ei = _edge_index_from_dense(A)

    g = torch.Generator().manual_seed(9)
    x = torch.randn(N, D, generator=g)

    # PyG reference: out = mlp( (1+eps) x + sum_{j in N(i)} x_j ).
    conv = GINConv(nn=layer.mlp, eps=float(layer.eps), train_eps=False).double()
    ref = conv(x, ei)

    # Custom dense core: weighted sum over neighbours via matmul (A binary here).
    agg = (1.0 + layer.eps) * x + A @ x         # A has zero diagonal -> pure neigh sum
    mine = layer.mlp(agg)

    assert torch.allclose(mine, ref, atol=1e-10), (mine - ref).abs().max()


# ===========================================================================
# GATv2  ==  GATv2Conv
# ===========================================================================

def _dense_gatv2_core(x, Wsrc, Wdst, Wval, att, adj, H, dh, slope=0.2):
    """Dense GATv2 attention aggregation, mirroring GATv2Layer lines 284-306.

    Convention (verified against PyG): query i uses Wsrc, key/value j use
    Wdst/Wval; a^T applied AFTER LeakyReLU; log(adj) additive bias pre-softmax.
    x: (B,N,D); returns (B,N,D=H*dh) pre-out_proj.
    """
    B, N, D = x.shape
    Whs = (x @ Wsrc.T).view(B, N, H, dh)
    Whd = (x @ Wdst.T).view(B, N, H, dh)
    V   = (x @ Wval.T).view(B, N, H, dh)
    src = Whs.permute(0, 2, 1, 3).unsqueeze(-2)     # (B,H,N,1,dh) query i
    dst = Whd.permute(0, 2, 1, 3).unsqueeze(-3)     # (B,H,1,N,dh) key j
    pair = F.leaky_relu(src + dst, slope)
    scores = (pair * att.view(1, H, 1, 1, dh)).sum(-1)          # (B,H,N,N)
    scores = scores + torch.log(adj.unsqueeze(1).clamp(min=1e-12))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V.permute(0, 2, 1, 3))            # (B,H,N,dh)
    return out.permute(0, 2, 1, 3).reshape(B, N, D)


@pytest.mark.parametrize("H", [1, 2, 4])
def test_gatv2_core_equals_pyg_gatv2conv(H):
    D, N = 8, 5
    dh = D // H
    A = _small_graph(N, self_loops=True)        # self-loops as explicit edges
    ei = _edge_index_from_dense(A)

    g = torch.Generator().manual_seed(11)
    x = torch.randn(N, D, generator=g)

    # PyG reference. add_self_loops=False: our A already contains them.
    conv = GATv2Conv(D, dh, heads=H, bias=False, add_self_loops=False,
                     share_weights=False, negative_slope=0.2).double()
    ref = conv(x, ei)                           # (N, H*dh)

    # Map PyG weights into the custom dense formula:
    #   Wsrc <- lin_r (query),  Wdst = Wval <- lin_l (key/value),  att <- conv.att
    Wsrc = conv.lin_r.weight.detach()
    Wdst = conv.lin_l.weight.detach()
    Wval = conv.lin_l.weight.detach()
    att = conv.att.detach().reshape(H, dh)

    mine = _dense_gatv2_core(x.unsqueeze(0), Wsrc, Wdst, Wval, att,
                             A.unsqueeze(0), H, dh)[0]

    assert torch.allclose(mine, ref, atol=1e-9), (mine - ref).abs().max()


def test_gatv2_layer_uses_same_dense_core():
    """Sanity: the real GATv2Layer's internal aggregation equals _dense_gatv2_core
    with the layer's own weights (ties the PyG-verified core to the actual layer).
    """
    D, N, H = 8, 5, 2
    dh = D // H
    layer = GATv2Layer(D, n_heads=H, dropout=0.0).eval()
    A = _small_graph(N, self_loops=True)

    g = torch.Generator().manual_seed(13)
    x = torch.randn(1, N, D, generator=g)

    h = layer.norm(x)                           # layer aggregates on normed input
    core = _dense_gatv2_core(h, layer.W_src.weight, layer.W_dst.weight,
                             layer.W_val.weight, layer.a, A.unsqueeze(0), H, dh)
    # Reproduce the layer's post-core wrapper to reach the full output.
    out_wrapped = layer.out_proj(core)
    xr = x + out_wrapped
    expected = xr + layer.ffn.ff(layer.ffn.norm(xr))

    got = layer(x, A.unsqueeze(0))
    assert torch.allclose(got, expected, atol=1e-10)
