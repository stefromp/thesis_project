"""
Differentiability of the neighbour-selection rules (fixed top-k, adaptive_jenks,
adaptive_entmax15) inside the Route A model: AttnGNN V2 as TabDiff's denoiser,
trained by TabDiff's own loss (UnifiedCtimeDiffusion.mixed_loss, weighted as
Trainer.train_step does: d_lambda * dloss + c_lambda * closs).

    python -m pytest -q tests/test_selection_differentiability.py

CPU only, no data files: a small synthetic mixed-type batch with the node count
and GNN configuration (gnn type, layers, heads, top_k, d_model 256) of three
Route A datasets (magic: GATv2 k=3, shoppers: GCN k=5, beijing: GATv2 k=5).
Needs torch, numpy, tqdm (TabDiff's model modules) and, for entmax-1.5, the
entmax package (skips without it).

Checked per rule, for every block's DynamicAdjacency:
  * the selection is a discrete mask: off-diagonal weights are exactly 0 off the
    mask and exactly sigmoid(score) on it, diagonal exactly 1; the mask is the
    rule's own (top-k indices / an independent brute-force Jenks split /
    entmax-1.5 support)
  * after backward of the TabDiff loss: dL/dscore is exactly 0 off the selected
    edges (dropped edges and the diagonal) and non-zero on them; W_Q and W_K get
    finite, non-zero gradients; every other denoiser parameter gets a finite
    gradient, every sub-module a non-zero one
  * entmax-1.5 probabilities are used only through their support (p > 0):
    rescaling them leaves the model output bit-identical
  * every .detach() / torch.no_grad() on the forward path is a known one
    (logging, the rule's integer budget / support, the unused per_column noise
    path); a new one fails the audit test

The raw scores are captured by wrapping torch.bmm for tab_ddpm/graph_builder.py
only (DynamicAdjacency.forward calls it once, for Q K^T); nothing in the model
is changed.
"""
from __future__ import annotations

import os
import re
import sys
import tomllib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tabdiff_gnn"))
import config as C  # noqa: E402

RULES = ("fixed", "adaptive_jenks", "adaptive_entmax15")
# dataset -> (d_numerical, n_cat) of TabDiff's typing (hpc_transfer/jobs_tabdiff_gnn/params.csv)
SHAPES = {"magic": (10, 1), "shoppers": (10, 8), "beijing": (7, 5)}
B = 64


def _need(mode):
    if mode == "adaptive_entmax15":
        pytest.importorskip("entmax")


def _raw_config():
    with open(os.path.join(REPO, C.TABDIFF_REL, "tabdiff", "configs", "tabdiff_configs.toml"), "rb") as fh:
        return tomllib.load(fh)


def _build(ds, mode, seed=0):
    from denoiser import build_denoiser                  # puts TabDiff on sys.path
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model
    import numpy as np
    d_num, n_cat = SHAPES[ds]
    cats = np.array([2 + j % 4 for j in range(n_cat)])  # K_i (without the [MASK] class)
    raw = _raw_config()
    up = dict(raw["unimodmlp_params"], d_numerical=d_num, categories=(cats + 1).tolist())
    g, _ = C.gnn_config(ds)
    assert g["d_model"] == C.D_MODEL
    torch.manual_seed(seed)
    backbone = build_denoiser("attngnn", up, {**g, **C.GNN_FIXED, **C.gnn_rule_fields(mode)}, dim_t=C.DIM_T)
    model = Model(backbone, **raw["diffusion_params"]["edm_params"])
    dp = dict(raw["diffusion_params"], scheduler="power_mean_per_column",   # as entry.build / main.py:212-214
              cat_scheduler="log_linear_per_column")
    diffusion = UnifiedCtimeDiffusion(num_classes=cats, num_numerical_features=d_num, denoise_fn=model,
                                      y_only_model=None, **dp, device="cpu")
    diffusion.train()
    gen = torch.Generator().manual_seed(seed + 1)
    x_num = torch.randn(B, d_num, generator=gen)
    x_cat = torch.stack([torch.randint(0, int(k), (B,), generator=gen) for k in cats], 1).float()
    x = torch.cat([x_num, x_cat], 1)
    return diffusion, backbone, g, x, raw["train"]["main"]


class _Recorder:
    """torch for graph_builder only: torch.bmm outputs (= Q K^T) are kept with retain_grad."""

    def __init__(self):
        self.out = []

    def __getattr__(self, name):
        attr = getattr(torch, name)
        if name != "bmm":
            return attr

        def bmm(*a, **k):
            o = attr(*a, **k)
            if o.requires_grad:
                o.retain_grad()
            self.out.append(o)
            return o
        return bmm


def _instrument(backbone, monkeypatch):
    import tab_ddpm.graph_builder as GB
    rec = _Recorder()
    monkeypatch.setattr(GB, "torch", rec)
    caught = []
    for i, blk in enumerate(backbone.blocks):
        def hook(mod, inp, adj, i=i):
            adj.retain_grad()
            caught.append({"block": i, "qk": rec.out[-1], "adj": adj, "mod": mod})
        blk.dynamic_adj.register_forward_hook(hook)
    return caught


def _loss(diffusion, x, train_cfg):
    dloss, closs = diffusion.mixed_loss(x)
    return train_cfg["d_lambda"] * dloss + train_cfg["c_lambda"] * closs     # Trainer.train_step


def _jenks_k(row):
    """Independent brute-force plain 2-class Jenks: size of the strong class of the
    descending-sorted row, minimising the within-class sum of squares."""
    s = sorted(row, reverse=True)
    n = len(s)

    def ss(v):
        m = sum(v) / len(v)
        return sum((a - m) ** 2 for a in v)
    costs = [ss(s[:m]) + ss(s[m:]) for m in range(1, n)]
    return 1 + min(range(len(costs)), key=costs.__getitem__)


@pytest.mark.parametrize("ds", list(SHAPES))
@pytest.mark.parametrize("mode", RULES)
def test_selection_is_a_mask_and_gradients_flow_through_kept_edges(ds, mode, monkeypatch):
    _need(mode)
    diffusion, backbone, g, x, train_cfg = _build(ds, mode)
    caught = _instrument(backbone, monkeypatch)
    q_xt, noised = diffusion.q_xt, {}

    def keep_xt(*a, **k):
        out = q_xt(*a, **k)
        noised["x_cat_t"] = out[0]
        return out
    monkeypatch.setattr(diffusion, "q_xt", keep_xt)
    torch.manual_seed(2)
    loss = _loss(diffusion, x, train_cfg)
    assert torch.isfinite(loss)
    assert len(caught) == g["n_layers"]                    # one adjacency per block, one forward
    for c in caught:
        mod, adj = c["mod"], c["adj"].detach()
        assert mod.topk_mode == mode
        Bc, N, _ = adj.shape
        assert Bc == B and N == sum(SHAPES[ds])
        scores = c["qk"].detach() / mod.scale              # exactly as DynamicAdjacency.forward
        eye = torch.eye(N, dtype=torch.bool).expand(Bc, N, N)
        kept = (adj > 0) & ~eye
        # discrete mask: 0 off the mask, sigmoid(score) on it, 1 on the diagonal
        assert torch.equal(adj[eye], torch.ones_like(adj[eye]))
        assert torch.equal(adj[~kept & ~eye], torch.zeros_like(adj[~kept & ~eye]))
        assert torch.equal(adj[kept], torch.sigmoid(scores)[kept])
        k = kept.sum(-1)
        off = scores.masked_fill(eye, float("-inf"))
        if mode == "fixed":
            assert 0 < g["top_k"] < N - 1 and bool((k == g["top_k"]).all())
            want = torch.zeros_like(kept).scatter_(-1, off.topk(g["top_k"], dim=-1).indices, True)
            assert torch.equal(kept, want)
        elif mode == "adaptive_jenks":
            assert int(k.min()) >= 1 and int(k.max()) <= N - 2
            for b in range(4):
                for i in range(N):
                    row = [float(off[b, i, j]) for j in range(N) if j != i]
                    assert int(k[b, i]) == _jenks_k(row), (b, i)
        else:
            from entmax import entmax15
            assert int(k.min()) >= 1 and int(k.max()) <= N - 1
            assert torch.equal(kept, entmax15(off, dim=-1) > 0)
    loss.backward()
    # TabDiff's masked (absorbing-state) loss scores a categorical cell only where it
    # is masked at the sampled t, so an unmasked categorical node has no loss term of
    # its own; every numerical node and every masked categorical node does.
    d_num = SHAPES[ds][0]
    unmasked = noised["x_cat_t"] != torch.as_tensor(diffusion.num_classes)       # [MASK] index = K_i
    has_loss = torch.cat([torch.ones(B, d_num, dtype=torch.bool), ~unmasked], 1)  # (B, N), per node
    for c in caught:
        grad = c["qk"].grad
        assert grad is not None and torch.isfinite(grad).all()
        N = grad.shape[-1]
        eye = torch.eye(N, dtype=torch.bool).expand_as(grad)
        kept = (c["adj"].detach() > 0) & ~eye
        assert float(grad[~kept].abs().max()) == 0.0, "gradient on a dropped edge or the diagonal"
        into_loss = kept & has_loss[:, :, None]           # row i = edges into node i
        assert bool((grad[into_loss] != 0).all()), "a kept edge into a node with a loss term has no gradient"
        assert float((grad[kept] != 0).float().mean()) > 0.5
        for proj in (c["mod"].q_proj, c["mod"].k_proj):    # W_Q, W_K
            assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all()
            assert float(proj.weight.grad.abs().sum()) > 0
    for name, p in backbone.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for name, child in backbone.named_children():
        assert sum(float(p.grad.abs().sum()) for p in child.parameters()) > 0 or not list(child.parameters()), name
    for name, p in diffusion.named_parameters():           # incl. TabDiff's learnable schedules
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_entmax_probabilities_are_used_only_through_their_support(monkeypatch):
    pytest.importorskip("entmax")
    import entmax
    diffusion, backbone, _, x, _ = _build("beijing", "adaptive_entmax15")
    from tabdiff.modules.main_modules import Model  # noqa: F401  (TabDiff on sys.path)
    d_num = SHAPES["beijing"][0]
    x_num = x[:, :d_num]
    oh = torch.cat([torch.nn.functional.one_hot(x[:, d_num + j].long(), int(k) + 1).float()
                    for j, k in enumerate(diffusion.num_classes)], 1)
    c_noise = torch.log(torch.rand(B) * 80 + 0.002) / 4
    out_a = backbone(x_num, oh, c_noise)
    orig = entmax.entmax15
    monkeypatch.setattr(entmax, "entmax15", lambda z, dim=-1: 7.0 * orig(z, dim=dim))   # same support, other values
    out_b = backbone(x_num, oh, c_noise)
    assert torch.equal(out_a[0], out_b[0]) and torch.equal(out_a[1], out_b[1])


# Every .detach() / no_grad on the forward path of the Route A denoiser, with the
# reason it does not block a gradient that should flow. file -> [(line fragment, reason)]
GRAD_STOPS = {
    "adaptive_adjacency.py": [
        ("@torch.no_grad()", "jenks_budget: integer budget k only; the mask is boolean anyway"),
        ("jenks_budget(s.detach()", "same: k is not differentiable"),
        ("@torch.no_grad()", "entmax_support: boolean support only"),
        ("phi.detach().masked_fill", "same: the support is not differentiable; weights = sigmoid(phi) outside"),
    ],
    "tab_ddpm/graph_builder.py": [
        ('stats["k_per_sample"].detach()', "_log_topk: logging"),
        ("(mu_strong <= 0).float().mean(dim=-1).detach()", "_log_topk: logging"),
        ("timesteps.detach().float()", "_log_topk: logging (train)"),
        ("timesteps.detach().float().reshape(-1)[0]", "_log_topk: logging (sample)"),
    ],
    "tab_ddpm/graph_denoiser.py": [],
    "tab_ddpm/gnn_layers.py": [],
    "tabdiff_gnn/denoiser.py": [
        ("with torch.no_grad():", "node_noise_levels: node_noise='per_column' only (off in every run)"),
        ("(sig.log() / 4).detach()", "same"),
    ],
    "tabdiff_gnn/topk_log.py": [
        ("with torch.no_grad():", "KStatsLogger: observation only"),
        ("c_noise.detach()", "KStatsLogger: observation only"),
        ("with torch.no_grad():", "KStatsLogger: observation only"),
    ],
}
STOP_RE = re.compile(r"\.detach\(\)|torch\.no_grad\(\)|inference_mode|set_grad_enabled|requires_grad_\(False\)")


def test_every_gradient_stop_on_the_forward_path_is_known():
    for rel, allowed in GRAD_STOPS.items():
        lines = [ln.strip() for ln in open(os.path.join(REPO, rel)) if STOP_RE.search(ln)]
        lines = [ln for ln in lines if not ln.startswith("#") and not ln.startswith('"')]
        unmatched = list(lines)
        for frag, _ in allowed:
            hit = next((ln for ln in unmatched if frag in ln), None)
            assert hit is not None, f"{rel}: listed gradient stop {frag!r} not found (update GRAD_STOPS)"
            unmatched.remove(hit)
        assert not unmatched, f"{rel}: unlisted gradient stop(s) on the forward path: {unmatched}"
