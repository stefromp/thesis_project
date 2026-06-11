"""Count trainable parameters of the whole TabDDPM model (GaussianMultinomial-
Diffusion wrapping a denoiser) for the MLP denoiser (ddpm_cb_best config) vs the
GNN-aware denoiser (ablation grid), on adult / california / churn2 / gesture.

The diffusion wrapper holds only non-trainable buffers (the noise schedule), so
the whole-model trainable count equals the denoiser's; this script demonstrates
that explicitly by building the full model both ways."""
import os
import sys
import itertools
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Reuse the ablation runner's zero/rtdl stubs so tab_ddpm imports cleanly.
from ablation_runner import _setup_zero_rtdl_stubs
_setup_zero_rtdl_stubs()

import torch
from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.graph_denoiser import GraphAwareDenoiser
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def n_buffers(m):
    return sum(b.numel() for b in m.buffers())


def wrap_tabddpm(denoise_fn, K, d_num, num_timesteps, scheduler):
    """Wrap a denoiser in the full TabDDPM (GaussianMultinomialDiffusion)."""
    return GaussianMultinomialDiffusion(
        num_classes=np.array(K) if len(K) else np.array([0]),
        num_numerical_features=d_num,
        denoise_fn=denoise_fn,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=torch.device("cpu"),
    )


def cat_sizes(name):
    p = os.path.join(REPO, "data", name, "X_cat_train.npy")
    if not os.path.exists(p):
        return []
    X = np.load(p, allow_pickle=True)
    return [len(set(X[:, i].tolist())) for i in range(X.shape[1])]


def num_features(name, is_y_cond, task_type):
    X = np.load(os.path.join(REPO, "data", name, "X_num_train.npy"), allow_pickle=True)
    d = X.shape[1]
    # regression + not y-cond -> label concatenated into X_num (see make_dataset)
    if task_type == "regression" and not is_y_cond:
        d += 1
    return d


# ddpm_cb_best configs for the four datasets.
DATASETS = {
    "adult": dict(num_classes=2, is_y_cond=True, task="binclass",
                  d_layers=[256, 1024, 1024, 1024, 1024, 256],
                  num_timesteps=100, scheduler="cosine"),
    "california": dict(num_classes=0, is_y_cond=False, task="regression",
                       d_layers=[512, 256, 256, 256, 256, 128],
                       num_timesteps=1000, scheduler="cosine"),
    "churn2": dict(num_classes=2, is_y_cond=True, task="binclass",
                   d_layers=[512, 1024, 1024, 1024, 1024, 512],
                   num_timesteps=100, scheduler="cosine"),
    "gesture": dict(num_classes=5, is_y_cond=True, task="multiclass",
                    d_layers=[128, 512, 512, 1024],
                    num_timesteps=1000, scheduler="cosine"),
}

# Ablation grid (from ablation_runner.py)
GNN_TYPES = ["gcn", "gatv2", "gin"]
N_LAYERS = [3, 4]
D_MODELS = [256, 512, 1024]
TOP_K = 5
N_HEADS = 4

for name, cfg in DATASETS.items():
    K = cat_sizes(name)
    d_num = num_features(name, cfg["is_y_cond"], cfg["task"])
    d_in = int(np.sum(K)) + d_num
    eff_cat = [k for k in K if k > 0]
    n_nodes = d_num + len(eff_cat)

    mlp = MLPDiffusion(
        d_in=d_in,
        num_classes=cfg["num_classes"],
        is_y_cond=cfg["is_y_cond"],
        rtdl_params=dict(d_layers=list(cfg["d_layers"]), dropout=0.0),
    )
    tabddpm_mlp = wrap_tabddpm(mlp, K, d_num, cfg["num_timesteps"], cfg["scheduler"])
    mlp_n = n_params(tabddpm_mlp)          # whole-model trainable params
    mlp_buf = n_buffers(tabddpm_mlp)        # non-trainable schedule buffers

    print("=" * 80)
    print(f"{name}  | task={cfg['task']}  d_num={d_num}  cat_sizes={K}  "
          f"d_in={d_in}  n_nodes={n_nodes}  T={cfg['num_timesteps']}")
    print(f"  TabDDPM[MLP] (ddpm_cb_best, d_layers={cfg['d_layers']})")
    print(f"     trainable params : {mlp_n:,}")
    print(f"     (+ non-trainable schedule buffers: {mlp_buf:,})")
    print("-" * 80)
    print(f"  TabDDPM[GNN-denoiser] (dynamic, top_k={TOP_K}, n_heads={N_HEADS})")
    print(f"  attn ON  = self-attention per block | attn OFF = GNN-only")
    print(f"    {'gnn_type':8s} {'layers':>6s} {'d_model':>7s} "
          f"{'attn ON':>14s} {'attn OFF':>14s} {'OFF/ON':>7s}  "
          f"{'OFFxMLP':>8s}")
    for gnn, nl, dm in itertools.product(GNN_TYPES, N_LAYERS, D_MODELS):
        def build(use_attention):
            den = GraphAwareDenoiser(
                d_in=d_in,
                num_classes=cfg["num_classes"],
                is_y_cond=cfg["is_y_cond"],
                d_num=d_num,
                cat_sizes=eff_cat,
                d_model=dm,
                n_layers=nl,
                n_heads=N_HEADS,
                graph_mode="dynamic",
                top_k=TOP_K,
                gnn_type=gnn,
                dropout=0.0,
                use_attention=use_attention,
            )
            return n_params(wrap_tabddpm(den, K, d_num,
                                         cfg["num_timesteps"], cfg["scheduler"]))
        on = build(True)
        off = build(False)
        print(f"    {gnn:8s} {nl:6d} {dm:7d} {on:14,d} {off:14,d} "
              f"{off/on:6.2f}x  {off/mlp_n:7.2f}x")
print("=" * 80)
