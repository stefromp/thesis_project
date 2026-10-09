"""
The denoiser switch of Route A.

    build_denoiser("transformer", ...)  -> TabDiff's UniModMLP, built exactly as
                                           tabdiff/main.py:185 builds it
    build_denoiser("attngnn", ...)      -> AttnGNNDenoiser (V2, below)

Both have TabDiff's denoiser contract (UniModMLP.forward, main_modules.py:90),
so tabdiff.modules.main_modules.Model / Precond wrap either unchanged:

    forward(x_num  (B, d_num)  float32, already c_in-scaled by Precond,
            x_cat  (B, sum(K_i + 1)) one-hot incl. the [MASK] slot,
            timesteps (B,) = c_noise = log(sigma_cond) / 4, one per row)
      -> (F_x (B, d_num) float32 -- before EDM preconditioning,
          logits (B, sum(K_i + 1)) -- unnormalised, mask slot included)

AttnGNNDenoiser (V2): every cross-column interaction goes through the learned
sparse graph.

    TabDiff Tokenizer(d_numerical, categories, d_model)   per-column encoder
      (+ its [CLS] token dropped, as UniModMLP does)
    + noise embedding added to every node                 TabDiff's
      PositionalEmbedding(dim_t) with the same sin/cos swap as MLPDiffusion,
      then Linear(dim_t, dim_t) - SiLU - Linear(dim_t, d_model)
        node_noise="shared"     one embedding of the row's c_noise, broadcast
                                to all nodes -- exactly the information
                                TabDiff's denoiser receives (default)
        node_noise="per_column" each node embeds ITS OWN noise level: numerical
                                column j log(sigma_j(t))/4 from the live learnable
                                schedule, categorical column j log(sigma_cat_j(t))/4;
                                schedules read (detached) through a weakref to the
                                diffusion, so EMA evaluation sees the EMA schedule
    -> n_layers x tab_ddpm.graph_denoiser.AttnGNNBlock    imported unchanged:
       (dynamic DynamicAdjacency per block, fixed top-k,   2 GNN layers per block
        use_attention=False)
    -> TabDiff Reconstructor(d_numerical, categories, d_model)   per-column heads

Reserved hooks (accepted, default off, not implemented): geo_loss (GATD-style
pairwise angle/length supervision) and geo_edge_bias (geometric term in the
edge score). Enabling either raises NotImplementedError.
"""
from __future__ import annotations

import math
import os
import sys
import weakref
from typing import Optional

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TABDIFF_ROOT = os.path.join(REPO, "third_party", "TabDiff")
for _p in (TABDIFF_ROOT, REPO):          # TabDiff's packages, then ours (tab_ddpm, adaptive_adjacency)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tabdiff.modules.main_modules import PositionalEmbedding, UniModMLP  # noqa: E402
from tabdiff.modules.transformer import Reconstructor, Tokenizer  # noqa: E402
from tab_ddpm.graph_denoiser import AttnGNNBlock  # noqa: E402

DENOISERS = ("transformer", "attngnn")
NODE_NOISE = ("shared", "per_column")

# Precond's sigma_cond schedule (main_modules.py:120): the fixed rho=7 EDM
# schedule from which c_noise is computed when the schedule is learnable.
_SMIN, _SMAX, _RHO = 0.002, 80.0, 7.0


def t_from_c_noise(c_noise: torch.Tensor) -> torch.Tensor:
    """Invert Precond: c_noise = log(sigma_cond)/4, sigma_cond = rho-7 schedule(t)."""
    sigma = torch.exp(4.0 * c_noise)
    a, b = _SMIN ** (1 / _RHO), _SMAX ** (1 / _RHO)
    return ((sigma.pow(1 / _RHO) - a) / (b - a)).clamp(0.0, 1.0)


class AttnGNNDenoiser(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories,                 # K_i + 1 per categorical column (mask included), as UniModMLP
        d_model: int,
        n_layers: int,
        gnn_type: str,
        n_heads: int,
        top_k: int,
        dim_t: int = 128,
        graph_mode: str = "dynamic",
        topk_mode: str = "fixed",
        use_attention: bool = False,
        dropout: float = 0.0,
        node_noise: str = "shared",
        geo_loss: bool = False,
        geo_edge_bias: bool = False,
        bias: bool = True,
        k_min: int = 1,
        topk_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        if node_noise not in NODE_NOISE:
            raise ValueError(f"node_noise must be one of {NODE_NOISE}, got {node_noise!r}")
        if geo_loss or geo_edge_bias:
            raise NotImplementedError("geo_loss / geo_edge_bias are reserved hooks (not implemented yet)")
        self.d_numerical = d_numerical
        self.categories = list(categories)
        self.n_nodes = d_numerical + len(self.categories)
        self.d_model, self.dim_t, self.node_noise = d_model, dim_t, node_noise
        self.topk_mode = topk_mode
        self.k_logger = None        # topk_log.KStatsLogger (adaptive arms), observation only
        self.geo_loss, self.geo_edge_bias = geo_loss, geo_edge_bias
        if self.n_nodes == 0:
            raise ValueError("AttnGNNDenoiser needs at least one column")

        self.tokenizer = Tokenizer(d_numerical, self.categories, d_model, bias=bias)
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, d_model))
        self.blocks = nn.ModuleList([
            AttnGNNBlock(gnn_type, d_model=d_model, n_heads=n_heads, dropout=dropout,
                         graph_mode=graph_mode, n_nodes=self.n_nodes, top_k=top_k,
                         use_attention=use_attention, topk_mode=topk_mode,
                         **({} if topk_mode == "fixed" else
                            {"k_min": k_min, "topk_warmup_steps": topk_warmup_steps}))
            for _ in range(n_layers)
        ])
        self.detokenizer = Reconstructor(d_numerical, self.categories, d_model)
        self._diffusion_ref = None          # weakref; deepcopy keeps the same referent

    # -- per-column noise ---------------------------------------------------
    def attach_diffusion(self, diffusion) -> None:
        """Needed only for node_noise='per_column': the live schedules are read
        from the diffusion at call time (Trainer.to_ema_model swaps them there)."""
        self._diffusion_ref = weakref.ref(diffusion)

    def _embed(self, c: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(c)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)   # swap sin/cos (MLPDiffusion)
        return self.time_embed(emb)

    def node_noise_levels(self, c_noise: torch.Tensor) -> torch.Tensor:
        """(B, N) per-node conditioning values log(sigma_node)/4."""
        diffusion = self._diffusion_ref() if self._diffusion_ref is not None else None
        if diffusion is None:
            raise RuntimeError("node_noise='per_column' needs attach_diffusion(diffusion) first")
        with torch.no_grad():
            t = t_from_c_noise(c_noise.float())[:, None]            # (B, 1)
            parts = []
            if self.d_numerical:
                parts.append(diffusion.num_schedule.total_noise(t).expand(-1, self.d_numerical))
            if self.categories:
                parts.append(diffusion.cat_schedule.total_noise(t).expand(-1, len(self.categories)))
            sig = torch.cat(parts, dim=1).clamp_min(1e-12)
        return (sig.log() / 4).detach()

    # -- forward ------------------------------------------------------------
    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)[:, 1:, :]                  # drop [CLS], (B, N, d)
        B, N, _ = e.shape
        if self.node_noise == "shared":
            h = e + self._embed(timesteps)[:, None, :]
        else:
            lv = self.node_noise_levels(timesteps)                   # (B, N)
            h = e + self._embed(lv.reshape(-1)).reshape(B, N, -1)
        if self.k_logger is not None:
            self.k_logger.begin(timesteps, self.training)
        # DynamicAdjacency reads `timesteps` only for its own per-t logging, which assumes
        # integer t in [0, T); TabDiff passes c_noise, so the adaptive arms hand it None
        # (topk_log.KStatsLogger logs k against t in [0, 1] instead). Fixed: unchanged call.
        block_t = timesteps if self.topk_mode == "fixed" else None
        for i, block in enumerate(self.blocks):
            h = block(h, timesteps=block_t, block_idx=i)
        x_num_pred, x_cat_pred = self.detokenizer(h)
        x_cat_pred = (torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred) > 0
                      else torch.zeros_like(x_cat).to(x_num_pred.dtype))
        return x_num_pred, x_cat_pred


def build_denoiser(denoiser: str, unimodmlp_params: dict, gnn: Optional[dict] = None,
                   dim_t: int = 128, node_noise: str = "shared",
                   geo_loss: bool = False, geo_edge_bias: bool = False) -> nn.Module:
    """unimodmlp_params = raw_config['unimodmlp_params'] after main.py:158-159
    (d_numerical, categories incl. the mask class, + the toml's keys)."""
    if denoiser == "transformer":
        if geo_loss or geo_edge_bias or node_noise != "shared":
            raise ValueError("transformer arm is TabDiff's UniModMLP unchanged: no hooks / node_noise")
        return UniModMLP(**unimodmlp_params)
    if denoiser != "attngnn":
        raise ValueError(f"denoiser must be one of {DENOISERS}, got {denoiser!r}")
    if gnn is None:
        raise ValueError("attngnn needs the GNN config")
    return AttnGNNDenoiser(
        unimodmlp_params["d_numerical"], unimodmlp_params["categories"],
        d_model=gnn["d_model"], n_layers=gnn["n_layers"], gnn_type=gnn["gnn_type"],
        n_heads=gnn["n_heads"], top_k=gnn["top_k"], dim_t=dim_t,
        graph_mode=gnn.get("graph_mode", "dynamic"), topk_mode=gnn.get("topk_mode", "fixed"),
        use_attention=gnn.get("use_attention", False), dropout=gnn.get("dropout", 0.0),
        node_noise=node_noise, geo_loss=geo_loss, geo_edge_bias=geo_edge_bias,
        bias=unimodmlp_params.get("bias", True),
        k_min=gnn.get("k_min", 1), topk_warmup_steps=gnn.get("topk_warmup_steps", 0))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_breakdown(module: nn.Module) -> dict:
    # UniModMLP also registers its five blocks a second time as `model`
    # (main_modules.py:88); skip that alias so the blocks sum to the total.
    out = {name: count_params(child) for name, child in module.named_children() if name != "model"}
    out["total"] = count_params(module)
    return out
