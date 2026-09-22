"""
Graph construction utilities for GraphAwareDenoiser.

Two modes:
  static  — edges derived once from training-data statistics (Pearson
             correlation, Cramér's V, normalised mutual information).
  dynamic — soft adjacency learned end-to-end from node embeddings via
             scaled dot-product attention (DynamicAdjacency module).
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# adaptive_adjacency.py lives at the repo root, not inside the tab_ddpm
# package. Resolve the repo root relative to this file (parent of tab_ddpm/)
# so the import works regardless of the caller's sys.path / cwd setup.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adaptive_adjacency import (  # noqa: E402
    ENTMAX_ALPHA,
    build_adjacency,
    build_adjacency_entmax,
)

from . import topk_logging

# Every value DynamicAdjacency accepts for topk_mode. Anything else raises in
# __init__ -- the forward only tests for "fixed", so an unchecked typo would
# otherwise run an adaptive rule silently.
TOPK_MODES = ("fixed", "adaptive_jenks", *ENTMAX_ALPHA)


# ---------------------------------------------------------------------------
# Static adjacency
# ---------------------------------------------------------------------------

def build_static_adjacency(
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    d_num: int,
    cat_sizes: List[int],
    threshold: float = 0.1,
) -> torch.Tensor:
    """Return a (N, N) float32 adjacency matrix derived from training data.

    N = d_num + len(cat_sizes).  One node per numerical feature and one node
    per categorical feature (regardless of cardinality).

    Edge rules (undirected, based on absolute-value scores):
      num–num : |Pearson correlation| > threshold
      cat–cat : Cramér's V            > threshold
      num–cat : normalised MI         > threshold

    Self-loops are always included.

    Args:
        X_num:     (n_samples, d_num) float array, or None when d_num == 0.
        X_cat:     (n_samples, n_cat) integer array, or None when no cats.
        d_num:     number of numerical features.
        cat_sizes: cardinalities of each categorical feature (no zeros).
        threshold: minimum score required to add an edge.

    Returns:
        adj: float32 tensor of shape (N, N) with values in {0.0, 1.0}.
    """
    n_cat = len(cat_sizes)
    N = d_num + n_cat
    adj = np.zeros((N, N), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)

    if N == 0:
        return torch.from_numpy(adj)

    # --- num–num : Pearson correlation ---
    if X_num is not None and d_num >= 2:
        corr = np.corrcoef(X_num.T)          # (d_num, d_num), may contain NaN
        corr = np.nan_to_num(corr, nan=0.0)
        for i in range(d_num):
            for j in range(i + 1, d_num):
                if abs(corr[i, j]) > threshold:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

    # --- cat–cat : Cramér's V ---
    if X_cat is not None and n_cat >= 2:
        from scipy.stats import chi2_contingency  # lazy import

        for i in range(n_cat):
            for j in range(i + 1, n_cat):
                table = _contingency_table(X_cat[:, i], X_cat[:, j])
                chi2, _, _, _ = chi2_contingency(table, correction=False)
                n_obs = table.sum()
                r, k = table.shape
                denom = n_obs * (min(r, k) - 1)
                cramer_v = float(np.sqrt(chi2 / (denom + 1e-12)))
                cramer_v = min(cramer_v, 1.0)
                if cramer_v > threshold:
                    ni, nj = d_num + i, d_num + j
                    adj[ni, nj] = 1.0
                    adj[nj, ni] = 1.0

    # --- num–cat : normalised mutual information ---
    if X_num is not None and X_cat is not None and d_num > 0 and n_cat > 0:
        from sklearn.feature_selection import mutual_info_classif  # lazy import

        for j in range(n_cat):
            mi_scores = mutual_info_classif(
                X_num, X_cat[:, j], discrete_features=False, random_state=0
            )
            _, counts = np.unique(X_cat[:, j], return_counts=True)
            probs = counts / counts.sum()
            h_cat = float(-np.sum(probs * np.log(probs + 1e-12)))
            for i in range(d_num):
                nmi = float(mi_scores[i]) / (h_cat + 1e-12)
                nmi = min(nmi, 1.0)
                if nmi > threshold:
                    nj = d_num + j
                    adj[i, nj] = 1.0
                    adj[nj, i] = 1.0

    return torch.from_numpy(adj)


def _contingency_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build an (r, k) integer contingency table for two integer arrays."""
    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    table = np.zeros((len(a_vals), len(b_vals)), dtype=np.int64)
    np.add.at(table, (a_inv, b_inv), 1)
    return table


# ---------------------------------------------------------------------------
# Dynamic adjacency
# ---------------------------------------------------------------------------

class DynamicAdjacency(nn.Module):
    """Learnable soft adjacency derived from node embeddings each forward pass.

    Uses scaled dot-product attention between node embeddings to produce a
    (B, N, N) adjacency in (0, 1).  Optionally sparsifies to the top-k
    neighbours per node (off-diagonal), keeping the result differentiable via
    a straight-through mask.  Self-loops are always set to 1.

    Args:
        n_nodes: number of graph nodes N.
        d_model: node embedding dimension.
        top_k:   if > 0, keep only the top-k off-diagonal entries per row;
                 0 means dense soft adjacency (full sigmoid). Also the k used
                 during the adaptive modes' warm-up phase (see topk_mode).
        topk_mode: one of TOPK_MODES.
                 "fixed" (default; today's behaviour, bit-identical);
                 "adaptive_jenks" (per-node budget via a 2-class Jenks split
                 of the raw scores, see adaptive_adjacency.build_adjacency);
                 "adaptive_sparsemax" / "adaptive_entmax15" (kept edges = the
                 support of sparsemax / entmax-1.5 over each row, see
                 adaptive_adjacency.build_adjacency_entmax). All three
                 adaptive modes only choose the mask; the weights are the same
                 sigmoid gates as in "fixed".
        k_min:   floor on the adaptive budget (adaptive modes only).
        topk_warmup_steps: while the number of training-mode forward calls
                 seen so far is below this, the adaptive modes use the fixed
                 top_k path instead (warm-up). Ignored in "fixed" mode.
        num_timesteps: diffusion horizon T, used only to bucket the logged
                 per-sample budgets by t (10 equal-width buckets over
                 [0, T)). Not used by the "fixed" path at all.
    """

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        top_k: int = 0,
        topk_mode: str = "fixed",
        k_min: int = 1,
        topk_warmup_steps: int = 0,
        num_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if topk_mode not in TOPK_MODES:
            raise ValueError(f"topk_mode must be one of {TOPK_MODES}, got {topk_mode!r}")
        self.n_nodes = n_nodes
        self.top_k = top_k
        self.scale = math.sqrt(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # --- adaptive top-k (Section 5.4.4) ---
        self.topk_mode = topk_mode
        self.k_min = k_min
        self.topk_warmup_steps = topk_warmup_steps
        self.num_timesteps = num_timesteps
        # Plain Python ints, not buffers: a buffer would land in state_dict
        # (breaking strict loading of pre-existing checkpoints) and reading
        # it back via .item() would force a GPU->CPU sync on every forward.
        # Warm-up only matters during a fresh training run, so neither
        # counter needs to survive checkpointing.
        self._train_step = 0         # += 1 once per forward when self.training
        self._sample_call_count = 0  # += 1 once per forward when not self.training

    def forward(
        self,
        node_emb: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute a (B, N, N) soft adjacency from node embeddings.

        Args:
            node_emb:  (B, N, d_model) node embedding tensor.
            timesteps: (B,) diffusion timestep per sample, or None. Used only
                       for adaptive-mode logging (t-bucket assignment); the
                       "fixed" path never reads it.
            block_idx: index of the owning block, for logging only.

        Returns:
            adj: (B, N, N) float tensor with values in [0, 1].
        """
        B, N, _ = node_emb.shape
        Q = self.q_proj(node_emb)                              # (B, N, D)
        K = self.k_proj(node_emb)                              # (B, N, D)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, N, N)

        if self.topk_mode == "fixed":
            # -----------------------------------------------------------
            # Unmodified original path -- byte-for-byte identical to the
            # pre-adaptive-top-k implementation, so --topk_mode fixed is
            # bit-identical to today by construction. This branch never
            # calls build_adjacency, so it can't inherit that module's
            # edge cases (e.g. k_fixed=0 raising in torch.gather).
            # -----------------------------------------------------------
            if self.top_k > 0 and self.top_k < N - 1:
                # Select top-k off-diagonal neighbours on the raw scores and
                # push the rest to -1e9 *before* sigmoid, so gradients flow
                # through the softmax-like nonlinearity for survivors.
                eye = torch.eye(N, device=scores.device, dtype=torch.bool).unsqueeze(0)
                scores_no_self = scores.masked_fill(eye, float("-inf"))
                _, topk_idx = torch.topk(scores_no_self, k=self.top_k, dim=-1)
                keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
                scores = scores.masked_fill(~keep, -1e9)

            adj = torch.sigmoid(scores)

            # Enforce self-loops
            eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = torch.clamp(adj + eye, max=1.0)
            return adj

        # -------------------------------------------------------------------
        # Adaptive modes: "adaptive_jenks", "adaptive_sparsemax",
        # "adaptive_entmax15". Warm-up, the dense fallback and logging are
        # shared; only the rule applied after warm-up differs.
        # -------------------------------------------------------------------
        train_step = self._train_step
        warmed_up = (not self.training) or (train_step >= self.topk_warmup_steps)
        if self.training:
            self._train_step += 1

        # Today's masking is only active for 0 < top_k < N-1; outside that
        # range the fixed-mode code above is already a no-op (dense). Reuse
        # that same range check for warm-up instead of handing an
        # out-of-range k_fixed to build_adjacency: k_fixed=0 raises (negative
        # gather index), and while k_fixed>=N-1 happens to reduce to dense
        # there too, relying on that coincidence is fragile.
        out_of_range = self.top_k <= 0 or self.top_k >= N - 1

        if not warmed_up and out_of_range:
            eye_bool = torch.eye(N, device=scores.device, dtype=torch.bool).unsqueeze(0)
            A = torch.sigmoid(scores)
            A = A.masked_fill(eye_bool, 1.0)
            k_dense = torch.full((B,), float(N - 1), device=scores.device, dtype=scores.dtype)
            stats = {"k_mean": k_dense.mean(), "k_per_sample": k_dense}
        elif warmed_up and self.topk_mode in ENTMAX_ALPHA:
            A, stats = build_adjacency_entmax(
                scores,
                alpha=ENTMAX_ALPHA[self.topk_mode],
                k_min=self.k_min,
            )
        else:
            # adaptive_jenks after warm-up, or any adaptive mode during it
            k_fixed = None if warmed_up else self.top_k
            A, stats = build_adjacency(
                scores,
                k_fixed=k_fixed,
                k_min=self.k_min,
                sym="none",
                use_gate_rule=True,
                delta=None,
                share_across_batch=False,
            )

        self._log_topk(stats, timesteps, block_idx, train_step)
        return A

    def _log_topk(self, stats, timesteps, block_idx, train_step: int) -> None:
        """Bucket the per-sample budget by t and append to topk_stats.csv.

        Training: each sample in the batch has its own t, so a batch-level
        mean belongs to no single bucket -- group per-sample budgets by that
        sample's own bucket and emit one row per occupied bucket. Sampling:
        every sample in a batch shares one t, so it's always a single row.
        """
        if timesteps is None or "k_per_sample" not in stats:
            return
        k_per_sample = stats["k_per_sample"].detach().float()
        t_horizon = max(self.num_timesteps, 1)

        if self.training:
            t = timesteps.detach().float()
            t_bucket = torch.clamp((t / t_horizon * 10).long(), 0, 9)
            rows = []
            for bucket in t_bucket.unique().tolist():
                in_bucket = t_bucket == bucket
                n_samples = int(in_bucket.sum().item())
                k_mean = float(k_per_sample[in_bucket].mean().item())
                rows.append((bucket, k_mean, n_samples))
            topk_logging.log_rows(step=train_step, block=block_idx, rows=rows, phase="train")
        else:
            t_val = float(timesteps.detach().float().reshape(-1)[0].item())
            t_bucket_val = max(0, min(9, int(t_val / t_horizon * 10)))
            k_mean = float(k_per_sample.mean().item())
            n_samples = int(k_per_sample.shape[0])
            step = self._sample_call_count
            self._sample_call_count += 1
            topk_logging.log_rows(
                step=step, block=block_idx,
                rows=[(t_bucket_val, k_mean, n_samples)],
                phase="sample",
            )
