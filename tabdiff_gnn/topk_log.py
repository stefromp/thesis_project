"""
Per-node selected-neighbour statistics for the adaptive top-k rules inside
TabDiff (Route A). Observation only: it never changes the computation.

TabDiff hands the denoiser c_noise = log(sigma_cond)/4, a continuous noise
level, not an integer timestep, so DynamicAdjacency's own per-t logging (which
buckets integer t in [0, T)) does not apply; for the adaptive arms the
denoiser passes timesteps=None to it (that argument is read only by that
logging). Instead this logger recovers the diffusion time t in [0, 1] from
c_noise (denoiser.t_from_c_noise, the inverse of Precond's rho-7 schedule) and
records, per (phase, block, t bin, node):

    k = number of kept off-diagonal edges of node i in that block's adjacency
        (entries with weight > 0; the masked ones are exactly sigmoid(-1e9) = 0)

as additive accumulators (count, sum, sum of squares, min, max) plus a
histogram of k per (phase, block). Forward hooks on each block's
DynamicAdjacency read its output; accumulation runs under no_grad, uses no
randomness and never touches the model's tensors, so outputs are identical
with logging on or off (tests/test_tabdiff_gnn.py).

Phases: "train" = training-mode forwards, every `log_every`-th one (the thesis
runs' TOPK_LOG_EVERY = 100); eval-mode forwards are logged only when
`eval_phase` is set (None in the training process, where eval-mode forwards
are TabDiff's per-epoch EMA-loss pass). `force_phase` labels EVERY forward
with that phase whatever the module mode: the sampling process uses
force_phase="sample", because entry.cmd_sample calls Trainer.sample_synthetic
without TabDiff's diffusion.eval() (evaluate_generation calls it), so the
denoiser samples in training mode. That does not change any output -- dropout
is 0, the fixed path never reads self.training and with topk_warmup_steps 0
neither rule does (tested) -- but it would mislabel the phase.

Files in the run dir (merged on every write, so the resumable news chain and
a resubmitted job accumulate instead of overwriting):
    topk_k_stats.csv  phase, block, t_bin, t_lo, t_hi, node, column, n, k_sum, k_sumsq, k_min, k_max, k_mean
    topk_k_hist.csv   phase, block, k, count
"""
from __future__ import annotations

import csv
import os

import torch

N_BINS = 10
STATS_CSV = "topk_k_stats.csv"
HIST_CSV = "topk_k_hist.csv"


class KStatsLogger:
    def __init__(self, n_nodes: int, n_blocks: int, column_names=None, log_every: int = 100,
                 eval_phase: str | None = None, force_phase: str | None = None):
        self.n_nodes, self.n_blocks = n_nodes, n_blocks
        self.columns = list(column_names) if column_names else [f"node{i}" for i in range(n_nodes)]
        self.log_every = max(1, int(log_every))
        self.eval_phase = eval_phase
        self.force_phase = force_phase
        self._acc = {}            # (phase, block) -> dict of tensors
        self._t_bin = None
        self._phase = None
        self._train_calls = 0
        self._handles = []

    # deepcopy (TabDiff's EMA copy of the denoiser) must share, not clone, the logger
    def __deepcopy__(self, memo):
        return self

    # -- wiring --------------------------------------------------------------
    def attach(self, denoiser) -> "KStatsLogger":
        denoiser.k_logger = self
        for b, block in enumerate(denoiser.blocks):
            if block.dynamic_adj is None:
                continue
            self._handles.append(block.dynamic_adj.register_forward_hook(self._hook(b)))
        return self

    def begin(self, c_noise: torch.Tensor, training: bool) -> None:
        """Called by the denoiser at the start of each forward (before the blocks)."""
        from denoiser import t_from_c_noise
        if self.force_phase is not None:
            self._phase = self.force_phase
        elif training:
            self._train_calls += 1
            self._phase = "train" if (self._train_calls - 1) % self.log_every == 0 else None
        else:
            self._phase = self.eval_phase
        if self._phase is None:
            self._t_bin = None
            return
        with torch.no_grad():
            t = t_from_c_noise(c_noise.detach().float().reshape(-1))
            self._t_bin = (t * N_BINS).long().clamp_(0, N_BINS - 1)

    def _hook(self, block: int):
        def hook(module, inputs, A):
            if self._phase is None or self._t_bin is None:
                return None
            with torch.no_grad():
                Bn, N, _ = A.shape
                eye = torch.eye(N, dtype=torch.bool, device=A.device)
                k = ((A > 0) & ~eye).sum(-1).to(torch.float64)            # (B, N)
                tb = self._t_bin.to(A.device)
                if tb.numel() == 1 and Bn > 1:
                    tb = tb.expand(Bn)
                key = (self._phase, block)
                acc = self._acc.get(key)
                if acc is None:
                    dev = A.device
                    acc = {"n": torch.zeros(N_BINS, N, dtype=torch.float64, device=dev),
                           "s": torch.zeros(N_BINS, N, dtype=torch.float64, device=dev),
                           "ss": torch.zeros(N_BINS, N, dtype=torch.float64, device=dev),
                           "mn": torch.full((N_BINS, N), float("inf"), dtype=torch.float64, device=dev),
                           "mx": torch.full((N_BINS, N), float("-inf"), dtype=torch.float64, device=dev),
                           "hist": torch.zeros(N, dtype=torch.float64, device=dev)}     # k in [0, N-1]
                    self._acc[key] = acc
                idx = tb.view(-1, 1).expand(Bn, N)
                acc["n"].scatter_add_(0, idx, torch.ones_like(k))
                acc["s"].scatter_add_(0, idx, k)
                acc["ss"].scatter_add_(0, idx, k * k)
                acc["mn"].scatter_reduce_(0, idx, k, reduce="amin")
                acc["mx"].scatter_reduce_(0, idx, k, reduce="amax")
                acc["hist"] += torch.bincount(k.reshape(-1).long(), minlength=N)[:N].to(torch.float64)
            return None
        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- output --------------------------------------------------------------
    def rows(self):
        stats, hist = [], []
        for (phase, block), acc in sorted(self._acc.items()):
            a = {k: v.cpu() for k, v in acc.items()}
            for tb in range(N_BINS):
                for i in range(self.n_nodes):
                    n = float(a["n"][tb, i])
                    if n == 0:
                        continue
                    stats.append({"phase": phase, "block": block, "t_bin": tb, "t_lo": tb / N_BINS,
                                  "t_hi": (tb + 1) / N_BINS, "node": i, "column": self.columns[i], "n": n,
                                  "k_sum": float(a["s"][tb, i]), "k_sumsq": float(a["ss"][tb, i]),
                                  "k_min": float(a["mn"][tb, i]), "k_max": float(a["mx"][tb, i])})
            for kk in range(self.n_nodes):
                if a["hist"][kk] > 0:
                    hist.append({"phase": phase, "block": block, "k": kk, "count": float(a["hist"][kk])})
        return stats, hist

    def write(self, run_dir: str) -> None:
        """Merge this process's accumulators into the run dir's CSVs (additive)."""
        stats, hist = self.rows()
        sp, hp = os.path.join(run_dir, STATS_CSV), os.path.join(run_dir, HIST_CSV)
        merged = {}
        if os.path.exists(sp):
            with open(sp) as fh:
                for r in csv.DictReader(fh):
                    key = (r["phase"], int(r["block"]), int(r["t_bin"]), int(r["node"]))
                    merged[key] = {**r, **{k: float(r[k]) for k in ("n", "k_sum", "k_sumsq", "k_min", "k_max")}}
        for r in stats:
            key = (r["phase"], r["block"], r["t_bin"], r["node"])
            if key in merged:
                m = merged[key]
                m["n"] += r["n"]
                m["k_sum"] += r["k_sum"]
                m["k_sumsq"] += r["k_sumsq"]
                m["k_min"] = min(m["k_min"], r["k_min"])
                m["k_max"] = max(m["k_max"], r["k_max"])
            else:
                merged[key] = dict(r)
        cols = ["phase", "block", "t_bin", "t_lo", "t_hi", "node", "column", "n", "k_sum", "k_sumsq",
                "k_min", "k_max", "k_mean"]
        with open(sp + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for key in sorted(merged):
                m = merged[key]
                m["k_mean"] = m["k_sum"] / m["n"] if m["n"] else float("nan")
                w.writerow({c: m[c] for c in cols})
        os.replace(sp + ".tmp", sp)
        hm = {}
        if os.path.exists(hp):
            with open(hp) as fh:
                for r in csv.DictReader(fh):
                    hm[(r["phase"], int(r["block"]), int(r["k"]))] = float(r["count"])
        for r in hist:
            key = (r["phase"], r["block"], r["k"])
            hm[key] = hm.get(key, 0.0) + r["count"]
        with open(hp + ".tmp", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "block", "k", "count"])
            for key in sorted(hm):
                w.writerow([*key, hm[key]])
        os.replace(hp + ".tmp", hp)
        self._acc = {}                      # written: a second write must not double-count
