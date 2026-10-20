#!/usr/bin/env python3
"""
Figure 2 -- the two adaptive selection rules on a worked example.

    python3 analysis/summary/fig_selection_example.py     (needs entmax==1.3, as the model)

One node with 7 candidate neighbours A..G, raw scores s = [2.1, 1.9, 1.8, 0.3, 0.1, -0.2, -0.5];
panel (a) s, panel (b) s / 10 (same order, flatter row). The kept neighbours come from the
model's own module: tab_ddpm.graph_builder.DynamicAdjacency.forward in eval mode (the sampling
path), with W_Q = I, W_K = phi * sqrt(d) and node embeddings I, so that its score matrix is
exactly phi (row 0 = this node). Jenks: build_adjacency(use_gate_rule=False, ...) ->
adaptive_adjacency.jenks_budget; entmax-1.5: build_adjacency_entmax -> entmax_support ->
entmax.entmax15. The budgets k, tau and the probabilities p are read from those same functions
(jenks_budget; entmax15 and the package's own threshold routine), and each is checked against
the mask the module produced.

entmax-1.5 (alpha = 1.5): p_j = [s_j / 2 - tau]_+^2 with sum p = 1, so j is kept iff s_j > 2 tau;
the threshold line is drawn at 2 tau in score units.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_a_figs as F  # noqa: E402

np, plt, ct = F.np, F.plt, F.ct
sys.path.insert(0, str(F.BS.REPO))

S0 = [2.1, 1.9, 1.8, 0.3, 0.1, -0.2, -0.5]
NAMES = list("ABCDEFG")
K_MIN = 1                                   # tabdiff_gnn/config.RULE_SETTINGS["k_min"]


def _module_mask(phi, mode: str):
    """Kept neighbours of node 0 according to DynamicAdjacency.forward (eval mode)."""
    import torch
    from tab_ddpm.graph_builder import DynamicAdjacency
    n = phi.shape[-1]
    da = DynamicAdjacency(n, n, top_k=1, topk_mode=mode, k_min=K_MIN).eval()
    with torch.no_grad():
        da.q_proj.weight.copy_(torch.eye(n))
        da.k_proj.weight.copy_(phi[0] * da.scale)   # scores = I @ (phi * scale) / scale = phi
        adj = da(torch.eye(n).unsqueeze(0))
    return (adj[0, 0, 1:] > 0).numpy()


def compute(scores) -> dict:
    """Both rules on one node's off-diagonal scores (any order), via the model's functions."""
    import torch
    from adaptive_adjacency import ENTMAX_ALPHA, entmax_support, jenks_budget
    from entmax import entmax15
    from entmax.activations import _entmax_threshold_and_support

    s = torch.tensor(scores, dtype=torch.float32)
    n = len(s) + 1
    phi = torch.zeros(1, n, n)
    phi[0, 0, 1:] = s                      # node 0's row; phi[0, 0, 0] (self) is masked by the rule
    keep_j = _module_mask(phi, "adaptive_jenks")
    keep_e = _module_mask(phi, "adaptive_entmax15")
    # Jenks budget: the function build_adjacency calls, with the same arguments
    s_sorted = s.sort(descending=True).values.unsqueeze(0)
    k_j = int(jenks_budget(s_sorted, K_MIN, use_gate_rule=False, delta=None)[0][0])
    # entmax-1.5: support size from entmax_support, probabilities from entmax15 (the function it
    # calls), tau from the package's threshold routine on s / 2 (as Entmax15Function does)
    alpha = ENTMAX_ALPHA["adaptive_entmax15"]
    assert alpha == 1.5
    _, k_rows = entmax_support(phi, alpha, K_MIN)
    k_e = int(k_rows[0, 0])
    p = entmax15(s, dim=-1).numpy()
    tau = float(_entmax_threshold_and_support(s / 2, dim=-1)[0])
    s_np = s.numpy()
    # consistency of every number with the module's own mask
    assert keep_j.sum() == k_j, (keep_j, k_j)
    assert set(np.flatnonzero(keep_j)) == set(np.argsort(-s_np, kind="stable")[:k_j]), keep_j
    assert keep_e.sum() == k_e, (keep_e, k_e)
    assert np.array_equal(keep_e, p > 0), (keep_e, p)
    assert np.array_equal(keep_e, s_np > 2 * tau), (keep_e, tau)
    assert np.allclose(p, np.clip(s_np / 2 - tau, 0, None) ** 2, atol=1e-6) and abs(p.sum() - 1) < 1e-6
    return {"s": s_np, "keep_jenks": keep_j, "keep_entmax": keep_e, "k_jenks": k_j, "k_entmax": k_e,
            "tau": tau, "p": p}


def panel(ax, r: dict, title: str):
    F.style_ax(ax)
    x = np.arange(len(r["s"]))
    w = 0.38
    grey = ct.C_GRID
    for i, v in enumerate(r["s"]):
        ax.bar(i - w / 2, v, w, color=F.C_ARM["jenks"] if r["keep_jenks"][i] else grey, ec="white", lw=1)
        ax.bar(i + w / 2, v, w, color=F.C_ARM["entmax15"] if r["keep_entmax"][i] else grey, ec="white", lw=1)
        if r["p"][i] > 0:
            ax.text(i + w / 2, v + (0.03 if v >= 0 else -0.03) * np.ptp(r["s"]), f"{r['p'][i]:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, color=ct.C_TEXT2)
    ax.axhline(0, color=ct.C_TEXT2, lw=0.8)
    xs = r["k_jenks"] - 0.5
    ax.axvline(xs, color=F.C_ARM["jenks"], lw=1.5, ls="--")
    thr = 2 * r["tau"]
    ax.axhline(thr, color=F.C_ARM["entmax15"], lw=1.5, ls=":")
    lo = min(r["s"].min(), thr, 0)
    hi = max(r["s"].max(), 0)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    ax.text(xs + 0.08, hi + 1.9 * pad, "Jenks split", color=F.C_ARM["jenks"], fontsize=7.5, va="top")
    ax.text(len(x) - 0.55, thr, f"entmax-1.5 threshold 2τ = {thr:.3f}", color=F.C_ARM["entmax15"], fontsize=7.5,
            ha="right", va="bottom")
    ax.text(0.98, 0.97, f"k_Jenks = {r['k_jenks']}\nk_entmax = {r['k_entmax']}\nτ = {r['tau']:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=ct.C_TEXT,
            bbox=dict(fc="white", ec=ct.C_GRID, lw=0.8))
    ax.set_xticks(x, NAMES)
    ax.set_xlabel("candidate neighbour (sorted by score)", fontsize=8, color=ct.C_TEXT2)
    ax.set_ylabel("raw score φ", fontsize=8, color=ct.C_TEXT2)
    ax.set_title(title, loc="left", fontsize=10, color=ct.C_TEXT)


# What the figure must show (checked before drawing; a disagreement stops the script)
EXPECTED = {"a": {"k_jenks": 3, "k_entmax": 3, "tau": 0.3927, "p_kept": (0.432, 0.311, 0.257)},
            "b": {"k_jenks": 3, "k_entmax": 7, "tau": -0.3352}}


def check_expected(r: dict, e: dict, name: str):
    got = (r["k_jenks"], r["k_entmax"])
    assert got == (e["k_jenks"], e["k_entmax"]), f"panel {name}: (k_Jenks, k_entmax) = {got}, expected {e}"
    assert abs(r["tau"] - e["tau"]) < 5e-5, f"panel {name}: tau {r['tau']} != {e['tau']}"
    if "p_kept" in e:
        p = r["p"][r["p"] > 0]
        assert np.allclose(p, e["p_kept"], atol=5e-4), f"panel {name}: p {p} != {e['p_kept']}"


def main():
    a = compute(S0)
    b = compute([v / 10 for v in S0])
    check_expected(a, EXPECTED["a"], "a")
    check_expected(b, EXPECTED["b"], "b")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    panel(axes[0], a, "(a) scores s")
    panel(axes[1], b, "(b) scores s / 10 (same order, flatter)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=F.C_ARM["jenks"], label="kept by Jenks"),
               plt.Rectangle((0, 0), 1, 1, color=F.C_ARM["entmax15"], label="kept by entmax-1.5"),
               plt.Rectangle((0, 0), 1, 1, color=ct.C_GRID, label="dropped")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    F.caption(fig, "One node, 7 candidate neighbours; left bar of each pair = Jenks, right bar = entmax-1.5. "
                   "Kept sets computed by the model's DynamicAdjacency (eval mode) on these scores. Jenks: 2-class "
                   "natural-breaks split of the sorted scores (dashed line). entmax-1.5: p_j = [s_j/2 − τ]₊², kept iff "
                   "s_j > 2τ (dotted line); numbers above the bars = p_j. Dividing every score by 10 keeps the Jenks "
                   "split (it depends only on the shape of the row) but entmax keeps all 7.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    F.savefig(fig, "selection_example")
    for nm, r in (("a", a), ("b", b)):
        print(f"({nm}) k_Jenks {r['k_jenks']}, k_entmax {r['k_entmax']}, tau {r['tau']:.4f}, "
              f"p {np.round(r['p'], 4).tolist()}")
    print(f"wrote {F.BS.rel(F.FIG)}/selection_example.{{pdf,png}}")


if __name__ == "__main__":
    main()
