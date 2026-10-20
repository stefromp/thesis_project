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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_style as S  # noqa: E402

plt = S.plt
sys.path.insert(0, str(HERE.parents[1]))         # repo root: tab_ddpm/, adaptive_adjacency.py

S0 = [2.1, 1.9, 1.8, 0.3, 0.1, -0.2, -0.5]
NAMES = list("ABCDEFG")
K_MIN = 1                                   # tabdiff_gnn/config.RULE_SETTINGS["k_min"]
DROPPED = "#d9d9d9"


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


def panel(ax, r: dict, title: str, letter: str):
    S.style_axes(ax)
    s, n = r["s"], len(r["s"])
    x = np.arange(n)
    w = 0.38
    cj, ce = S.RULE_COLOR["jenks"], S.RULE_COLOR["entmax15"]
    ax.bar(x - w / 2, s, w, color=[cj if k else DROPPED for k in r["keep_jenks"]], lw=0)
    ax.bar(x + w / 2, s, w, color=[ce if k else DROPPED for k in r["keep_entmax"]], lw=0)
    ax.axhline(0, color=S.INK2, lw=S.LW_AXIS)
    thr = 2 * r["tau"]
    lo, hi = min(s.min(), thr, 0.0), max(s.max(), 0.0)
    span = hi - lo
    ax.set_ylim(lo - 0.04 * span, hi + 0.13 * span)
    for i in np.flatnonzero(r["keep_entmax"]):           # p_j on the bars entmax-1.5 keeps
        v = s[i]
        ax.annotate(S.tnum(r["p"][i], 2), (i + w / 2, v), xytext=(0, 1.5 if v >= 0 else -1.5),
                    textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=S.FS_CELL, color=S.INK2)
    xs = r["k_jenks"] - 0.5
    ax.axvline(xs, color=cj, lw=0.9, ls="--")
    ax.annotate("Jenks split", (xs, 1), xycoords=("data", "axes fraction"), xytext=(2, -1),
                textcoords="offset points", ha="left", va="top", fontsize=S.FS_CELL + 0.5, color=cj)
    ax.axhline(thr, color=ce, lw=0.9, ls=(0, (1.5, 1.5)))
    ax.annotate(rf"$2\tau = {S.mnum(thr, 3)}$", (n - 0.5, thr), xytext=(0, 1.5), textcoords="offset points",
                ha="right", va="bottom", fontsize=S.FS_CELL + 0.5, color=ce)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(x, NAMES)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Candidate neighbour (sorted by score)")
    ax.set_ylabel(r"Raw score $\varphi$")
    S.title2(ax, title, rf"$k_{{\mathrm{{Jenks}}}} = {r['k_jenks']}$,  $k_{{\mathrm{{entmax}}}} = {r['k_entmax']}$,"
                        rf"  $\tau = {S.mnum(r['tau'], 3)}$")
    S.panel_label(ax, letter)


CAPTION = (
    r"Neighbour selection of the two adaptive rules on one node with seven candidate neighbours A--G, raw "
    r"scores $s = (2.1, 1.9, 1.8, 0.3, 0.1, -0.2, -0.5)$ in (a) and the same scores divided by 10 in (b). "
    r"For each candidate, the left bar is coloured if the Jenks rule keeps it and the right bar if "
    r"entmax-1.5 keeps it (grey: dropped). Jenks splits the sorted scores into two natural-breaks classes "
    r"(dashed line) and keeps the upper class; entmax-1.5 assigns $p_j = [s_j/2 - \tau]_+^2$ with "
    r"$\sum_j p_j = 1$ and keeps every candidate with $p_j > 0$, i.e.\ $s_j > 2\tau$ (dotted line); the "
    r"numbers above the bars are $p_j$. The Jenks split depends only on the shape of the score row, so it "
    r"keeps three neighbours in both panels, whereas entmax-1.5 keeps three in (a) and all seven once the "
    r"scores are flatter (b). The kept sets were computed by the model's own adjacency module on these "
    r"scores (no training involved).")


def main():
    a = compute(S0)
    b = compute([v / 10 for v in S0])
    check_expected(a, EXPECTED["a"], "a")
    check_expected(b, EXPECTED["b"], "b")
    fig = S.figure(S.FULL, 2.4)
    axes = fig.subplots(1, 2)
    panel(axes[0], a, r"Scores $s$", "a")
    panel(axes[1], b, r"Scores $s/10$", "b")
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.RULE_COLOR["jenks"], label="Kept by Jenks"),
               plt.Rectangle((0, 0), 1, 1, color=S.RULE_COLOR["entmax15"], label="Kept by entmax-1.5"),
               plt.Rectangle((0, 0), 1, 1, color=DROPPED, label="Dropped")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=1.0)
    size = S.save(fig, "selection_example", CAPTION)
    for nm, r in (("a", a), ("b", b)):
        print(f"({nm}) k_Jenks {r['k_jenks']}, k_entmax {r['k_entmax']}, tau {r['tau']:.4f}, "
              f"p {np.round(r['p'], 4).tolist()}")
    print(f"wrote selection_example ({size[0]} x {size[1]} in)")


if __name__ == "__main__":
    main()
