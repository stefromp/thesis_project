"""
Paper-style figures (analysis/summary/fig_paper_style.py, mirroring the TabDiff paper's Figures 3
and 4): column resolution of the paper's six columns, the heatmap matrices, the category counts
and the shared KDE bandwidth.

    python -m pytest -q tests/test_figures_paper_style.py
Cases that need the Route-A runs (and their <run>/analysis/ outputs) skip where those are missing.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis", "summary"))
import fig_paper_style as P  # noqa: E402

F = P.F


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ds,name,typ,want", [
    ("beijing", "DEWP", "numerical", "DEWP"), ("shoppers", "BounceRates", "numerical", "BounceRates"),
    ("adult", "marital.status", "categorical", "marital.status"), ("beijing", "month", "categorical", "month"),
    ("magic", "class", "categorical", "class"), ("shoppers", "OperatingSystems", "categorical", "OperatingSystems"),
])
def test_paper_columns_resolve(ds, name, typ, want):
    assert P.resolve_column(ds, name, typ)["name"] == want


@pytest.mark.parametrize("ds,variant,want", [
    ("adult", "Marital-Status", "marital.status"), ("adult", "marital_status", "marital.status"),
    ("shoppers", "bounce_rates", "BounceRates"), ("shoppers", "Operating Systems", "OperatingSystems"),
    ("beijing", "dewp", "DEWP"), ("magic", "CLASS", "class"),
])
def test_resolution_tolerates_separators_and_case(ds, variant, want):
    typ = "numerical" if want in ("DEWP", "BounceRates") else "categorical"
    assert P.resolve_column(ds, variant, typ)["name"] == want


def test_resolution_rejects_wrong_type_and_unknown_names():
    with pytest.raises(AssertionError, match="expected"):
        P.resolve_column("beijing", "DEWP", "categorical")
    with pytest.raises(AssertionError, match="matches"):
        P.resolve_column("adult", "no_such_column", "categorical")


def test_paper_column_lists():
    assert [(d, c) for d, c, _ in P.KDE_COLS] == [("beijing", "DEWP"), ("shoppers", "BounceRates")]
    assert [(d, c) for d, c, _ in P.BAR_COLS] == [("adult", "marital.status"), ("beijing", "month"),
                                                  ("magic", "class"), ("shoppers", "OperatingSystems")]
    assert P.VMAX == 0.10 and P.PAPER_DS == ["adult", "beijing", "magic", "shoppers"]


# ---------------------------------------------------------------------------
# Counts and KDE on constructed data
# ---------------------------------------------------------------------------
def test_category_counts_sum_to_real_size_and_natural_order():
    real = pd.Series([" a", "b", "b", " c"] * 25)
    syns = [pd.Series(["b"] * 60 + ["a"] * 40), pd.Series(["c"] * 50)]           # 2nd table half the size
    cats, rc, sc, note = P.category_counts(real, syns)
    assert cats == ["a", "b", "c"] and rc.sum() == len(real) and sc.sum() == pytest.approx(len(real))
    assert "scaled" in note
    assert np.allclose(sc, [(40 + 0) / 2, (60 + 0) / 2, (0 + 100) / 2])
    cats, *_ = P.category_counts(pd.Series([12, 2, 1, 10] * 3), [pd.Series([1.0, 2.0, 10.0, 12.0] * 3)])
    assert cats == ["1", "2", "10", "12"]                                          # numeric order, no ".0"


def test_kde_uses_the_real_bandwidth_for_both():
    rng = np.random.default_rng(0)
    real, syn = rng.normal(0, 1, 2000), rng.normal(0.5, 3, 5000)
    grid = np.linspace(-5, 5, 101)
    dr, ds, bw, n = P.kde_pair(real, syn, grid)
    assert bw == pytest.approx(real.std(ddof=1) * len(real) ** (-1 / 5))          # Scott's rule, 1-D
    assert n == len(syn) and np.trapz(dr, grid) == pytest.approx(1, abs=0.01)


# ---------------------------------------------------------------------------
# On the Route-A runs
# ---------------------------------------------------------------------------
def _have(ds, arm):
    run = F.run_dir(ds, arm)
    return run is not None and (run / P.PM.OUT_DIR / "pair_trend_error.csv").exists()


@pytest.mark.parametrize("ds,arm", [(d, a) for d in P.PAPER_DS for a in F.ARMS if F.RA.has_arm(d, a)])
def test_heatmap_matrix_symmetric_white_diagonal_mean_is_trend(ds, arm):
    if not _have(ds, arm):
        pytest.skip(f"{ds}/{arm}: no pair-trend outputs on disk")
    M, n_num = P.full_pair_matrix(ds, arm)
    n = len(M)
    assert np.isnan(np.diag(M)).all()                                              # white diagonal
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(M[off], M.T[off]) and (M[off] >= 0).all() and (M[off] <= 1).all()
    assert M[off].mean() == pytest.approx(P.trend_mean(ds, arm), abs=1e-9)
    cols = P.columns(ds)
    assert all(c["type"] == "numerical" for c in cols[:n_num]) and all(c["type"] == "categorical" for c in cols[n_num:])


@pytest.mark.parametrize("ds,col,typ", P.BAR_COLS)
def test_category_counts_on_runs_sum_to_real_table(ds, col, typ):
    if F.run_dir(ds, "fixed") is None:
        pytest.skip("Route-A results not on disk")
    name = P.resolve_column(ds, col, typ)["name"]
    real = P.real_column(ds, name)
    for arm in P.arms_of(ds):
        cats, rc, sc, note = P.category_counts(real, P.synthetic_columns(ds, arm, name))
        assert rc.sum() == len(real) and sc.sum() == pytest.approx(len(real), abs=1e-6), (ds, arm)
        assert note == "synthetic tables have the real table's size"
