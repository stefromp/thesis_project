"""
Figure 5 (analysis/summary/fig_vs_tabdiff_heatmap.py): orientation of the relative difference
(positive = ours better) for every metric, the B6 tie rule, and agreement with table B6.

    python -m pytest -q tests/test_figures_vs_tabdiff.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis", "summary"))
import fig_vs_tabdiff_heatmap as H  # noqa: E402

B6_DETAIL = os.path.join(REPO, "analysis", "summary", "csv", "B6_detail.csv")


@pytest.mark.parametrize("ds,metric,lower", [
    ("adult", "shape", True), ("adult", "trend", True), ("adult", "mle", False), ("adult", "alpha", False),
    ("adult", "beta", False), ("adult", "c2st", False),
    ("beijing", "mle", True),                       # RMSE: lower is better
    ("beijing", "shape", True), ("beijing", "c2st", False), ("magic", "mle", False), ("diabetes", "mle", False),
])
def test_direction_per_metric(ds, metric, lower):
    assert H.lower_is_better(ds, metric) is lower


@pytest.mark.parametrize("lower", [True, False])
def test_positive_means_ours_better(lower):
    better, worse = (0.8, 1.2) if lower else (1.2, 0.8)
    assert H.rel_diff(better, 1.0, lower) == pytest.approx(20.0)
    assert H.rel_diff(worse, 1.0, lower) == pytest.approx(-20.0)


def test_beijing_rmse_orientation():
    # paper RMSE 0.555; ours 0.814 (worse) must be negative, 0.5 (better) positive
    assert H.rel_diff(0.814, 0.555, H.lower_is_better("beijing", "mle")) < 0
    assert H.rel_diff(0.5, 0.555, H.lower_is_better("beijing", "mle")) > 0


def test_tie_rule():
    assert H.outcome(1.5, 0.5, 1.0, lower_better=True) == "T"            # |diff| == std -> tie
    assert H.outcome(1.75, 0.5, 1.0, lower_better=True) == "L"
    assert H.outcome(0.25, 0.5, 1.0, lower_better=True) == "W"
    assert H.outcome(0.25, 0.5, 1.0, lower_better=False) == "L"
    assert H.outcome(1.0, 0.0, 1.0, lower_better=False) == "T"


def test_tabdiff_reference_values():
    H.check_reference(H.F.tabdiff_values())


def _runs_on_disk():
    return os.path.exists(B6_DETAIL) and H.F.run_dir("adult", "fixed") is not None


@pytest.mark.skipif(not _runs_on_disk(), reason="Route-A results or B6_detail.csv not on disk")
def test_reproduces_b6_with_topk3():
    D = H.collect("topk3")
    assert H.check_b6(D, "topk3") == []
    ok = D[D.status == "ok"].set_index(["rule", "dataset", "metric"])
    det = pd.read_csv(B6_DETAIL).set_index(["rule", "dataset", "metric"])
    assert sorted(ok.index) == sorted(det.index)
    assert (ok.result == det.result.reindex(ok.index)).all()


@pytest.mark.skipif(not _runs_on_disk(), reason="Route-A results or B6_detail.csv not on disk")
def test_dense_default_changes_only_magic_fixed():
    a, b = H.collect("dense"), H.collect("topk3")
    m = a.merge(b, on=["rule", "dataset", "metric"], suffixes=("_d", "_t"))
    diff = m[(m.arm_d != m.arm_t)]
    assert set(zip(diff.rule, diff.dataset)) == {("fixed", "magic")}
    assert (diff.arm_d == "dense").all()
    H.check_b6(a, "dense")                       # cell-by-cell elsewhere, counts after swapping magic back
