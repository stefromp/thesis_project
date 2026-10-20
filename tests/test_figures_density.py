"""
Figure 3 (analysis/summary/fig_density_vs_fidelity.py) and the arm-directory resolver it uses.

    python -m pytest -q tests/test_figures_density.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis", "summary"))
import fig_density_vs_fidelity as FD  # noqa: E402

F = FD.F


def test_n_minus_1_from_the_data():
    for ds, n1 in F.EXPECTED_N1.items():
        assert F.n_minus_1(ds) == n1
    assert F.EXPECTED_N1 == {"adult": 14, "beijing": 11, "default": 23, "diabetes": 36, "magic": 10,
                             "shoppers": 17}


def test_rule_tokens():
    assert F.RA.rule_token("beijing", "fixed") == "topk5"
    assert F.RA.rule_token("magic", "fixed") == "topk3"
    assert F.RA.rule_token("beijing", "dense") == "topk11"
    assert F.RA.rule_token("magic", "dense") == "topk10"
    assert F.RA.rule_token("diabetes", "jenks") == "jenks"
    assert not F.RA.has_arm("adult", "dense")


def _have(ds, arm):
    return F.run_dir(ds, arm) is not None


@pytest.mark.parametrize("ds,arm", [(d, a) for d in F.DATASETS for a in ("fixed", "dense") if F.RA.has_arm(d, a)])
def test_fixed_and_dense_x_is_k_over_n_minus_1(ds, arm):
    run = F.run_dir(ds, arm)
    if run is None:
        pytest.skip(f"{ds}/{arm} not on disk")
    x, _ = FD.kept_fraction(ds, arm, run)
    k = F.run_meta(run)["gnn"]["top_k"]
    assert x == k / F.EXPECTED_N1[ds]
    if arm == "dense":
        assert x == 1.0


def test_adaptive_x_is_count_weighted(tmp_path):
    """sum k_sum / sum n / (N - 1) over the sampling phase only; training rows ignored."""
    n1 = F.n_minus_1("magic")
    rows = []
    for node in range(n1 + 1):
        rows.append({"phase": "sample", "block": 0, "t_bin": 0, "node": node, "n": 10, "k_sum": 20, "k_min": 1,
                     "k_max": 3})
        rows.append({"phase": "sample", "block": 1, "t_bin": 5, "node": node, "n": 30, "k_sum": 30, "k_min": 1,
                     "k_max": 1})
        rows.append({"phase": "train", "block": 0, "t_bin": 0, "node": node, "n": 1000, "k_sum": 9000,
                     "k_min": 9, "k_max": 9})
    pd.DataFrame(rows).to_csv(tmp_path / "topk_k_stats.csv", index=False)
    x, _ = FD.kept_fraction("magic", "jenks", tmp_path)
    assert x == pytest.approx((20 + 30) / (10 + 30) / n1)


@pytest.mark.skipif(not _have("beijing", "entmax15"), reason="Route-A results not on disk")
def test_collected_rows_and_csv_columns():
    D = FD.collect()
    fixed = D[D.arm.isin(["fixed", "dense"])]
    for _, r in fixed.iterrows():
        assert r.kept_fraction == F.run_meta(Path(REPO) / r.run)["gnn"]["top_k"] / r["N-1"]
    assert {"c2st_xgb_mean", "c2st_xgb_std", "wnmis_mean", "wnmis_std", "kept_fraction"} <= set(D.columns)
    assert set(D.dataset) <= set(F.DATASETS)
