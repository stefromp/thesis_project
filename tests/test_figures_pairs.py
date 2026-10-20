"""
analysis/tabdiff/pair_and_marginals.py (figures 6-7) on a tiny synthetic real/synthetic pair:
the per-pair / per-column errors come from TabDiff's own evaluate_density, their means
reproduce the property scores and a matching all_results.csv, a wrong all_results.csv fails
loudly, and the marginals are normalised as documented.

    python -m pytest -q tests/test_figures_pairs.py
The end-to-end cases need TabDiff's metrics import chain (sdmetrics, prdc, xgboost, ...):
run them with the eval venv; they skip where it is missing.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis", "tabdiff"))
import pair_and_marginals as PM  # noqa: E402

COLS = ["a", "b", "c", "d", "y"]
INFO = {"name": "magic", "task_type": "binclass", "column_names": COLS, "num_col_idx": [0, 1, 3],
        "cat_col_idx": [2], "target_col_idx": [4],
        "inverse_idx_mapping": {str(i): i for i in range(5)},
        "metadata": {"columns": {"0": {"sdtype": "numerical", "computer_representation": "Float"},
                                 "1": {"sdtype": "numerical", "computer_representation": "Float"},
                                 "3": {"sdtype": "numerical", "computer_representation": "Float"},
                                 "2": {"sdtype": "categorical"}, "4": {"sdtype": "categorical"}}}}


def _table(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(shift, 1, n)
    return pd.DataFrame({"a": a, "b": 0.7 * a + rng.normal(0, 0.5, n), "c": rng.choice(["x", "y", "z"], n),
                         "d": rng.exponential(1 + shift, n), "y": (a > 0).astype(int)})


def test_eval_columns_follow_tabdiff_reorder():
    cols = PM.eval_columns(INFO)
    assert [c["name"] for c in cols] == ["a", "b", "d", "c", "y"]        # num block, then cat (+ target)
    assert [c["type"] for c in cols] == ["numerical"] * 3 + ["categorical"] * 2
    reg = {**INFO, "task_type": "regression", "num_col_idx": [0, 1], "cat_col_idx": [2], "target_col_idx": [3]}
    assert [c["name"] for c in PM.eval_columns(reg)][:3] == ["a", "b", "d"]  # regression target ends the num block


def test_marginals_normalisation():
    real, syns = _table(400, 0), [_table(300, 1, 0.3), _table(300, 2, 0.3)]
    M = PM.marginals(real, syns, PM.eval_columns(INFO))
    for col in ("a", "b", "d"):
        m = M[M.column == col]
        assert len(m) == PM.N_BINS
        w = (m.bin_right - m.bin_left).to_numpy()
        x = real[col].to_numpy()
        lo, hi = np.quantile(x, [PM.Q_LO, PM.Q_HI])
        assert m.bin_left.iloc[0] == pytest.approx(lo) and m.bin_right.iloc[-1] == pytest.approx(hi)
        # area = fraction of rows inside the real 0.5-99.5 % window
        assert (m.real * w).sum() == pytest.approx(((x >= lo) & (x <= hi)).mean())
        s = np.concatenate([t[col].to_numpy() for t in syns])
        assert (m.syn * w).sum() == pytest.approx(((s >= lo) & (s <= hi)).mean())
    for col in ("c", "y"):
        m = M[M.column == col]
        assert m.real.sum() == pytest.approx(1) and m.syn.sum() == pytest.approx(1)
        assert dict(zip(m.category, m.real)) == pytest.approx(real[col].astype(str).value_counts(normalize=True).to_dict())


def test_check_fails_loudly():
    assert PM.check("x", 1.0, 1.0 + 5e-7) <= PM.TOL
    with pytest.raises(AssertionError, match="SELF-CHECK FAILED"):
        PM.check("x", 1.0, 1.0 + 2e-6)
    with pytest.raises(AssertionError):
        PM.check("x", float("nan"), 1.0)


# ---------------------------------------------------------------------------
# end to end through TabDiff's evaluate_density
# ---------------------------------------------------------------------------
def _factory():
    try:
        return PM.tabdiff_metrics_factory(REPO)
    except ImportError as e:                                   # TabDiff's import chain incomplete here
        pytest.skip(f"TabDiff metrics not importable: {e}")


def _fake_repo(tmp_path, n_tables=2):
    """data/tabdiff_gnn/tabdiff_magic/{info.json, real.csv, test.csv} + a run with n_tables tables."""
    d = tmp_path / "repo" / "data" / "tabdiff_gnn" / "tabdiff_magic"
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps(INFO))
    _table(400, 0).to_csv(d / "real.csv", index=False)
    _table(200, 9).to_csv(d / "test.csv", index=False)
    run = tmp_path / "run"
    (run / "samples").mkdir(parents=True)
    for i in range(n_tables):
        _table(400, 10 + i, 0.2 * (i + 1)).to_csv(run / "samples" / f"samples_{i}.csv", index=False)
    return str(tmp_path / "repo"), str(run), d


def _scores(factory, d, run, n_tables):
    """shape / trend error per table as TabDiff's report_test computes them (100 x (1 - property))."""
    info = json.loads((d / "info.json").read_text())
    m = factory({"real": str(d / "real.csv"), "test": str(d / "test.csv")}, info)
    rows = []
    for i in range(n_tables):
        out, _ = m.evaluate_density(pd.read_csv(os.path.join(run, "samples", f"samples_{i}.csv")))
        rows.append({"shape": 100 * (1 - out["density/Shape"]), "trend": 100 * (1 - out["density/Trend"])})
    return pd.DataFrame(rows)


def test_self_checks_pass_and_outputs(tmp_path):
    factory = _factory()
    repo, run, d = _fake_repo(tmp_path)
    os.makedirs(os.path.join(run, "tabdiff_eval"))
    _scores(factory, d, run, 2).to_csv(os.path.join(run, "tabdiff_eval", "all_results.csv"))
    meta = PM.process_run(run, "magic", 2, factory, repo)
    assert max(meta["self_check_max_abs_dev"].values()) <= PM.TOL
    assert meta["samples_from"] == "samples"
    P = pd.read_csv(os.path.join(run, "analysis", "pair_trend_error.csv"))
    assert len(P) == 10 and set(P.metric) == {"correlation", "contingency"}            # C(5, 2) pairs
    num_num = P[(P.type_a == "numerical") & (P.type_b == "numerical")]
    assert (num_num.metric == "correlation").all() and len(num_num) == 3
    # sdmetrics >= 0.2x scores weakly associated pairs as NaN (skipped by its property mean);
    # the pinned 0.18.0 scores every pair. Either way a pair is scored in all tables or in none.
    assert P.n_tables.isin([0, 2]).all() and P[P.n_tables > 0].err_mean.between(0, 1).all()
    assert (P.n_tables == 2).any()
    S = pd.read_csv(os.path.join(run, "analysis", "column_shape_error.csv"))
    assert sorted(S.column) == sorted(COLS)
    ar = pd.read_csv(os.path.join(run, "tabdiff_eval", "all_results.csv"), index_col=0)
    assert 100 * P.err_mean.mean() == pytest.approx(ar.trend.mean(), abs=1e-9)       # mean of means = Trend
    assert 100 * S.err_mean.mean() == pytest.approx(ar["shape"].mean(), abs=1e-9)
    assert os.path.exists(os.path.join(run, "analysis", "marginals.csv"))


def test_self_check_rejects_wrong_all_results(tmp_path):
    factory = _factory()
    repo, run, d = _fake_repo(tmp_path)
    os.makedirs(os.path.join(run, "tabdiff_eval"))
    s = _scores(factory, d, run, 2)
    s.loc[1, "trend"] += 1e-4
    s.to_csv(os.path.join(run, "tabdiff_eval", "all_results.csv"))
    with pytest.raises(AssertionError, match="table 1 trend vs all_results"):
        PM.process_run(run, "magic", 2, factory, repo)


def test_sample_copies_must_agree(tmp_path):
    repo, run, d = _fake_repo(tmp_path)
    alls = os.path.join(run, "tabdiff_eval", "all_samples")
    os.makedirs(alls)
    for i in range(2):
        _table(400, 10 + i, 0.2 * (i + 1)).to_csv(os.path.join(alls, f"samples_{i}.csv"), index=False)
    assert PM.sample_paths(run, 2)[1] == "tabdiff_eval/all_samples"
    _table(400, 99).to_csv(os.path.join(alls, "samples_1.csv"), index=False)
    with pytest.raises(RuntimeError, match="differ"):
        PM.sample_paths(run, 2)


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------
def test_analysis_job_script():
    import importlib.util
    import re
    import subprocess
    job = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff_gnn", "analysis_pair_marginals.sh")
    t = open(job).read()
    assert subprocess.run(["bash", "-n", job]).returncode == 0
    line = [ln for ln in t.splitlines() if re.match(r"#\s+sbatch ", ln)]
    assert line and all("--gres=gpu:a100:1" in ln and "--mem=24G" in ln and re.search(r"--time=\d+:\d\d:\d\d", ln)
                        and ln.rstrip().endswith("analysis_pair_marginals.sh") for ln in line), line
    assert "#SBATCH --partition=gpu" in t and "venvs/tabdiff_eval310/bin/python" in t
    assert "module purge" in t and all(f"module load {m}" in t for m in ("hdf5", "python/3.10", "cuda/11.8", "cmake"))
    assert "analysis/tabdiff/pair_and_marginals.py" in t and 'DATASETS="${DATASETS:-beijing,default,magic}"' in t
    spec = importlib.util.spec_from_file_location("route_a_make_jobs_pm", os.path.join(REPO, "tabdiff_gnn", "make_jobs.py"))
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    assert mj.keep_on_regenerate(job)                                   # make_jobs keeps analysis_*.sh
