"""
Shared data pieces of the Route-A thesis figures (fig_*.py next to this file): datasets, arms,
run directories and the per-run values -- all taken from build_summary.py (b_values /
b_tabdiff_values) and analysis/tabdiff/route_a_arms.py (arm directories by rule token);
nothing is parsed twice. Style, colours and display names: thesis_style.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_style as S  # noqa: E402  (applies the house style)
import build_summary as BS  # noqa: E402

sys.path.insert(0, str(BS.REPO / "analysis" / "tabdiff"))
import route_a_arms as RA  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt, RC = S.plt, BS.RC
DATASETS = S.DATASETS
# candidate neighbours per node, N - 1 (as in topk_k_vs_t.png); asserted against the data
EXPECTED_N1 = {"adult": 14, "beijing": 11, "default": 23, "diabetes": 36, "magic": 10, "shoppers": 17}
ARMS = list(RA.ARMS)
assert ARMS == S.RULES, (ARMS, S.RULES)
# b_values tag of each arm (dense = the topk_dense ablation run)
ARM_TAG = {"fixed": "fixed", "jenks": "jenks", "entmax15": "entmax15", "dense": "dense"}
CSV = BS.CSV


def n_minus_1(ds: str) -> int:
    n1 = RC.n_nodes(ds) - 1                   # from data/tabdiff_gnn/<target>/info.json
    assert n1 == EXPECTED_N1[ds], (ds, n1, EXPECTED_N1[ds])
    return n1


def run_dir(ds: str, arm: str) -> Path | None:
    """The seed-0 run of (ds, arm), found by rule token and checked against
    build_summary.ra_run (the directory its tables read)."""
    p = RA.arm_dir(str(BS.EXP_ROOT), ds, arm)
    if p is None:
        return None
    mode, abl = RA.ARM_MODE[arm]
    assert Path(p).resolve() == BS.ra_run(ds, mode, abl).resolve(), (ds, arm, p)
    return Path(p)


# build_summary.build_figures() hands over the records it already computed (computing them again
# would append duplicate value records to its tables); standalone scripts compute them once.
_BV: dict | None = None
_TDV: dict | None = None


def b_values() -> dict:
    """build_summary's Route-A records: (ds, tag, metric) -> record (mean, std ddof=1 over the
    20 tables, status, source)."""
    global _BV
    if _BV is None:
        _BV = BS.b_values()
    return _BV


def tabdiff_values() -> dict:
    """build_summary's TabDiff-as-reported records (table B5): (ds, metric) -> record."""
    global _TDV
    if _TDV is None:
        _TDV = BS.b_tabdiff_values()
    return _TDV


def ok(rec) -> bool:
    return rec is not None and rec.get("status") == "ok"


def run_meta(run: Path) -> dict:
    return json.loads((run / "run_meta.json").read_text())


PROTOCOL = r"Route A, training seed 0, 20 sampled tables."
