"""
Shared pieces of the Route-A thesis figures (fig_*.py next to this file): datasets, arms,
colours, run directories and the per-run values -- all taken from build_summary.py
(b_values / b_tabdiff_values / style) and analysis/tabdiff/route_a_arms.py (arm
directories by rule token); nothing is parsed twice.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_summary as BS  # noqa: E402

sys.path.insert(0, str(BS.REPO / "analysis" / "tabdiff"))
import route_a_arms as RA  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ct, plt, RC = BS.ct, BS.plt, BS.RC
DATASETS = ["adult", "beijing", "default", "diabetes", "magic", "shoppers"]
# candidate neighbours per node, N - 1 (as in topk_k_vs_t.png); asserted against the data
EXPECTED_N1 = {"adult": 14, "beijing": 11, "default": 23, "diabetes": 36, "magic": 10, "shoppers": 17}
ARMS = list(RA.ARMS)
ARM_LABEL = RA.ARM_LABEL
# B6 / b_values tag of each arm (dense = the topk_dense ablation run)
ARM_TAG = {"fixed": "fixed", "jenks": "jenks", "entmax15": "entmax15", "dense": "dense"}
C_ARM = {"fixed": ct.C_TEXT2, "jenks": BS.C_RULE["adaptive_jenks"], "entmax15": BS.C_RULE["adaptive_entmax15"],
         "dense": ct.C_TEXT}
MK_ARM = {"fixed": "s", "jenks": "o", "entmax15": "^", "dense": "D"}
# one colour per dataset: palette slots 3-8 (slots 1-2 = Jenks / entmax-1.5 in every thesis figure)
C_DS = dict(zip(DATASETS, ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
style_ax, savefig = BS.style_ax, BS.savefig
FIG, CSV = BS.FIG, BS.CSV


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


@functools.lru_cache(maxsize=None)
def b_values() -> dict:
    """build_summary's Route-A records: (ds, tag, metric) -> record (mean, std ddof=1 over the
    20 tables, status, source)."""
    return BS.b_values()


@functools.lru_cache(maxsize=None)
def tabdiff_values() -> dict:
    """build_summary's TabDiff-as-reported records (table B5): (ds, metric) -> record."""
    return BS.b_tabdiff_values()


def ok(rec) -> bool:
    return rec is not None and rec.get("status") == "ok"


def run_meta(run: Path) -> dict:
    return json.loads((run / "run_meta.json").read_text())


def caption(fig, text: str, y: float = -0.005):
    fig.text(0.01, y, text, fontsize=8, color=ct.C_TEXT2, wrap=True, va="top")


__all__ = ["BS", "RA", "np", "pd", "plt", "ct", "RC"]
