"""
Figure 2 (analysis/summary/fig_selection_example.py): the worked example is computed by the
model's own selection code and shows the numbers the thesis text quotes.

    python -m pytest -q tests/test_figures_selection.py      (needs torch + entmax==1.3)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis", "summary"))

pytest.importorskip("torch")
pytest.importorskip("entmax")
import fig_selection_example as FS  # noqa: E402


def test_panel_a_scores_as_given():
    r = FS.compute(FS.S0)
    assert (r["k_jenks"], r["k_entmax"]) == (3, 3)
    assert r["tau"] == pytest.approx(0.3927, abs=5e-5)
    assert np.allclose(r["p"][:3], [0.432, 0.311, 0.257], atol=5e-4) and np.all(r["p"][3:] == 0)
    assert r["keep_jenks"].tolist() == [True] * 3 + [False] * 4
    assert r["keep_entmax"].tolist() == [True] * 3 + [False] * 4


def test_panel_b_scores_divided_by_10():
    r = FS.compute([v / 10 for v in FS.S0])
    assert (r["k_jenks"], r["k_entmax"]) == (3, 7)
    assert r["tau"] == pytest.approx(-0.3352, abs=5e-5)
    assert r["keep_entmax"].all() and r["p"].sum() == pytest.approx(1, abs=1e-6)


def test_expected_checks_reject_a_wrong_number():
    r = FS.compute(FS.S0)
    FS.check_expected(r, FS.EXPECTED["a"], "a")
    with pytest.raises(AssertionError):
        FS.check_expected(r, {**FS.EXPECTED["a"], "k_entmax": 4}, "a")


def test_order_of_candidates_does_not_matter():
    """The module sees the scores unsorted; the kept set follows the scores, not positions."""
    perm = [4, 0, 6, 2, 5, 1, 3]
    r = FS.compute([FS.S0[i] for i in perm])
    kept = {perm[i] for i in np.flatnonzero(r["keep_jenks"])}
    assert kept == {0, 1, 2} and r["k_entmax"] == 3
