"""
Thesis figure style (analysis/summary/thesis_style.py): rendered text carries no code identifiers,
datasets are capitalised, nothing is below 6 pt, and every figure is saved at the text width
with its caption file.

    python -m pytest -q tests/test_figures_style.py
The end-to-end case builds every figure into a temporary directory (Figure 2 needs entmax; the
Route-A results and the <run>/analysis/ outputs must be on disk, else those parts skip).
"""
from __future__ import annotations

import os
import re
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis", "summary"))
import thesis_style as S  # noqa: E402


def _fig_with(*texts, fontsize=8):
    fig = S.figure(S.HALF, 1.5)
    ax = fig.subplots()
    for i, t in enumerate(texts):
        ax.text(0.1, 0.1 + 0.1 * i, t, fontsize=fontsize)
    return fig


@pytest.mark.parametrize("bad", ["adaptive_jenks", "entmax15", "topk5", "top_k = 5", "jenks", "beijing (N-1 = 11)",
                                 "Default vs default", "c2st_xgb", "wnmis", "seed0", "a_b"])
def test_code_identifiers_and_lowercase_datasets_are_rejected(bad):
    fig = _fig_with(bad)
    assert S.check_texts(fig), bad
    with pytest.raises(ValueError, match="rendered text problems"):
        S.save(fig, "should_not_exist", "x")
    S.plt.close(fig)
    assert not (S.OUT_DIR / "should_not_exist.pdf").exists()


@pytest.mark.parametrize("good", ["Jenks", "entmax-1.5", "Fixed top-k", r"Dense ($k = N-1$)", "Beijing ($N-1 = 11$)",
                                  r"$k_{\mathrm{Jenks}} = 3$", "C2ST (XGBoost)", "wNMIS", "Magic (dense)"])
def test_display_names_pass(good):
    fig = _fig_with(good)
    assert S.check_texts(fig) == []
    S.plt.close(fig)


def test_data_labels_can_be_allowed_and_small_fonts_are_not():
    fig = _fig_with("BILL_AMT1", "default payment next month")
    assert S.check_texts(fig) and S.check_texts(fig, {"BILL_AMT1", "default payment next month"}) == []
    S.plt.close(fig)
    fig = _fig_with("Jenks", fontsize=5)
    assert any("pt <" in p for p in S.check_texts(fig))
    S.plt.close(fig)


def test_palette_and_names_are_fixed():
    assert S.RULE_COLOR == {"fixed": "#6b6b6b", "jenks": "#1f6fd1", "entmax15": "#e8662e", "dense": "#111111"}
    assert S.RULE_MARKER == {"fixed": "s", "jenks": "o", "entmax15": "^", "dense": "D"}
    assert [S.rule_label(r) for r in S.RULES] == ["Fixed top-k", "Jenks", "entmax-1.5", r"Dense ($k = N-1$)"]
    assert [S.ds_label(d) for d in S.DATASETS] == ["Adult", "Beijing", "Default", "Diabetes", "Magic", "Shoppers"]
    assert len(set(S.DS_COLOR.values())) == 6


def test_saved_width_is_the_text_width(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT_DIR", tmp_path)
    fig = S.figure(S.FULL, 2.0)
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel("A long label to make the margins matter")
    S.save(fig, "w", r"Caption with $k$.")
    assert abs(_pdf_width_in(tmp_path / "w.pdf") - S.FULL) < 0.002
    assert (tmp_path / "w.caption.tex").read_text().strip() == r"Caption with $k$."


def _pdf_width_in(path) -> float:
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", open(path, "rb").read())
    return (float(m.group(3)) - float(m.group(1))) / 72


def test_every_figure_builds_clean(tmp_path, monkeypatch):
    """All thesis figures through the real save() path (which runs the text checks) into tmp_path."""
    import build_summary as BS
    import route_a_figs as RF
    if RF.run_dir("beijing", "entmax15") is None:
        pytest.skip("Route-A results not on disk")
    csv = tmp_path / "csv"
    csv.mkdir()
    for f in ("B6.csv", "B6_detail.csv"):                 # read by the Figure 5 checks
        shutil.copy(BS.CSV / f, csv / f)
    monkeypatch.setattr(S, "OUT_DIR", tmp_path)
    monkeypatch.setattr(BS, "FIG", tmp_path)
    monkeypatch.setattr(RF, "CSV", csv)
    monkeypatch.setattr(RF, "_BV", None)
    monkeypatch.setattr(RF, "_TDV", None)
    status = BS.build_figures(RF.b_values(), RF.tabdiff_values())
    for name, where, _ in BS.THESIS_FIGURES:
        if status[name] != "rebuilt":
            assert name in ("selection_example",) or "analysis" in status[name], (name, status[name])
            continue
        for ext in ("pdf", "png", "caption.tex"):
            assert (tmp_path / f"{name}.{ext}").exists(), (name, ext)
        assert abs(_pdf_width_in(tmp_path / f"{name}.pdf") - S.FULL) < 0.002, name
        cap = (tmp_path / f"{name}.caption.tex").read_text()
        assert "Route A, training seed 0, 20 sampled tables" in cap or name == "selection_example", name
        assert not re.search(r"adaptive_|entmax15|topk\d", cap), name
    assert (tmp_path / "_contact_sheet.pdf").exists() and (tmp_path / "_preview.tex").exists()
    assert "selection_example" in status


@pytest.mark.parametrize("cap,ok", [(r"Fine: 5\% and $k$.", True), ("Bad: 5% here.", False), ("Odd $k.", False),
                                    ("Unicode minus − is not pdflatex-safe.", False)])
def test_caption_must_be_latex_safe(tmp_path, monkeypatch, cap, ok):
    monkeypatch.setattr(S, "OUT_DIR", tmp_path)
    fig = S.figure(S.HALF, 1.0)
    fig.subplots()
    if ok:
        S.save(fig, "c", cap)
        assert (tmp_path / "c.caption.tex").read_text().strip() == cap
    else:
        with pytest.raises(ValueError, match="caption"):
            S.save(fig, "c", cap)
        S.plt.close(fig)
