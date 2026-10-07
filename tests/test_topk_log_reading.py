"""
compare_topk.read_topk_log: a re-run gen seed must not count twice, and a
truncated copy must be flagged.

    python3 -m pytest -q tests/test_topk_log_reading.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "analysis", "topk_comparison"))
import compare_topk as ct  # noqa: E402

HEAD = "step,block,t_bucket,k_mean,n_samples,phase,tag\n"


def _rows(phase, tag, k, steps=3, blocks=2):
    return "".join(f"{s},{b},7,{k},100,{phase},{tag}\n" for s in range(steps) for b in range(blocks))


def test_rerun_seed_is_deduplicated_and_training_rows_kept(tmp_path):
    p = tmp_path / "topk_stats.csv"
    # training, then seed 0 and 1 sampled, then seed 0 re-run (split job)
    p.write_text(HEAD + _rows("train", "", 4.0) + _rows("sample", 0, 2.0) + _rows("sample", 1, 3.0)
                 + _rows("sample", 0, 2.5))
    t, truncated = ct.read_topk_log(p)
    assert not truncated
    s = t[t.phase == "sample"]
    assert len(s) == 12, "each (step, block, t_bucket, tag) must appear once"
    assert set(s[s.tag == 0].k_mean) == {2.5}, "the re-run (last) copy must win"
    assert len(t[t.phase == "train"]) == 6, "training rows must be untouched"


def test_truncated_copy_is_flagged_and_partial_row_dropped(tmp_path):
    p = tmp_path / "topk_stats.csv"
    p.write_text(HEAD + _rows("train", "", 4.0) + _rows("sample", 0, 2.0) + "3,0,7,2.0")  # cut mid-row
    t, truncated = ct.read_topk_log(p)
    assert truncated
    assert len(t) == 12, "the cut-off row must be dropped"
