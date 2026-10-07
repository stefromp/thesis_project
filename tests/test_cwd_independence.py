"""
The pipeline must load the same repo files whatever the working directory.

lib.get_catboost_config used to read 'tuned_models/catboost/<ds>_cv.json'
relative to the CWD, so jobs submitted from hpc_transfer/jobs_catfix/ died in
their first CatBoost evaluation. It now resolves from the repo root. This
checks that (1) it works from an unrelated directory and (2) it returns
exactly the file the old CWD-relative lookup found when run from the repo
root -- i.e. working jobs load the same config as before.

    python3 -m pytest -q tests/test_cwd_independence.py
"""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _import_lib():
    # Same environment the runner sets up for itself (zero / rtdl stubs where
    # those packages are absent), as the backcompat tests do.
    abl = os.path.join(REPO, "ablation")
    if abl not in sys.path:
        sys.path.insert(0, abl)
    import ablation_runner as ar
    ar._setup_zero_rtdl_stubs()
    # lib/data.py imports category_encoders (-> statsmodels). Where that chain
    # is broken (e.g. statsmodels 0.14 with scipy >= 1.16) a stub is enough:
    # nothing here touches LeaveOneOutEncoder.
    try:
        import category_encoders  # noqa: F401
    except ImportError:
        sys.modules["category_encoders"] = types.SimpleNamespace(LeaveOneOutEncoder=object)
    import lib
    return lib


DATASETS = sorted(f[: -len("_cv.json")] for f in os.listdir(os.path.join(REPO, "tuned_models", "catboost"))
                  if f.endswith("_cv.json"))


@pytest.mark.parametrize("ds", DATASETS)
def test_get_catboost_config_is_cwd_independent(ds, tmp_path, monkeypatch):
    lib = _import_lib()
    old_relative = os.path.join("tuned_models", "catboost", f"{ds}_cv.json")

    # what the old code loaded when run from the repo root (every working job)
    monkeypatch.chdir(REPO)
    with open(old_relative) as fh:
        expected = json.load(fh)
    assert lib.get_catboost_config(os.path.join(REPO, "data", ds)) == expected

    # and the same from an unrelated directory, where the old lookup failed
    monkeypatch.chdir(tmp_path)
    assert not os.path.exists(old_relative), "test setup: tmp dir must not contain tuned_models/"
    assert lib.get_catboost_config(os.path.join(REPO, "data", ds)) == expected
    # the dataset name is all that matters, as before (relative real_data_path too)
    assert lib.get_catboost_config(f"data/{ds}/") == expected
