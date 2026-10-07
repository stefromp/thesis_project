#!/usr/bin/env python3
"""TabDiff-transfer study (hpc_transfer/tabdiff_study, hpc_transfer/jobs_tabdiff).

    pytest tests/test_tabdiff_transfer.py -v
    TABDIFF_EVAL_PYTHON=/path/to/eval-venv/bin/python pytest tests/test_tabdiff_transfer.py -v

1. Transferred values: for every dataset and arm, each value equals an
   independent read of exp/<src>/ddpm_cb_best/config.toml / configs/<src>.sh,
   each cited file:line holds that key, and the GNN config equals the row of
   analysis/paper_tables/tables/gnn_configs.md.
2. Names and trees: the tabdiff_ prefix is mandatory, results only under
   <exp_root>/exp_tabdiff/, no target collides with a TabDDPM dataset.
3. Checkpoint refusal: the study runner refuses a model trained with another
   cat_input / train_setup / lr / source, or without a record; the existing
   split_runner refusals (cat_input, train_setup) and backcompat suite still pass.
   Stages: one dataset job runs train -> sample -> eval_main -> eval_quality ->
   (dcr) train -> sample -> eval_dcr with done-markers; a rerun skips finished
   stages, a crash resumes at the failed stage (heavy work replaced by a fake).
4. Data: TabDiff code digests, TabDiff split sizes, manifest digests, and the
   export round trip (real train -> export -> read_csv == TabDiff's real.csv).
5. Jobs: one per dataset (or a chained main/dcr pair when > 48 h), gpu
   partition, a100, 24G, cd to repo root, walltimes from walltime_estimates.csv,
   no MLP arm, no mig partition; submission loop commented out, cheapest first.
6. (TABDIFF_EVAL_PYTHON set) real-vs-real through TabDiff's evaluator gives
   sensible metrics.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TD = os.path.join(REPO, "hpc_transfer", "tabdiff_study")
sys.path.insert(0, TD)
import study as S  # noqa: E402
import export as E  # noqa: E402
import runner as R  # noqa: E402

try:
    import tomllib
except ImportError:          # python 3.10
    import tomli as tomllib

JOBS = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff")
TABDDPM_DATASETS = sorted(d for d in os.listdir(os.path.join(REPO, "exp"))
                          if os.path.exists(os.path.join(REPO, "exp", d, "ddpm_cb_best", "config.toml")))


# ---------------------------------------------------------------------------
# 1. transferred values
# ---------------------------------------------------------------------------
def _cited_line(source: str):
    m = re.match(r"^(\S+):(\d+) ", source)
    assert m, f"no file:line in {source!r}"
    path, line = os.path.join(REPO, m.group(1)), int(m.group(2))
    with open(path) as fh:
        return fh.read().splitlines()[line - 1]


@pytest.mark.parametrize("ds", S.ORDER)
def test_transferred_values_match_source_files(ds):
    src = S.SOURCES[ds]
    values, sources = S.transferred(ds)
    cfg = tomllib.load(open(os.path.join(REPO, "exp", src, "ddpm_cb_best", "config.toml"), "rb"))
    expect = {
        "lr": cfg["train"]["main"]["lr"], "steps": cfg["train"]["main"]["steps"],
        "batch_size": cfg["train"]["main"]["batch_size"],
        "weight_decay": cfg["train"]["main"]["weight_decay"],
        "num_timesteps": cfg["diffusion_params"]["num_timesteps"],
        "sample_batch_size": cfg["sample"]["batch_size"],
        "scheduler": cfg["diffusion_params"].get("scheduler", "cosine"),
        "normalization": cfg["train"]["T"]["normalization"],
        "is_y_cond": cfg["model_params"]["is_y_cond"],
    }
    shell = open(os.path.join(REPO, "hpc_transfer", "configs", f"{src}.sh")).read()
    get = lambda k: re.search(rf'^{k}="?([^"#\s]+)', shell, re.M).group(1)  # noqa: E731
    expect.update(gnn_type=get("GNN_TYPE"), n_layers=int(get("N_LAYERS")),
                  d_model=int(get("D_MODEL")), top_k=int(get("TOP_K")),
                  n_heads=int(get("N_HEADS")), use_attention=False)
    for k, v in expect.items():
        assert values[k] == v, (ds, k, values[k], v)
    assert set(values) == set(expect) == set(sources)
    # the user's protocol: 30 000 steps, T 1000 (100 for king and the others the
    # authors tuned so), magic keeps wilt's batch 256
    assert values["steps"] == 30000
    if ds == "magic":
        assert values["batch_size"] == 256
    if src == "king":
        assert values["num_timesteps"] == 100
    for k, s in sources.items():
        if s.startswith("absent"):
            assert k == "scheduler" and "scheduler" not in cfg["diffusion_params"]
            continue
        line = _cited_line(s)
        key = s.split()[-1]
        assert key in line, (k, s, line)


@pytest.mark.parametrize("ds", S.ORDER)
def test_gnn_config_is_the_paper_tables_config(ds):
    src = S.SOURCES[ds]
    g, _ = S.gnn_config(src)
    table = open(os.path.join(REPO, "analysis", "paper_tables", "tables", "gnn_configs.md")).read()
    row = next(l for l in table.splitlines() if l.startswith(f"| {src} ("))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1].lower() == g["gnn_type"]
    assert [int(c) for c in cells[2:6]] == [g["n_layers"], g["d_model"], g["top_k"], g["n_heads"]]


def test_sources_have_the_same_task_type():
    meta = {r["dataset"]: r["task_type"] for r in csv.DictReader(
        open(os.path.join(REPO, "hpc_transfer", "meta_features.csv")))}
    for ds, src in S.SOURCES.items():
        assert meta[src] == S.TASK[ds] == E.load_info(S.data_dir(S.target_name(ds)))["task_type"]


# ---------------------------------------------------------------------------
# 2. names and trees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["adult", "default", "tabdiff_", "tabdiff_diabetes",
                                 "tabdiff_magic_dcr", "Tabdiff_adult", "tabdiff_adult_x",
                                 "adult_tabdiff_split", "../tabdiff_adult"])
def test_unprefixed_or_unknown_names_are_refused(bad):
    with pytest.raises(ValueError):
        S.parse_target(bad)


def test_results_only_under_exp_tabdiff(tmp_path):
    root = str(tmp_path)
    for ds in S.ORDER:
        for dcr in ([False, True] if S.HAS_DCR[ds] else [False]):
            t = S.target_name(ds, dcr)
            p = S.run_dir(root, t)
            rel = os.path.relpath(p, root).split(os.sep)
            assert rel[0] == "exp_tabdiff" and rel[1] == t and rel[2] == "gnn"
            assert rel[1].startswith("tabdiff_")
    for outside in (os.path.join(root, "exp", "adult", "x"), os.path.join(root, "exp_tabdiff", "adult", "x"),
                    os.path.join(root, "exp_tabdiff", "..", "exp", "tabdiff_adult")):
        with pytest.raises(ValueError):
            S.assert_in_study_tree(outside, root)


def test_no_collision_with_tabddpm_datasets():
    for ds in S.ORDER:
        for dcr in (False, True):
            if dcr and not S.HAS_DCR[ds]:
                continue
            t = S.target_name(ds, dcr)
            assert t not in TABDDPM_DATASETS and t.startswith(S.PREFIX)
            assert os.path.basename(S.data_dir(t)) == t
    assert S.RESULTS_DIRNAME != "exp"


# ---------------------------------------------------------------------------
# 3. checkpoint refusal
# ---------------------------------------------------------------------------
def _fake_checkpoint(run, rec):
    os.makedirs(run, exist_ok=True)
    open(os.path.join(run, "model_ema.pt"), "w").close()
    if rec is not None:
        json.dump(rec, open(os.path.join(run, R.RUN_RECORD), "w"))


def test_runner_loads_only_a_matching_checkpoint(tmp_path):
    values, _ = S.transferred("shoppers")
    want = R.run_record("tabdiff_shoppers", values)
    assert want["cat_input"] == "prob" and want["train_setup"] == "tabddpm"
    run = str(tmp_path / "run")
    assert R.check_checkpoint(run, want) is False                     # nothing there
    _fake_checkpoint(run, want)
    assert R.check_checkpoint(run, want) is True                      # match -> load
    for key, other in [("cat_input", "log"), ("train_setup", "ablation"), ("lr", 1e-4),
                       ("num_timesteps", 1000), ("target", "tabdiff_adult"), ("source", "default")]:
        _fake_checkpoint(run, {**want, key: other})
        with pytest.raises(R.CheckpointMismatch, match=key):
            R.check_checkpoint(run, want)
    os.remove(os.path.join(run, R.RUN_RECORD))
    with pytest.raises(R.CheckpointMismatch, match="no denoiser_config.json"):
        R.check_checkpoint(run, want)


def test_model_params_follow_target_task():
    for ds in S.ORDER:
        info = E.load_info(S.data_dir(S.target_name(ds)))
        v, _ = S.transferred(ds)
        mp, gp = R.model_and_graph(ds, v, info)
        assert mp["num_classes"] == (0 if S.TASK[ds] == "regression" else 2)
        assert gp["cat_input"] == "prob" and gp["topk_mode"] == "fixed" and gp["mode"] == "dynamic"
        assert gp["use_attention"] is False and gp["num_timesteps"] == v["num_timesteps"]


def test_no_mlp_arm_left():
    assert not hasattr(S, "ARMS") and not hasattr(S, "mlp_config")
    for p in glob.glob(os.path.join(TD, "*.py")) + glob.glob(os.path.join(JOBS, "*")) + \
            [os.path.join(REPO, "hpc_transfer", "submit_tabdiff.sh"),
             os.path.join(REPO, "analysis", "tabdiff", "compare.py")]:
        text = open(p).read()
        assert "tabddpm_mlp" not in text and "_mlp" not in os.path.basename(p), p


# ---------------------------------------------------------------------------
# 3b. stages / done-markers (heavy work faked)
# ---------------------------------------------------------------------------
class FakeHeavy:
    """Stands in for runner.Heavy: training writes a model file, sampling copies
    the real train split (so the export checks pass), evaluation writes the
    output files tabdiff_eval.py would write. Records every call."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def train(self, t):
        self.calls.append((t.name, "train"))
        open(os.path.join(t.run, "model_ema.pt"), "w").close()

    def sample(self, t, k, gen):
        import numpy as np
        self.calls.append((t.name, f"sample{k}"))
        for f in ("X_num_train.npy", "X_cat_train.npy", "y_train.npy"):
            src = os.path.join(t.data, f)
            if os.path.exists(src):
                np.save(os.path.join(gen, f), np.load(src, allow_pickle=True))

    def evaluate(self, cmd):
        step = cmd[2]
        run = cmd[cmd.index("--run_dir") + 1]
        n = int(cmd[cmd.index("--n_runs") + 1])
        target = cmd[cmd.index("--target") + 1]
        self.calls.append((target, f"eval_{step}"))
        if self.fail_on == (target, step):
            return 1
        d = os.path.join(run, R.EVAL_DIR)
        os.makedirs(d, exist_ok=True)
        cols = {"main": ["shape", "trend", "mle", "c2st"], "dcr": ["dcr"], "quality": ["alpha", "beta"]}[step]
        name = "quality.csv" if step == "quality" else "all_results.csv"
        with open(os.path.join(d, name), "w") as fh:
            fh.write("," + ",".join(cols) + "\n" + "".join(f"{i}," + ",".join("1" for _ in cols) + "\n"
                                                            for i in range(n)))
        mp = os.path.join(d, "eval_meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        meta[step] = {"n_runs": n}
        json.dump(meta, open(mp, "w"))
        return 0


def _args(tmp_path, ds="shoppers", part="all"):
    return R.make_parser().parse_args(["--dataset", ds, "--part", part, "--exp_root", str(tmp_path),
                                       "--eval_python", "EVAL", "--synthcity_python", "SYNTH",
                                       "--n_runs", "2", "--device", "cpu"])


SHOPPERS_ORDER = [("tabdiff_shoppers", "train"), ("tabdiff_shoppers", "sample0"),
                  ("tabdiff_shoppers", "sample1"), ("tabdiff_shoppers", "eval_main"),
                  ("tabdiff_shoppers", "eval_quality"), ("tabdiff_shoppers_dcr", "train"),
                  ("tabdiff_shoppers_dcr", "sample0"), ("tabdiff_shoppers_dcr", "sample1"),
                  ("tabdiff_shoppers_dcr", "eval_dcr")]


def test_dataset_job_runs_every_stage_in_order_then_skips_all(tmp_path):
    h = FakeHeavy()
    R.run_dataset(_args(tmp_path), h)
    assert h.calls == SHOPPERS_ORDER
    for target, stages in (("tabdiff_shoppers", R.STAGES[False]), ("tabdiff_shoppers_dcr", R.STAGES[True])):
        run = S.run_dir(str(tmp_path), target)
        assert all(os.path.exists(R.marker(run, st)) for st in stages)
        meta = json.load(open(os.path.join(run, "run_meta.json")))
        assert meta["study"] == "tabdiff_transfer" and meta["source_dataset"] == "adult"
        assert meta["transferred"]["lr"]["source"].startswith("exp/adult/ddpm_cb_best/config.toml:")
    h2 = FakeHeavy()
    R.run_dataset(_args(tmp_path), h2)
    assert h2.calls == []                                   # nothing repeated


def test_crash_resumes_at_the_failed_stage(tmp_path):
    h = FakeHeavy(fail_on=("tabdiff_shoppers", "quality"))
    with pytest.raises(RuntimeError, match="quality"):
        R.run_dataset(_args(tmp_path), h)
    assert h.calls == SHOPPERS_ORDER[:5]
    run = S.run_dir(str(tmp_path), "tabdiff_shoppers")
    assert os.path.exists(R.marker(run, "eval_main")) and not os.path.exists(R.marker(run, "eval_quality"))
    h2 = FakeHeavy()
    R.run_dataset(_args(tmp_path), h2)
    assert h2.calls == SHOPPERS_ORDER[4:]                   # no retraining, no resampling


def test_a_lost_marker_reruns_only_that_stage(tmp_path):
    R.run_dataset(_args(tmp_path), FakeHeavy())
    run = S.run_dir(str(tmp_path), "tabdiff_shoppers_dcr")
    os.remove(R.marker(run, "eval_dcr"))
    h = FakeHeavy()
    R.run_dataset(_args(tmp_path), h)
    assert h.calls == [("tabdiff_shoppers_dcr", "eval_dcr")]
    # a marker whose outputs vanished does not count
    os.remove(os.path.join(run, "samples", "samples_1.csv"))
    h = FakeHeavy()
    R.run_dataset(_args(tmp_path), h)
    assert h.calls == [("tabdiff_shoppers_dcr", "sample1"), ("tabdiff_shoppers_dcr", "eval_dcr")]


def test_a_differently_trained_model_is_refused_even_with_a_marker(tmp_path):
    R.run_dataset(_args(tmp_path), FakeHeavy())
    run = S.run_dir(str(tmp_path), "tabdiff_shoppers")
    rec = json.load(open(os.path.join(run, R.RUN_RECORD)))
    rec["cat_input"] = "log"
    json.dump(rec, open(os.path.join(run, R.RUN_RECORD), "w"))
    with pytest.raises(R.CheckpointMismatch, match="cat_input"):
        R.run_dataset(_args(tmp_path), FakeHeavy())


def test_parts_and_magic(tmp_path):
    h = FakeHeavy()
    R.run_dataset(_args(tmp_path, "magic"), h)
    assert {c[0] for c in h.calls} == {"tabdiff_magic"}
    with pytest.raises(SystemExit):
        R.run_dataset(_args(tmp_path, "magic", "dcr"), FakeHeavy())
    h = FakeHeavy()
    R.run_dataset(_args(tmp_path, "adult", "dcr"), h)
    assert {c[0] for c in h.calls} == {"tabdiff_adult_dcr"}


def test_existing_refusals_and_backcompat_suite_still_pass():
    r = subprocess.run([sys.executable, os.path.join(REPO, "hpc_transfer", "test_split_runner_backcompat.py")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and "17/17 passed" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_train_setup.py",
                        "tests/test_cat_input.py", "-k", "refus or mismatch"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:]


# ---------------------------------------------------------------------------
# 4. data
# ---------------------------------------------------------------------------
def test_tabdiff_code_unchanged():
    info = S.verify_tabdiff(os.path.join(REPO, S.TABDIFF_REL))
    assert info["commit"] == "5ecdb3356261aea72716cc9a779f31d7ad083bf4"


# n_train / n_test of TabDiff's own processing (Step 0 report)
SIZES = {"tabdiff_adult": (32561, 16281), "tabdiff_default": (27000, 3000),
         "tabdiff_shoppers": (11097, 1233), "tabdiff_magic": (17117, 1902),
         "tabdiff_beijing": (37581, 4176), "tabdiff_news": (35679, 3965),
         "tabdiff_adult_dcr": (24421, 24421), "tabdiff_default_dcr": (15000, 15000),
         "tabdiff_shoppers_dcr": (6165, 6165), "tabdiff_beijing_dcr": (20878, 20879),
         "tabdiff_news_dcr": (19822, 19822)}


@pytest.mark.parametrize("target", sorted(SIZES))
def test_data_is_tabdiffs_split(target):
    d = S.data_dir(target)
    ours = E.load_info(d)
    theirs = json.load(open(os.path.join(d, "tabdiff", "info.json")))
    assert (ours["train_size"], ours["test_size"]) == SIZES[target]
    assert (theirs["train_num"], theirs["test_num"]) == SIZES[target]
    assert ours["column_names"] == theirs["column_names"]
    assert ours["tabdiff"]["commit"] == S.TABDIFF_COMMIT
    manifest = json.load(open(os.path.join(REPO, "data", "tabdiff_manifest.json")))
    for rel, digest in manifest["datasets"][target]["files_sha256_16"].items():
        assert S.sha16(os.path.join(d, rel)) == digest, rel


@pytest.mark.parametrize("target", sorted(SIZES))
def test_export_round_trip(target, tmp_path):
    import selfcheck
    selfcheck.round_trip(target, str(tmp_path), REPO)


def test_export_refuses_a_malformed_table(tmp_path):
    import numpy as np
    d = S.data_dir("tabdiff_shoppers")
    gen = tmp_path / "gen"
    gen.mkdir()
    np.save(gen / "X_num_train.npy", np.load(os.path.join(d, "X_num_train.npy"))[:10])
    xc = np.load(os.path.join(d, "X_cat_train.npy"))[:10].copy()
    xc[0, 0] = "NotAMonth"
    np.save(gen / "X_cat_train.npy", xc)
    np.save(gen / "y_train.npy", np.load(os.path.join(d, "y_train.npy"))[:10])
    with pytest.raises(ValueError, match="categories not in real"):
        E.export_seed(str(gen), d, str(tmp_path / "s.csv"))


# ---------------------------------------------------------------------------
# 5. jobs
# ---------------------------------------------------------------------------
def _job_rows():
    rows = list(csv.DictReader(open(os.path.join(JOBS, "walltime_estimates.csv"))))
    jobs = {r["target"].split()[1]: r for r in rows if r["stage"] == "walltime"}
    stages = [r for r in rows if r["stage"] != "walltime"]
    return jobs, stages


def test_job_scripts():
    jobs, stages = _job_rows()
    scripts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(JOBS, "*.sh"))
                     if not p.endswith("common_tabdiff.sh"))
    assert scripts == sorted(jobs)
    for ds in S.ORDER:
        mine = [j for j in jobs if j in (f"{ds}.sh", f"{ds}_main.sh", f"{ds}_dcr.sh")]
        assert mine == [f"{ds}.sh"] or sorted(mine) == sorted([f"{ds}_dcr.sh", f"{ds}_main.sh"])
        est = sum(float(r["hours_pessimistic"]) for r in stages if r["dataset"] == ds)
        for j in mine:
            g = open(os.path.join(JOBS, j)).read()
            for line in ("#SBATCH --partition=gpu", "#SBATCH --gres=gpu:a100:1", "#SBATCH --mem=24G"):
                assert line in g, (j, line)
            hours = int(jobs[j]["hours_pessimistic"])
            assert f"#SBATCH --time={hours}:00:00" in g and hours <= 48
            if j == f"{ds}.sh":
                assert hours == max(2, math.ceil(est * 1.3))
            assert 'cd "${REPO_ROOT}"' in g and "run_tabdiff_dataset" in g
            assert f'DS="{ds}"' in g
            assert "mig" not in g and "--gres=gpu:1" not in g
    common = open(os.path.join(JOBS, "common_tabdiff.sh")).read()
    assert "--dry_run" in common and "cat_input_preflight" in common and "divergence.json" in common
    assert "gpu_hist" in common and "mig" not in common


def test_submission_loop_is_commented_out_and_cheapest_first():
    text = open(os.path.join(REPO, "hpc_transfer", "submit_tabdiff.sh")).read()
    lines = text.splitlines()
    live = [l for l in lines if re.search(r"\bsbatch\b", l) and not l.lstrip().startswith("#")]
    assert not live, live
    order = [re.search(r"sbatch (?:--parsable )?(\w+?)(?:_main)?\.sh", l).group(1)
             for l in lines if re.search(r"^# (jid=\$\()?sbatch", l)]
    assert order == ["magic", "shoppers", "adult", "default", "beijing", "news"] == S.ORDER
    for l in lines:
        if "_main.sh" in l:
            assert "--dependency=afterok:${jid}" in l and "_dcr.sh" in l
    assert "mig" not in text and "mlp" not in text.lower()


# ---------------------------------------------------------------------------
# 6. real-vs-real through TabDiff's evaluator (needs the eval venv)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("TABDIFF_EVAL_PYTHON"), reason="set TABDIFF_EVAL_PYTHON")
def test_real_vs_real_through_tabdiff_evaluator(tmp_path):
    import shutil
    if shutil.which("nvidia-smi") is None and sys.platform == "darwin":
        extra = ["--cpu_xgb_patch"]       # macOS xgboost wheels have no gpu_hist
    else:
        extra = []
    r = subprocess.run([os.environ["TABDIFF_EVAL_PYTHON"], os.path.join(TD, "selfcheck.py"),
                        "--target", "tabdiff_shoppers", "--work", str(tmp_path), "--n_runs", "2"] + extra,
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = json.load(open(os.path.join(S.results_root(str(tmp_path)),
                                      "tabdiff_shoppers", "selfcheck", "real_boot", "tabdiff_eval",
                                      "selfcheck.json")))
    avg = out["avg_std"]["avg"]
    assert avg["shape"] < 3 and avg["trend"] < 3          # error rates, %
    assert 0.88 < avg["mle"] < 0.96                       # TabDiff Table 3 'Real': .926
    assert avg["c2st"] > 90
