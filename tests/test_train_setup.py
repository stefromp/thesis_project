#!/usr/bin/env python3
"""--train_setup {ablation, tabddpm}: budget-matched GNN runs.

    pytest tests/test_train_setup.py -v

1. The values read for the five target datasets match their config.toml (read
   independently here) and every cited line holds that key.
2. "ablation" (default) is unchanged: same values, same tree names, same
   denoiser record, no train_setup.json / run_meta change.
3. A missing key or a quoted number fails loudly.
4. _run_one with "tabddpm" (heavy functions stubbed): train / sample get the
   config's values, results land in *_tddpm_setup[_catfix], run_meta.json and
   denoiser_config.json record the setup.
5. split_runner refuses a checkpoint whose recorded train_setup differs, both ways.
6. Divergence: a non-finite or exploding loss aborts with divergence.json and
   no saved model; divergence_check=False leaves the loop untouched.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "hpc_transfer"))
import test_split_runner_backcompat as bc  # noqa: E402  (stubs: _import_runner, _stub_world, ...)

@pytest.fixture(autouse=True)
def _restore_patched():
    """bc._stub_world patches train.train, sample.sample, two sys.modules
    entries and two ablation_runner loaders. Undo it after every test so the
    backcompat suite's signature checks see the real functions whatever the
    test order."""
    ar = bc._import_runner()
    import sample as sample_mod
    import train as train_mod
    saved = (train_mod.train, sample_mod.sample, ar._load_raw_real, ar._load_raw_syn,
             {k: sys.modules.get(k) for k in ("eval_catboost", "eval_tabddpm_protocol")})
    yield
    train_mod.train, sample_mod.sample, ar._load_raw_real, ar._load_raw_syn, mods = saved
    for k, v in mods.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


TARGETS = ("cardio", "house", "king", "gesture", "california")
EXPENSIVE = ("higgs-small", "miniboone")


def _toml(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _cfg(ar, ds):
    import importlib
    return importlib.import_module(ds.replace("-", "_") + "_ablation").DATASET_CFG


# ---------------------------------------------------------------------------
# 1. values per dataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ds", TARGETS + EXPENSIVE)
def test_tabddpm_values_match_config_toml(ds):
    ar = bc._import_runner()
    values, sources = ar.training_setup(ds, _cfg(ar, ds), "tabddpm", REPO)
    path = os.path.join(REPO, "exp", ds, "ddpm_cb_best", "config.toml")
    c = _toml(path)
    expect = {"lr": c["train"]["main"]["lr"], "steps": c["train"]["main"]["steps"],
              "batch_size": c["train"]["main"]["batch_size"],
              "weight_decay": c["train"]["main"]["weight_decay"],
              "num_timesteps": c["diffusion_params"]["num_timesteps"],
              "scheduler": c["diffusion_params"].get("scheduler", "cosine"),
              "sample_batch_size": c["sample"]["batch_size"]}
    assert values == expect, (values, expect)
    lines = open(path).read().splitlines()
    for k, src in sources.items():
        if src.startswith("absent"):
            assert "scheduler" not in c["diffusion_params"] and k == "scheduler"
            continue
        line = int(src.split(":")[1].split()[0])
        key = src.split()[-1]
        assert lines[line - 1].split("=")[0].strip() == key, f"{src} -> {lines[line - 1]!r}"


def test_expected_changes_vs_current_runs():
    """The Step-0 list: only lr, steps (and king's num_timesteps) change."""
    ar = bc._import_runner()
    for ds in TARGETS:
        cfg = _cfg(ar, ds)
        old, _ = ar.training_setup(ds, cfg, "ablation", REPO)
        new, _ = ar.training_setup(ds, cfg, "tabddpm", REPO)
        changed = {k for k in old if old[k] != new[k]}
        assert changed == ({"lr", "steps", "num_timesteps"} if ds == "king" else {"lr", "steps"}), (ds, changed)


# ---------------------------------------------------------------------------
# 2. default unchanged
# ---------------------------------------------------------------------------
def test_ablation_setup_is_the_module_constants():
    ar = bc._import_runner()
    cfg = _cfg(ar, "cardio")
    values, _ = ar.training_setup("cardio", cfg, "ablation", REPO)
    assert values == dict(lr=ar.ABLATION_LR, steps=ar.ABLATION_STEPS, batch_size=ar.ABLATION_BATCH_SIZE,
                          weight_decay=cfg["weight_decay"], num_timesteps=ar.ABLATION_NUM_TIMESTEPS,
                          scheduler=cfg["scheduler"], sample_batch_size=cfg["sample_batch_size"])
    assert (ar.ABLATION_LR, ar.ABLATION_STEPS, ar.ABLATION_BATCH_SIZE, ar.ABLATION_NUM_TIMESTEPS) == \
        (1e-4, 20_000, 4096, 1000)


def test_tree_names_and_record():
    ar = bc._import_runner()
    # unchanged defaults
    assert ar.results_subdir() == "ablation"
    assert ar.results_subdir("fixed", "prob") == "ablation_catfix"
    assert ar.results_subdir("adaptive_jenks", "log") == "ablation_adaptive_jenks"
    assert ar.denoiser_record("prob", "fixed") == {"cat_input": "prob", "topk_mode": "fixed"}
    # new trees, never equal to an existing one
    assert ar.results_subdir("fixed", "log", "tabddpm") == "ablation_tddpm_setup"
    assert ar.results_subdir("fixed", "prob", "tabddpm") == "ablation_tddpm_setup_catfix"
    assert ar.denoiser_record("prob", "fixed", "tabddpm")["train_setup"] == "tabddpm"
    with pytest.raises(ValueError):
        ar.results_subdir("fixed", "log", "bogus")


def test_cli_default_and_steps_conflict():
    ar = bc._import_runner()
    assert ar.make_parser("x").parse_args([]).train_setup == "ablation"
    args = ar.make_parser("x").parse_args(["--train_setup", "tabddpm", "--steps", "5"])
    with pytest.raises(SystemExit):
        ar.check_train_setup_args(args)
    ar.check_train_setup_args(ar.make_parser("x").parse_args(["--steps", "5"]))  # fine for ablation


# ---------------------------------------------------------------------------
# 3. loud failures
# ---------------------------------------------------------------------------
def _fake_repo(text):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "exp", "fake", "ddpm_cb_best")
    os.makedirs(d)
    with open(os.path.join(d, "config.toml"), "w") as fh:
        fh.write(text)
    return tmp


GOOD = """seed = 0
[diffusion_params]
num_timesteps = 100
[train.main]
steps = 30000
lr = 0.001
weight_decay = 0.0
batch_size = 256
[sample]
num_samples = 10
batch_size = 5
"""


def test_missing_key_and_bad_type_fail_loudly():
    ar = bc._import_runner()
    tmp = _fake_repo(GOOD.replace("lr = 0.001\n", ""))
    try:
        with pytest.raises(KeyError, match=r"\[train.main\] lr"):
            ar.training_setup("fake", {}, "tabddpm", tmp)
    finally:
        shutil.rmtree(tmp)
    tmp = _fake_repo(GOOD.replace("steps = 30000", 'steps = "30000"'))
    try:
        with pytest.raises(TypeError, match="steps"):
            ar.training_setup("fake", {}, "tabddpm", tmp)
    finally:
        shutil.rmtree(tmp)
    with pytest.raises(FileNotFoundError):
        ar.training_setup("no_such_dataset", {}, "tabddpm", REPO)


# ---------------------------------------------------------------------------
# 4. _run_one end to end (stubbed)
# ---------------------------------------------------------------------------
def _setup_world(tmp_repo_text=GOOD):
    ar = bc._import_runner()
    tmp = tempfile.mkdtemp()
    repo = _fake_repo(tmp_repo_text)
    os.rename(os.path.join(repo, "exp", "fake"), os.path.join(repo, "exp", "unittest_ds"))
    rec = bc._Recorder()
    bc._stub_world(rec, tmp)
    ar._load_raw_real = lambda p: (None, None, [], None)
    ar._load_raw_syn = lambda *a, **k: None
    rk = {**bc._run_kwargs(tmp), "repo_root": repo, "topk_mode": "fixed"}
    return ar, rec, tmp, repo, rk


def test_run_one_tabddpm_setup_end_to_end():
    ar, rec, tmp, repo, rk = _setup_world()
    try:
        rk = {**rk, "cat_input": "prob", "train_setup": "tabddpm"}
        parent = os.path.join(tmp, "exp", "unittest_ds", "ablation_tddpm_setup_catfix",
                              ar.exp_dir_name("gatv2", 3, 64, 3, 4, False))
        os.makedirs(parent)
        with open(os.path.join(parent, "run_meta.json"), "w") as fh:   # as common.sh writes it
            json.dump({"dataset": "unittest_ds", "steps": "module default (ablation_runner.ABLATION_STEPS)"}, fh)
        ar._run_one(**{**rk, "skip_if_done": False})
        (_, tkw), = [c for c in rec.calls if c[0] == "train"]
        assert (tkw["lr"], tkw["steps"], tkw["batch_size"], tkw["weight_decay"], tkw["num_timesteps"],
                tkw["scheduler"], tkw["divergence_check"]) == (0.001, 30000, 256, 0.0, 100, "cosine", True)
        assert tkw["graph_params"]["num_timesteps"] == 100 and tkw["graph_params"]["cat_input"] == "prob"
        assert tkw["parent_dir"] == parent
        samples = [kw for n, kw in rec.calls if n == "sample"]
        assert len(samples) == ar.N_GEN_SEEDS
        assert all(kw["num_timesteps"] == 100 and kw["batch_size"] == 5 for kw in samples)
        meta = json.load(open(os.path.join(parent, "run_meta.json")))
        assert meta["train_setup"] == "tabddpm" and meta["steps"] == 30000
        assert meta["train_setup_sources"]["lr"].startswith("exp/unittest_ds/ddpm_cb_best/config.toml:")
        assert json.load(open(os.path.join(parent, ar.DENOISER_CONFIG_FILE)))["train_setup"] == "tabddpm"
        assert os.path.exists(os.path.join(parent, ar.TRAIN_SETUP_FILE))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(repo, ignore_errors=True)


def test_run_one_default_writes_nothing_new():
    ar, rec, tmp, repo, rk = _setup_world()
    try:
        ar._run_one(**{**rk, "skip_if_done": False})
        (_, tkw), = [c for c in rec.calls if c[0] == "train"]
        assert (tkw["lr"], tkw["steps"], tkw["batch_size"], tkw["num_timesteps"], tkw["divergence_check"]) == \
            (1e-4, 20000, 4096, 1000, False)
        parent = tkw["parent_dir"]
        assert os.path.basename(os.path.dirname(parent)) == "ablation"
        assert json.load(open(os.path.join(parent, ar.DENOISER_CONFIG_FILE))) == \
            {"cat_input": "log", "topk_mode": "fixed"}
        assert not os.path.exists(os.path.join(parent, ar.TRAIN_SETUP_FILE))
        assert not os.path.exists(os.path.join(parent, "run_meta.json"))
        samples = [kw for n, kw in rec.calls if n == "sample"]
        assert all(kw["num_timesteps"] == 1000 and kw["batch_size"] == 10 for kw in samples)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. split_runner checkpoint check
# ---------------------------------------------------------------------------
def test_split_runner_refuses_mismatched_train_setup():
    ar, rec, tmp, repo, rk = _setup_world()
    sys.path.insert(0, os.path.join(REPO, "hpc_transfer"))
    import split_runner as sr
    try:
        rk_t = {**rk, "train_setup": "tabddpm"}
        parent_t = sr.parent_dir_for(ar, rk_t)
        assert os.path.basename(os.path.dirname(parent_t)) == "ablation_tddpm_setup"
        sr.run_split(ar, rk_t, [0], load_checkpoint=False)
        assert json.load(open(os.path.join(parent_t, ar.DENOISER_CONFIG_FILE)))["train_setup"] == "tabddpm"
        tkw = [kw for n, kw in rec.calls if n == "train"][-1]
        assert (tkw["lr"], tkw["steps"], tkw["num_timesteps"]) == (0.001, 30000, 100)
        sr.run_split(ar, rk_t, [1], load_checkpoint=True)               # matching: loads
        assert [kw for n, kw in rec.calls if n == "sample"][-1]["batch_size"] == 5

        # tabddpm checkpoint requested as ablation (e.g. copied by hand)
        rk_a = {**rk, "train_setup": "ablation"}
        parent_a = sr.parent_dir_for(ar, rk_a)
        shutil.copytree(parent_t, parent_a)
        with pytest.raises(SystemExit, match="train_setup='tabddpm'"):
            sr.run_split(ar, rk_a, [2], load_checkpoint=True)

        # an ablation checkpoint (record without the key) requested as tabddpm
        with open(os.path.join(parent_a, ar.DENOISER_CONFIG_FILE), "w") as fh:
            json.dump(ar.denoiser_record("log", "fixed"), fh)
        shutil.rmtree(parent_t)
        shutil.copytree(parent_a, parent_t)
        with pytest.raises(SystemExit, match="train_setup='ablation'"):
            sr.run_split(ar, rk_t, [2], load_checkpoint=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. divergence detection
# ---------------------------------------------------------------------------
class _FakeDiffusion:
    """mixed_loss returns scripted (multi, gauss) losses; one real parameter so
    the optimizer has something to step."""

    def __init__(self, losses):
        import torch
        self.w = torch.nn.Parameter(torch.zeros(1))
        self._denoise_fn = torch.nn.Linear(1, 1)
        self.losses = iter(losses)

    def parameters(self):
        return [self.w, *self._denoise_fn.parameters()]

    def mixed_loss(self, x, out):
        m, g = next(self.losses)
        return self.w * 0 + m, self.w * 0 + g


def _trainer(losses, check, tmp):
    import torch
    bc._import_runner()
    import train as train_mod
    import importlib
    train_mod = importlib.reload(train_mod)     # bc._stub_world replaces train.train only; Trainer is real
    batches = iter([(torch.zeros(4, 1), torch.zeros(4))] * len(losses))
    return train_mod, train_mod.Trainer(_FakeDiffusion(losses), batches, lr=1e-3, weight_decay=0.0,
                                        steps=len(losses), device=torch.device("cpu"),
                                        parent_dir=tmp, divergence_check=check)


def test_nan_loss_aborts_with_report():
    tmp = tempfile.mkdtemp()
    try:
        losses = [(1.0, 1.0)] * 50 + [(float("nan"), 1.0)] + [(1.0, 1.0)] * 49
        tm, tr = _trainer(losses, True, tmp)
        with pytest.raises(tm.TrainingDiverged, match="step 51"):
            tr.run_loop()
        rep = json.load(open(os.path.join(tmp, tm.DIVERGENCE_REPORT)))
        assert rep["step"] == 51 and "non-finite" in rep["reason"]
        assert not os.path.exists(os.path.join(tmp, "model.pt"))
    finally:
        shutil.rmtree(tmp)


def test_explosion_aborts_but_noise_does_not():
    tmp = tempfile.mkdtemp()
    try:
        # 1500 steps near 1.0 then a jump to 50 -> 100-step window ~50 > 10 x best
        losses = [(0.5, 0.5)] * 1500 + [(25.0, 25.0)] * 200
        tm, tr = _trainer(losses, True, tmp)
        with pytest.raises(tm.TrainingDiverged, match="explosion"):
            tr.run_loop()
        # a 3x bump is noise, not divergence
        losses = [(0.5, 0.5)] * 1500 + [(1.5, 1.5)] * 200
        tm, tr = _trainer(losses, True, tmp)
        tr.run_loop()
    finally:
        shutil.rmtree(tmp)


def test_check_off_is_the_original_loop():
    """divergence_check=False: a NaN passes through exactly as before the option
    (logged into loss_history, no exception), and the logged history is the
    same as with the check on for finite losses."""
    import math
    tmp = tempfile.mkdtemp()
    try:
        losses = [(0.4 + 0.001 * i, 0.6) for i in range(300)]
        _, a = _trainer(losses, False, tmp)
        a.run_loop()
        _, b = _trainer(losses, True, tmp)
        b.run_loop()
        assert a.loss_history.equals(b.loss_history)
        _, c = _trainer([(1.0, 1.0)] * 99 + [(float("nan"), 1.0)], False, tmp)
        c.run_loop()
        assert math.isnan(c.loss_history.mloss.iloc[-1])
    finally:
        shutil.rmtree(tmp)
