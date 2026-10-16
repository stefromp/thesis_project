"""
Tests for analysis/tabdiff/schedules_vs_graph.py (learned schedules vs the learned graph).

    python -m pytest -q tests/test_schedules_vs_graph.py

Needs torch and TabDiff's model / diffusion modules (third_party/TabDiff). The check
against the production sampling path (entry.build + TabDiff's Trainer) also needs
TabDiff's full import chain (category_encoders, sdmetrics, ...): run it with the eval
venv; it skips where that is missing. adaptive_entmax15 needs `entmax`. Covered:
  * the quoted TabDiff code lines are TabDiff's (file:line holds the quoted code)
  * the numpy sigma_i(t) / mask_j(t) / rho / k / t* equal TabDiff's own schedule
    modules on the same raw parameters (init and moved, incl. softplus' threshold)
  * extract reads the EMA schedules of the sampled checkpoint
  * logging on/off gives bit-identical samples (fixed, jenks, entmax15) and logs
    every denoiser call of TabDiff's sampler and every block, with fixed top-k exact
  * the rebuilt sampling model equals entry.build + Trainer (eval venv)
  * the interaction statistic and permutation test on constructed graphs
  * stand-in end to end: extract -> sample -> report
  * the analysis job script survives make_jobs and carries the sbatch flags
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABDIFF = os.path.join(REPO, "third_party", "TabDiff")
for p in (os.path.join(REPO, "analysis", "tabdiff"), os.path.join(REPO, "tabdiff_gnn")):
    if p not in sys.path:
        sys.path.insert(0, p)
import config as C  # noqa: E402
import schedules_vs_graph as SG  # noqa: E402

TOML = os.path.join(TABDIFF, "tabdiff", "configs", "tabdiff_configs.toml")
JOB = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff_gnn", "analysis_schedules.sh")


def _toml():
    try:
        import tomllib
    except ImportError:                                  # python < 3.11
        tomllib = pytest.importorskip("tomli")
    with open(TOML, "rb") as fh:
        return tomllib.load(fh)


def _schedules():
    torch = pytest.importorskip("torch")
    if TABDIFF not in sys.path:
        sys.path.insert(0, TABDIFF)
    from tabdiff.models.noise_schedule import LogLinearNoise_PerColumn, PowerMeanNoise_PerColumn
    return torch, PowerMeanNoise_PerColumn, LogLinearNoise_PerColumn


# ---------------------------------------------------------------------------
# Part 1: the extracted schedules are TabDiff's
# ---------------------------------------------------------------------------
def test_quoted_tabdiff_code_lines_are_tabdiffs():
    for f, n, code, _ in SG.SCHEDULE_CODE:
        lines = open(os.path.join(TABDIFF, f)).read().splitlines()
        assert lines[n - 1].split("  #")[0].strip() == code, (f, n, lines[n - 1])   # trailing comment off


@pytest.mark.parametrize("which", ["init", "moved"])
def test_curves_and_tstar_match_tabdiff_schedule_functions(which):
    torch, PM, LL = _schedules()
    nsp = _toml()["diffusion_params"]["noise_schedule_params"]
    sigma_data = _toml()["diffusion_params"]["edm_params"]["sigma_data"]
    num, cat = PM(num_numerical=7, **nsp), LL(num_categories=5, **nsp)
    if which == "moved":                                # incl. past softplus' threshold (20)
        with torch.no_grad():
            num.rho_raw.copy_(torch.tensor([-4.0, -0.5, 0.0, 3.3, 7.0, 12.5, 21.0]))
            cat.k_raw.copy_(torch.tensor([-8.0, -6.0, -1.0, 0.7, 2.5]))
    p = {"rho_offset": nsp["rho_offset"], "k_offset": nsp["k_offset"]}
    rho = SG.rho_from_raw(num.rho_raw.detach().double().numpy(), p["rho_offset"])
    k = SG.k_from_raw(cat.k_raw.detach().double().numpy(), p["k_offset"])
    np.testing.assert_allclose(rho, num.rho().detach().double().numpy(), rtol=1e-6)
    np.testing.assert_allclose(k, cat.k().detach().double().numpy(), rtol=1e-6)
    t = torch.tensor(SG.T_GRID, dtype=torch.float32)[:, None]
    with torch.no_grad():
        sig_td = num.total_noise(t).double().numpy()
        mask_td = (-torch.expm1(-cat.total_noise(t))).double().numpy()   # unified_ctime_diffusion.py:121
    sig = SG.sigma_curve(SG.T_GRID, rho, nsp["sigma_min"], nsp["sigma_max"])
    msk = SG.mask_curve(SG.T_GRID, k, nsp["eps_max"], nsp["eps_min"])
    np.testing.assert_allclose(sig, sig_td, rtol=2e-4)                   # float32 vs float64
    np.testing.assert_allclose(msk, mask_td, rtol=2e-4, atol=1e-7)
    # t*: TabDiff's own inverse at sigma_data; mask exactly 0.5 there
    ts_n = SG.tstar_num(rho, nsp["sigma_min"], nsp["sigma_max"], sigma_data)
    ts_c = SG.tstar_cat(k, nsp["eps_max"], nsp["eps_min"])
    with torch.no_grad():
        inv = num.inverse_to_t(torch.full((1, 7), float(sigma_data))).double().numpy()[0]
    np.testing.assert_allclose(ts_n, inv, atol=2e-5)
    for i in range(7):
        assert SG.sigma_curve([ts_n[i]], rho[i:i + 1], nsp["sigma_min"], nsp["sigma_max"])[0, 0] == \
            pytest.approx(sigma_data, rel=1e-10)
    for j in range(5):
        assert SG.mask_curve([ts_c[j]], k[j:j + 1], nsp["eps_max"], nsp["eps_min"])[0, 0] == \
            pytest.approx(0.5, rel=1e-10)
        with torch.no_grad():
            got = float(-torch.expm1(-cat.total_noise(torch.tensor([[ts_c[j]]], dtype=torch.float64)))[0, j])
        assert got == pytest.approx(0.5, abs=1e-5)
    if which == "init":                                 # TabDiff's toml: rho 12.0009, k 1.00248
        assert rho[0] == pytest.approx(12.0009115, abs=1e-6) and k[0] == pytest.approx(1.0024757, abs=1e-6)
        assert ts_n[0] == pytest.approx(0.4784, abs=1e-4) and ts_c[0] == pytest.approx(0.50136, abs=1e-4)


# ---------------------------------------------------------------------------
# Stand-in runs: real node layout (info.json), synthetic category sizes
# ---------------------------------------------------------------------------
def _layout(ds):
    info = json.load(open(C.data_files(C.target_name(ds))["info"]))
    reg = info["task_type"] == "regression"
    d_num = len(info["num_col_idx"]) + (1 if reg else 0)
    n_cat = len(info["cat_col_idx"]) + (0 if reg else 1)
    assert d_num + n_cat == len(C.model_node_columns(ds))
    return d_num, [3] * n_cat


def make_stand_in(exp_root, ds="magic", arm="fixed", seed=0, checkpoint="best_ema_model_1.2345_4501.pt"):
    """A finished-looking Route-A run: spec, config.pkl, best EMA checkpoint (random
    denoiser, moved schedules), sample/train records, run_meta."""
    torch = pytest.importorskip("torch")
    if TABDIFF not in sys.path:
        sys.path.insert(0, TABDIFF)
    import entry
    from denoiser import build_denoiser
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model
    mode, abl = SG.ARMS[arm]
    target = C.target_name(ds)
    run = C.run_dir(str(exp_root), target, "attngnn", 0, topk_mode=mode, ablation=abl)
    os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
    spec = entry.spec_for(target, "attngnn", 0, ablation=abl, topk_mode=mode)
    raw = _toml()
    d_num, cats = _layout(ds)
    raw["unimodmlp_params"].update(d_numerical=d_num, categories=[c + 1 for c in cats])
    raw["diffusion_params"].update(scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    torch.manual_seed(seed)
    backbone = build_denoiser("attngnn", raw["unimodmlp_params"], spec["gnn"], dim_t=spec["dim_t"])
    model = Model(backbone, **raw["diffusion_params"]["edm_params"])
    diffusion = UnifiedCtimeDiffusion(num_classes=np.array(cats), num_numerical_features=d_num, denoise_fn=model,
                                      y_only_model=None, **raw["diffusion_params"], device="cpu")
    with torch.no_grad():
        diffusion.num_schedule.rho_raw.normal_(7.0, 3.0)
        diffusion.cat_schedule.k_raw.normal_(-2.0, 2.0)
    torch.save({"denoise_fn": model.state_dict(), "num_schedule": diffusion.num_schedule.state_dict(),
                "cat_schedule": diffusion.cat_schedule.state_dict()}, os.path.join(run, "ckpt", checkpoint))
    with open(os.path.join(run, "ckpt", "config.pkl"), "wb") as fh:
        pickle.dump(raw, fh)
    json.dump(spec, open(os.path.join(run, "route_a_spec.json"), "w"))
    json.dump({"checkpoint": checkpoint}, open(os.path.join(run, "sample_record.json"), "w"))
    json.dump({"checkpoint": checkpoint, "completed": True}, open(os.path.join(run, "train_record.json"), "w"))
    json.dump({"git_commit": "standin0000"}, open(os.path.join(run, "run_meta.json"), "w"))
    return run, diffusion


def test_extract_reads_the_sampled_ema_schedules(tmp_path):
    torch = pytest.importorskip("torch")
    run, diffusion = make_stand_in(tmp_path / "exp", "magic", "fixed")
    out = tmp_path / "out"
    rows = SG.extract(str(tmp_path / "exp"), str(out), ["magic"])
    import pandas as pd
    s = pd.read_csv(out / "schedules.csv")
    assert len(rows) == len(s) == 11 and set(s["arm"]) == {"fixed"}
    with torch.no_grad():
        rho = diffusion.num_schedule.rho().double().numpy()
        k = diffusion.cat_schedule.k().double().numpy()
    np.testing.assert_allclose(s[s["type"] == "num"].sort_values("node")["value"], rho, rtol=1e-6)
    np.testing.assert_allclose(s[s["type"] == "cat"].sort_values("node")["value"], k, rtol=1e-6)
    assert list(s.sort_values("node")["column"]) == C.model_node_columns("magic")
    assert sorted(s["reveal_rank"]) == list(range(1, 12))
    assert (s["source_commit"].str.contains("untracked sha256/16=")).all() and (s["producer_commit"] == "standin0000").all()
    runs = pd.read_csv(out / "runs.csv")
    assert set(runs[runs["status"] != "ok"]["arm"]) == {"adaptive_jenks", "adaptive_entmax15", "topk_dense"}
    # a checkpoint that is not the sampled one is refused, not silently used
    os.rename(os.path.join(run, "ckpt", "best_ema_model_1.2345_4501.pt"), os.path.join(run, "ckpt", "best_ema_model_1.0_5000.pt"))
    assert SG.resolve_run("magic", "fixed", str(tmp_path / "exp"))["status"].startswith("sampled checkpoint")


# ---------------------------------------------------------------------------
# Part 2: logging never changes the samples
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["fixed", "adaptive_jenks", "adaptive_entmax15"])
def test_edge_logging_on_off_gives_identical_samples(tmp_path, arm):
    torch = pytest.importorskip("torch")
    if arm == "adaptive_entmax15":
        pytest.importorskip("entmax")
    make_stand_in(tmp_path / "exp", "magic", arm)
    run = SG.resolve_run("magic", arm, str(tmp_path / "exp"))
    assert run["status"] == "ok"
    diffusion, backbone, raw = SG.build_lean(run, "cpu")
    n, T = 24, raw["diffusion_params"]["num_timesteps"]
    a = SG.draw(diffusion, n, 0)
    log = SG.EdgeLogger().attach(backbone)
    b = SG.draw(diffusion, n, 0)
    log.detach()
    c = SG.draw(diffusion, n, 0)
    assert torch.equal(a, b) and torch.equal(a, c)
    assert not backbone._forward_pre_hooks and not any(blk.dynamic_adj._forward_hooks for blk in backbone.blocks)
    # every call of TabDiff's sampler: T first-order + (T - 1) 2nd-order corrections, every block
    nb = len(backbone.blocks)
    assert len(log.calls) == 2 * T - 1 and len(log.records) == (2 * T - 1) * nb
    ts = np.array([t for _, t, _ in log.calls])
    assert ts[0] > 0.97 and ts[-1] < 0.03 and (ts >= 0).all() and (ts <= 1).all()
    N = backbone.n_nodes
    for _, _, _, counts, rows in log.records:
        assert rows == n and counts.shape == (N, N) and (np.diag(counts) == 0).all()
        k = counts.sum(1)
        if arm == "fixed":                              # exactly top_k kept per row
            assert (k == n * run_gnn(run)["top_k"]).all()
        else:
            assert (k >= n).all() and (k <= n * (N - 1)).all()


def run_gnn(run):
    return json.load(open(run["spec"]))["gnn"]


def test_rebuilt_model_equals_entry_build_path(tmp_path):
    """Production path (entry.build + TabDiff's Trainer loading the checkpoint) vs the
    rebuilt model: same draw. Needs TabDiff's full import chain (eval venv)."""
    for m in ("torch", "category_encoders", "icecream", "tomli", "sdmetrics"):
        pytest.importorskip(m)
    try:
        import entry
        entry._tabdiff_imports()
    except Exception as e:                              # e.g. a broken local statsmodels
        pytest.skip(f"TabDiff import chain unavailable: {e}")
    make_stand_in(tmp_path / "exp", "magic", "fixed")
    run = SG.resolve_run("magic", "fixed", str(tmp_path / "exp"))
    diffusion, _, raw = SG.build_lean(run, "cpu")
    res = SG.verify_against_entry(run, diffusion, "cpu", 16, 0, raw["unimodmlp_params"]["d_numerical"])
    assert res.get("identical") is True, res


# ---------------------------------------------------------------------------
# Interaction statistic
# ---------------------------------------------------------------------------
def _calls(N=6, M=40):
    tstar = np.array([0.9, 0.75, 0.6, 0.45, 0.3, 0.15])[:N]
    t = np.linspace(1.0, 0.0, M)
    return tstar, t


def test_uniform_graph_has_zero_excess_and_p_one():
    tstar, t = _calls()
    N = len(tstar)
    counts = np.ones((len(t), N, N))
    for m in range(len(t)):
        np.fill_diagonal(counts[m], 0)
    r = SG.permutation_test(t, counts, tstar, n_perm=99)
    assert r["excess_clean"] == pytest.approx(0.0, abs=1e-12) and r["p_two_sided"] == 1.0
    assert r["excess_earlier"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_informative_pairs"] > 0


def test_graph_pointing_to_clean_columns_is_detected():
    tstar, t = _calls()
    N = len(tstar)
    counts = np.zeros((len(t), N, N))
    for m, tm in enumerate(t):
        clean = np.where(tstar > tm)[0]
        for i in range(N):
            js = [j for j in clean if j != i] or [j for j in range(N) if j != i]
            counts[m, i, js] = 10
    r = SG.permutation_test(t, counts, tstar, n_perm=499, seed=1)
    assert r["obs_clean"] == pytest.approx(1.0) and r["excess_clean"] > 0.3
    assert r["p_two_sided"] < 0.05 and r["p_greater"] < 0.05
    # by hand: one informative call, node 0 keeps clean node 1 only; 1 of 5 others clean
    c1 = np.zeros((1, N, N))
    c1[0, 0, 1] = 1
    o, e, npairs, ned = SG.clean_share(np.array([0.7]), c1, tstar)
    assert (o, npairs, ned) == (1.0, 1, 1.0) and e == pytest.approx(1 / 5)


def test_holm():
    np.testing.assert_allclose(SG.holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan])


def test_softplus_is_torchs():
    torch = pytest.importorskip("torch")
    x = np.array([-30.0, -6.0, -1.0, 0.0, 2.0, 7.0, 19.9, 20.0, 20.1, 40.0])
    np.testing.assert_allclose(SG.softplus(x), torch.nn.functional.softplus(torch.tensor(x)).numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# End to end on stand-ins
# ---------------------------------------------------------------------------
def test_end_to_end_stand_in(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    import pandas as pd
    arms = ["fixed", "adaptive_jenks"]
    for arm in arms:
        make_stand_in(tmp_path / "exp", "magic", arm, seed=len(arm))
    out = str(tmp_path / "out")
    SG.extract(str(tmp_path / "exp"), out, ["magic"])
    metas = SG.sample(str(tmp_path / "exp"), out, "cpu", 16, 0, 0, ["magic"])
    assert [m["arm"] for m in metas] == arms and all(m["checks"]["on_vs_off"]["identical"] for m in metas)
    text = SG.report(out, n_perm=30)
    for h in ("Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Table 6", "Table 7", "Reference arm"):
        assert h in text, h
    assert "Reading:" in text
    t7 = pd.read_csv(os.path.join(out, "table6_interaction.csv"))
    assert set(t7["arm"]) == set(arms) and {"0", "1", "all"} <= set(t7["block"].astype(str))
    for f in ("table3_spearman.csv", "table5_sampling_checks.csv", "table6_interaction.csv"):
        cols = set(pd.read_csv(os.path.join(out, f)).columns)
        assert {"source", "source_commit", "producer_commit"} <= cols, f
    for f in ("curves_magic", "tstar_by_arm", "edges_magic"):
        assert os.path.exists(os.path.join(out, "figures", f + ".png")), f
    prov = pd.read_csv(os.path.join(out, "provenance.csv"))
    assert prov["path"].str.contains("best_ema_model").any() and prov["path"].str.contains("edge_counts").any()
    ef = pd.read_csv(os.path.join(out, "edge_freq.csv.gz"))
    assert ef["freq"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------
def test_analysis_job_script():
    t = open(JOB).read()
    assert subprocess.run(["bash", "-n", JOB]).returncode == 0
    line = [ln for ln in t.splitlines() if re.match(r"#\s+sbatch ", ln)]
    assert line and all("--gres=gpu:a100:1" in ln and "--mem=24G" in ln and re.search(r"--time=\d+:\d\d:\d\d", ln)
                        for ln in line), line
    assert "#SBATCH --partition=gpu" in t and "#SBATCH --time=01:00:00" in t
    assert "analysis/tabdiff/schedules_vs_graph.py" in t and "--verify_entry" in t
    assert "_dcr" not in re.sub(r"#.*", "", t)
    # make_jobs regenerates the training scripts but keeps analysis_*.sh
    import importlib.util
    spec = importlib.util.spec_from_file_location("route_a_make_jobs_sv", os.path.join(REPO, "tabdiff_gnn", "make_jobs.py"))
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    assert mj.keep_on_regenerate(JOB) and mj.keep_on_regenerate("x/common_tabdiff_gnn.sh")
    assert not mj.keep_on_regenerate("x/magic_tabdiff_gnn.sh")
