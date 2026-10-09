"""
Tests for Route A (tabdiff_gnn/: AttnGNN as the denoiser inside TabDiff).

    python -m pytest -q tests/test_tabdiff_gnn.py

Model tests need TabDiff's imports (torch, category_encoders, icecream, tomli,
sdmetrics for the main.py test) -- run them with the eval venv's python; they
skip where an import is missing. Covered:
  * denoiser=transformer: the wrapper path is TabDiff's UniModMLP -- bit-identical
    parameters and outputs (raw and through Model/Precond) for the same seed/batch
  * the mirrored entry point builds the model tabdiff/main.py builds
  * attngnn: shapes/dtypes, gradients, unchanged imported modules, no
    transformer/MLP left, shared vs per-node noise conditioning, reserved hooks
  * TabDiff digests (study + Route-A pins) and tabpc digests unchanged
  * names / trees / data / runner stages / jobs / analysis helpers
"""
from __future__ import annotations

import copy
import csv
import glob
import json
import os
import re
import subprocess
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tabdiff_gnn"))
import config as C  # noqa: E402

TABDIFF = os.path.join(REPO, C.TABDIFF_REL)


def _load_route_a(name):
    """tabdiff_gnn/<name>.py under a unique module name: hpc_transfer/tabdiff_study has
    modules with the same names (runner, make_jobs, prepare_data) on sys.path too."""
    import importlib.util
    key = f"route_a_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, os.path.join(REPO, "tabdiff_gnn", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _need_tabdiff():
    for m in ("torch", "category_encoders", "icecream", "tomli"):
        pytest.importorskip(m)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def test_route_a_pin_set_covers_executed_files_and_matches():
    want = {"tabdiff/modules/main_modules.py", "tabdiff/modules/transformer.py",
            "tabdiff/models/unified_ctime_diffusion.py", "tabdiff/models/noise_schedule.py",
            "tabdiff/main.py", "tabdiff/configs/tabdiff_configs.toml"}
    srcs = {os.path.relpath(p, TABDIFF) for p in glob.glob(os.path.join(TABDIFF, "src", "*.py"))}
    assert want | srcs <= set(C.ROUTE_A_TABDIFF_FILES)
    assert set(C.S.PINNED_TABDIFF_FILES) <= set(C.ALL_TABDIFF_FILES)       # study pins kept
    info = C.verify_tabdiff()
    assert info["commit"] == C.TABDIFF_COMMIT


def test_tabpc_pinned_and_unchanged():
    assert C.verify_tabpc()["commit"] == C.TABPC_COMMIT
    head = subprocess.run(["git", "-C", os.path.join(REPO, C.TABPC_REL), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode == 0:
        assert head.stdout.strip() == C.TABPC_COMMIT


def test_a_modified_tabdiff_file_is_refused(tmp_path):
    import shutil
    fake = tmp_path / "repo"
    shutil.copytree(TABDIFF, fake / C.TABDIFF_REL, ignore=shutil.ignore_patterns("data", "__pycache__", "images"))
    shutil.copytree(os.path.join(TABDIFF, "data", "Info"), fake / C.TABDIFF_REL / "data" / "Info")
    C.verify_tabdiff(str(fake))
    p = fake / C.TABDIFF_REL / "tabdiff" / "modules" / "main_modules.py"
    p.write_text(p.read_text() + "\n# edit\n")
    with pytest.raises(RuntimeError, match="main_modules.py"):
        C.verify_tabdiff(str(fake))


def test_tabdiff_submodule_has_no_local_changes():
    r = subprocess.run(["git", "-C", TABDIFF, "status", "--porcelain"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no git checkout of the submodule")
    assert r.stdout.strip() == ""


# ---------------------------------------------------------------------------
# Names, trees, data
# ---------------------------------------------------------------------------
def test_targets_order_and_diabetes_last():
    assert C.ORDER[-1] == "diabetes" and C.ORDER[:6] == C.S.ORDER
    assert C.all_targets() == [f"tabdiff_{d}" for d in C.ORDER]          # no DCR in this study
    for bad in ("adult", "diabetes", "tabdiff_magic_dcr", "tabdiff_adult_dcr", "tabdiff_wilt", "exp_tabdiff"):
        with pytest.raises(ValueError):
            C.parse_target(bad)


def test_run_tree(tmp_path):
    r = C.run_dir(str(tmp_path), "tabdiff_news", "attngnn", 1)
    assert r == os.path.join(str(tmp_path), "exp_tabdiff_gnn", "tabdiff_news", "attngnn",
                             "v2_gatv2_layers3_dim256_topk5_heads8_dimt128_noise-shared", "seed1")
    t = C.run_dir(str(tmp_path), "tabdiff_magic", "transformer", 0)
    assert t.endswith(os.path.join("tabdiff_magic", "transformer", "unimodmlp_tabdiff_config", "seed0"))
    with pytest.raises(ValueError):
        C.assert_in_tree(os.path.join(str(tmp_path), "exp_tabdiff", "tabdiff_magic"), str(tmp_path))
    with pytest.raises(ValueError):
        C.assert_in_tree(os.path.join(str(tmp_path), "exp_tabdiff_gnn", "tabdiff_magic", "gnn"), str(tmp_path))


def test_gnn_config_is_the_source_mapping_with_d256():
    for ds in C.ORDER:
        g, src = C.gnn_config(ds)
        want = dict(C.S.gnn_config(C.SOURCES[ds])[0])
        assert g["d_model"] == C.D_MODEL == 256 and "256" in src["d_model"]
        assert {k: v for k, v in g.items() if k != "d_model"} == {k: v for k, v in want.items() if k != "d_model"}
        assert 256 % g["n_heads"] == 0, ds                       # heads divide 256 on every dataset
    assert C.SOURCES["diabetes"] == "adult"
    assert not hasattr(C, "CAPACITY_D_MODEL") and not hasattr(C, "REPRO_DATASETS")


@pytest.mark.parametrize("ds", ["magic", "shoppers", "adult", "default", "beijing", "news"])
def test_same_split_as_the_tabdiff_study(ds):
    t = C.target_name(ds)
    f = C.data_files(t)
    theirs = os.path.join(REPO, "data", t, "tabdiff")
    for name in ("real", "test", "info"):
        assert C.sha16(f[name]) == C.sha16(os.path.join(theirs, os.path.basename(f[name]))), (t, name)
    assert f["val"] is None
    for n in ("X_num_train.npy", "y_train.npy", "X_num_test.npy", "y_test.npy"):
        assert os.path.exists(os.path.join(f["dir"], n))


def test_no_dcr_anywhere_in_route_a(tmp_path):
    evaluate = _load_route_a("evaluate")
    runner = _load_route_a("runner")
    assert not glob.glob(os.path.join(REPO, C.DATA_REL, "*_dcr"))
    man = json.load(open(os.path.join(REPO, C.DATA_REL, "manifest.json")))
    assert not any(k.endswith("_dcr") for k in man["datasets"])
    assert runner.STAGES == ["train", "sample", "eval_main", "eval_quality", "eval_tabpc"]
    with pytest.raises(ValueError):
        evaluate.run("dcr", str(tmp_path), "tabdiff_adult")
    sys.path.insert(0, os.path.join(REPO, "analysis", "tabdiff"))
    import compare
    assert "dcr" not in [m[0] for m in compare.RA_METRICS]
    for f in glob.glob(os.path.join(REPO, "hpc_transfer", "jobs_tabdiff_gnn", "*.sh")):
        assert "eval_dcr" not in open(f).read() and "_dcr" not in re.sub(r"#.*", "", open(f).read())


def test_diabetes_data_is_tabdiffs_processing():
    f = C.data_files("tabdiff_diabetes")
    info = json.load(open(f["info"]))
    assert info["task_type"] == "binclass" and info["train_num"] == 61059 and info["test_num"] == 20354
    assert f["val"] is not None                       # TabDiff's MLE uses it -> so do we
    man = json.load(open(os.path.join(REPO, C.DATA_REL, "manifest.json")))
    for name, h in man["datasets"]["tabdiff_diabetes"]["files_sha256_16"].items():
        assert C.sha16(os.path.join(f["dir"], name)) == h


# ---------------------------------------------------------------------------
# Denoisers
# ---------------------------------------------------------------------------
def _shapes(target):
    import params
    return params.shapes(target)


def _unimod_params(target):
    from entry import load_raw_config
    d_num, cats, _ = _shapes(target)
    raw = load_raw_config()
    return dict(raw["unimodmlp_params"], d_numerical=d_num, categories=[k + 1 for k in cats]), raw


def _batch(d_num, cats_m, B=64, seed=1):
    import torch
    g = torch.Generator().manual_seed(seed)
    x_num = torch.randn(B, d_num, generator=g)
    oh = []
    for K in cats_m:                                   # include [MASK] (= index K-1) states
        idx = torch.randint(0, K, (B,), generator=g)
        oh.append(torch.nn.functional.one_hot(idx, K).float())
    x_cat = torch.cat(oh, 1) if oh else torch.zeros(B, 0)
    t = torch.rand(B, generator=g)
    sigma = torch.rand(B, d_num, generator=g) * 5 + 0.01
    return x_num, x_cat, t, sigma


def test_transformer_wrapper_is_bit_identical_to_tabdiff():
    _need_tabdiff()
    import torch
    from denoiser import build_denoiser
    from tabdiff.modules.main_modules import Model, UniModMLP
    up, raw = _unimod_params("tabdiff_shoppers")
    torch.manual_seed(0)
    ours = build_denoiser("transformer", up)
    torch.manual_seed(0)
    theirs = UniModMLP(**up)
    assert type(ours) is UniModMLP
    so, st = ours.state_dict(), theirs.state_dict()
    assert so.keys() == st.keys() and all(torch.equal(so[k], st[k]) for k in so)
    x_num, x_cat, t, sigma = _batch(up["d_numerical"], up["categories"])
    c = torch.log(torch.rand(len(t)) * 80 + 0.002) / 4
    a, b = ours(x_num, x_cat, c), theirs(x_num, x_cat, c)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    ma, mb = Model(ours, **raw["diffusion_params"]["edm_params"]), Model(theirs, **raw["diffusion_params"]["edm_params"])
    da, db = ma(x_num, x_cat, t, sigma), mb(x_num, x_cat, t, sigma)
    assert torch.equal(da[0], db[0]) and torch.equal(da[1], db[1])


def test_entry_builds_the_model_tabdiff_main_builds(tmp_path, monkeypatch):
    """Run tabdiff/main.py:main() (train mode, --debug so nothing is written into the
    submodule) up to Trainer construction, capture its diffusion; build ours with
    the same seed; same raw_config, schedules and bit-identical parameters."""
    _need_tabdiff()
    pytest.importorskip("sdmetrics")
    import pickle
    import torch
    import entry
    target, name = "tabdiff_shoppers", "shoppers"
    f = C.data_files(target)
    os.makedirs(tmp_path / "data")
    os.symlink(f["dir"], tmp_path / "data" / name)
    os.makedirs(tmp_path / "synthetic" / name)
    for k in ("real", "test"):
        os.symlink(f[k], tmp_path / "synthetic" / name / f"{k}.csv")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kw: types.SimpleNamespace(define_metric=lambda *a, **k: None,
                                                         log=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    entry._tabdiff_imports()
    import tabdiff.main as TM
    captured = {}

    class FakeTrainer:
        def __init__(self, diffusion, *a, **kw):
            captured["diffusion"], captured["kw"] = diffusion, kw

        def run_loop(self):
            captured["ran"] = True

    monkeypatch.setattr(TM, "Trainer", FakeTrainer)
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(dataname=name, mode="train", method="tabdiff", gpu=-1, debug=True,
                                 no_wandb=True, exp_name=None, deterministic=False, y_only=False,
                                 non_learnable_schedule=False, num_samples_to_generate=None,
                                 ckpt_path=None, report=False, num_runs=20, impute=False,
                                 device="cpu")
    torch.manual_seed(0)
    TM.main(args)
    theirs = captured["diffusion"]
    their_cfg = pickle.load(open(tmp_path / "debug" / "ckpt" / "config.pkl", "rb"))
    assert captured["ran"]

    spec = entry.spec_for(target, "transformer", 0)
    raw = entry.load_raw_config()
    torch.manual_seed(0)
    trainer, ours, backbone, raw = entry.build(spec, raw, "cpu", "train", str(tmp_path / "ours"))
    so, st = ours.state_dict(), theirs.state_dict()
    assert so.keys() == st.keys()
    assert all(torch.equal(so[k], st[k]) for k in so), [k for k in so if not torch.equal(so[k], st[k])]
    assert type(ours._denoise_fn.denoise_fn_D.denoise_fn_F) is type(theirs._denoise_fn.denoise_fn_D.denoise_fn_F)
    assert ours.scheduler == theirs.scheduler == "power_mean_per_column"
    assert ours.cat_scheduler == theirs.cat_scheduler == "log_linear_per_column"
    assert raw["unimodmlp_params"] == their_cfg["unimodmlp_params"]
    debug_keys = {"num_timesteps"}                     # changed by --debug in theirs only
    assert {k: v for k, v in raw["diffusion_params"].items() if k not in debug_keys} == \
           {k: v for k, v in their_cfg["diffusion_params"].items() if k not in debug_keys}
    for k in ("steps", "lr", "weight_decay", "ema_decay", "batch_size", "lr_scheduler", "factor",
              "reduce_lr_patience", "closs_weight_schedule", "c_lambda", "d_lambda"):
        assert raw["train"]["main"][k] == their_cfg["train"]["main"][k], k
    # trainer kwargs: ours differs only in the documented check_val_every deviation
    assert trainer.check_val_every == raw["train"]["main"]["steps"] + 1


@pytest.mark.parametrize("ds", C.ORDER)
def test_attngnn_shapes_dtypes_gradients(ds):
    _need_tabdiff()
    import torch
    from denoiser import AttnGNNDenoiser, build_denoiser
    from tabdiff.modules.main_modules import Model
    up, raw = _unimod_params(C.target_name(ds))
    g, _ = C.gnn_config(ds)
    torch.manual_seed(0)
    m = build_denoiser("attngnn", up, {**g, **C.GNN_FIXED}, dim_t=C.DIM_T)
    assert isinstance(m, AttnGNNDenoiser)
    x_num, x_cat, t, sigma = _batch(up["d_numerical"], up["categories"], B=32)
    c = torch.log(torch.rand(32) * 80 + 0.002) / 4
    F_x, logits = m(x_num, x_cat, c)
    assert F_x.shape == (32, up["d_numerical"]) and F_x.dtype == torch.float32
    assert logits.shape == x_cat.shape and logits.dtype == torch.float32
    model = Model(m, **raw["diffusion_params"]["edm_params"])
    D_x, lg = model(x_num, x_cat, t, sigma)               # Precond asserts float32
    (D_x.pow(2).mean() + lg.pow(2).mean()).backward()
    assert all(p.grad is not None for n, p in m.named_parameters()
               if not n.startswith("tokenizer.weight")), "every block receives gradient"


def test_attngnn_uses_the_unchanged_modules_and_no_global_mixer():
    _need_tabdiff()
    import torch.nn as nn
    from denoiser import build_denoiser
    from tab_ddpm.graph_builder import DynamicAdjacency
    from tab_ddpm.graph_denoiser import AttnGNNBlock
    from tab_ddpm.gnn_layers import GATv2Layer
    from tabdiff.modules.main_modules import MLPDiffusion
    from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer
    up, _ = _unimod_params("tabdiff_default")
    g, _ = C.gnn_config("default")
    m = build_denoiser("attngnn", up, {**g, **C.GNN_FIXED})
    assert type(m.tokenizer) is Tokenizer and type(m.detokenizer) is Reconstructor
    assert len(m.blocks) == g["n_layers"]
    for b in m.blocks:
        assert type(b) is AttnGNNBlock and b.attn is None
        assert type(b.dynamic_adj) is DynamicAdjacency
        assert b.dynamic_adj.topk_mode == "fixed" and b.dynamic_adj.top_k == g["top_k"]
        assert all(type(x) is GATv2Layer for x in b.gnns)
    assert not any(isinstance(x, (Transformer, MLPDiffusion, nn.MultiheadAttention)) for x in m.modules())


def test_param_counts_d256_and_tabdiff():
    _need_tabdiff()
    import params
    want = {"magic": (10589231, 3478403), "shoppers": (10647279, 2729675), "adult": (10622989, 2746353),
            "default": (10696637, 5237465), "beijing": (10598109, 5226705), "news": (10893027, 5222031),
            "diabetes": (10822529, 3843133)}
    for ds, (tf, gnn) in want.items():
        r = params.counts(C.target_name(ds))
        assert (r["transformer"], r["attngnn"]) == (tf, gnn), ds
        assert 10_500_000 < r["transformer"] < 10_900_000 and r["attngnn"] < r["transformer"]


def test_shared_noise_embedding_is_the_same_on_every_node():
    _need_tabdiff()
    import torch
    from denoiser import AttnGNNDenoiser
    m = AttnGNNDenoiser(3, [3, 4], d_model=16, n_layers=0, gnn_type="gcn", n_heads=1, top_k=2, dim_t=8)
    x_num, x_cat, _, _ = _batch(3, [3, 4], B=5)
    c = torch.randn(5)
    e = m.tokenizer(x_num, x_cat)[:, 1:]
    want = m.detokenizer(e + m._embed(c)[:, None, :])
    got = m(x_num, x_cat, c)
    assert torch.allclose(got[0], want[0]) and torch.allclose(got[1], torch.cat(want[1], -1))


def test_per_column_noise_gives_each_node_its_own_level():
    _need_tabdiff()
    import torch
    from denoiser import AttnGNNDenoiser, t_from_c_noise
    from tabdiff.models.noise_schedule import LogLinearNoise_PerColumn, PowerMeanNoise_PerColumn
    # c_noise -> t inverts Precond's sigma_cond schedule (main_modules.py:120,128)
    t = torch.linspace(0.01, 0.99, 50)
    c = ((0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))).pow(7)).log() / 4
    assert torch.allclose(t_from_c_noise(c), t, atol=1e-4)
    class Diffusion:                                   # stands in for UnifiedCtimeDiffusion
        pass
    diff = Diffusion()
    diff.num_schedule = PowerMeanNoise_PerColumn(3, rho_init=7, rho_offset=5)
    diff.cat_schedule = LogLinearNoise_PerColumn(2, k_init=-6, k_offset=1)
    with torch.no_grad():
        diff.num_schedule.rho_raw.copy_(torch.tensor([1.0, 5.0, 9.0]))
        diff.cat_schedule.k_raw.copy_(torch.tensor([-6.0, 2.0]))
    m = AttnGNNDenoiser(3, [3, 4], d_model=16, n_layers=1, gnn_type="gcn", n_heads=1, top_k=2, dim_t=8,
                        node_noise="per_column")
    x_num, x_cat, _, _ = _batch(3, [3, 4], B=4)
    cn = c[[10, 20, 30, 40]]
    with pytest.raises(RuntimeError, match="attach_diffusion"):
        m(x_num, x_cat, cn)
    m.attach_diffusion(diff)
    lv = m.node_noise_levels(cn)
    assert lv.shape == (4, 5)
    tt = t_from_c_noise(cn)[:, None]
    want = torch.cat([diff.num_schedule.total_noise(tt), diff.cat_schedule.total_noise(tt)], 1).log() / 4
    assert torch.allclose(lv, want, atol=1e-5)
    assert lv[:, :3].std(dim=1).min() > 0 and lv[:, 3:].std(dim=1).min() > 0    # nodes differ
    assert not lv.requires_grad
    m(x_num, x_cat, cn)
    ema = copy.deepcopy(m)                               # Trainer's EMA copy keeps the referent
    assert ema._diffusion_ref() is diff
    diff.num_schedule = PowerMeanNoise_PerColumn(3, rho_init=0, rho_offset=2)      # to_ema_model swap
    assert not torch.allclose(ema.node_noise_levels(cn)[:, :3], lv[:, :3])
    shared = AttnGNNDenoiser(3, [3, 4], d_model=16, n_layers=1, gnn_type="gcn", n_heads=1, top_k=2, dim_t=8)
    assert shared.node_noise == "shared" and C.NODE_NOISE == "shared"


def test_reserved_hooks_are_off_and_unimplemented():
    _need_tabdiff()
    from denoiser import AttnGNNDenoiser, build_denoiser
    runner = _load_route_a("runner")
    with pytest.raises(NotImplementedError):
        AttnGNNDenoiser(2, [3], 8, 1, "gcn", 1, 1, geo_loss=True)
    with pytest.raises(NotImplementedError):
        AttnGNNDenoiser(2, [3], 8, 1, "gcn", 1, 1, geo_edge_bias=True)
    a = runner.make_parser().parse_args(["--dataset", "magic", "--denoiser", "attngnn", "--exp_root", "x"])
    assert a.geo_loss is False and a.geo_edge_bias is False and a.node_noise == "shared"
    with pytest.raises(ValueError):
        build_denoiser("transformer", {}, node_noise="per_column")


# ---------------------------------------------------------------------------
# Categorical input scale (the cat_input lesson)
# ---------------------------------------------------------------------------
def test_categorical_tokens_on_the_scale_of_the_noise_embedding():
    _need_tabdiff()
    import torch
    from denoiser import build_denoiser
    up, _ = _unimod_params("tabdiff_adult")
    g, _ = C.gnn_config("adult")
    torch.manual_seed(0)
    m = build_denoiser("attngnn", up, {**g, **C.GNN_FIXED}, dim_t=C.DIM_T)
    x_num, x_cat, _, _ = _batch(up["d_numerical"], up["categories"], B=2048)
    c = torch.log(torch.rand(2048) * 80 + 0.002) / 4
    with torch.no_grad():
        tok = m.tokenizer(x_num, x_cat)[:, 1:]
        cat_norm = tok[:, up["d_numerical"]:].norm(dim=-1).mean().item()
        noise_norm = m._embed(c).norm(dim=-1).mean().item()
    assert 0.2 < cat_norm / noise_norm < 5, (cat_norm, noise_norm)


# ---------------------------------------------------------------------------
# Runner (stages, resume, refusal) with a fake executor
# ---------------------------------------------------------------------------
class FakeExec:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def run(self, cmd):
        stage = self._stage(cmd)
        self.calls.append(stage)
        if stage == self.fail_on:
            return 1
        run = cmd[cmd.index("--run_dir") + 1]
        n = next((int(cmd[cmd.index(f) + 1]) for f in ("--n_runs", "--n_tables") if f in cmd), 0)
        getattr(self, "_" + stage.split(":")[0])(cmd, run, n, stage)
        return 0

    @staticmethod
    def _stage(cmd):
        script = os.path.basename(cmd[1])
        if script == "entry.py":
            return cmd[2]
        if script == "tabpc_eval.py":
            return "tabpc"
        return "eval:" + cmd[2]

    def _train(self, cmd, run, n, _):
        import entry
        tgt = cmd[cmd.index("--target") + 1]
        spec = entry.spec_for(tgt, cmd[cmd.index("--denoiser") + 1], int(cmd[cmd.index("--seed") + 1]),
                              node_noise=cmd[cmd.index("--node_noise") + 1],
                              ablation=cmd[cmd.index("--ablation") + 1] if "--ablation" in cmd else None)
        spec["smoke_epochs"] = None
        os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
        open(os.path.join(run, "ckpt", "best_ema_model_1.0_5000.pt"), "w").close()
        json.dump(spec, open(os.path.join(run, "route_a_spec.json"), "w"))
        json.dump({"completed": True, "checkpoint": "best_ema_model_1.0_5000.pt"},
                  open(os.path.join(run, "train_record.json"), "w"))

    def _sample(self, cmd, run, n, _):
        os.makedirs(os.path.join(run, "samples"), exist_ok=True)
        for k in range(n):
            open(os.path.join(run, "samples", f"samples_{k}.csv"), "w").write("a\n1\n")
        json.dump({"checkpoint": "best_ema_model_1.0_5000.pt", "n_tables": n},
                  open(os.path.join(run, "sample_record.json"), "w"))

    def _write(self, path, cols, n):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["", *cols])
            for i in range(n):
                w.writerow([i] + [0.5] * len(cols))

    def _eval(self, cmd, run, n, stage):
        step = stage.split(":")[1]
        d = os.path.join(run, "tabdiff_eval")
        os.makedirs(d, exist_ok=True)
        if step == "main":
            self._write(os.path.join(d, "all_results.csv"), ["shape", "trend", "mle", "c2st"], n)
            with open(os.path.join(d, "mle2.csv"), "w") as fh:
                fh.write("run,auroc,f1\n" + "".join(f"{i},0.9,0.6\n" for i in range(n)))
        else:
            self._write(os.path.join(d, "quality.csv"), ["alpha", "beta"], n)
        mp = os.path.join(d, "eval_meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        meta[step] = {"n_runs": n}
        json.dump(meta, open(mp, "w"))

    def _tabpc(self, cmd, run, n, _):
        d = os.path.join(run, "tabpc_eval")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "tabpc_results.csv"), "w") as fh:
            fh.write("run,wnmis,c2st_xgb\n" + "".join(f"{i},0.98,0.7\n" for i in range(n)))
        json.dump({"n_runs": n}, open(os.path.join(d, "tabpc_meta.json"), "w"))


def _args(tmp_path, *extra):
    runner = _load_route_a("runner")
    return runner.make_parser().parse_args(
        ["--exp_root", str(tmp_path), "--eval_python", "py", "--synthcity_python", "py",
         "--tabpc_python", "py", "--train_python", "py", "--n_runs", "3", *extra])


def test_runner_order_then_skip(tmp_path):
    runner = _load_route_a("runner")
    ex = FakeExec()
    runner.run_job(_args(tmp_path, "--dataset", "shoppers", "--denoiser", "attngnn"), ex)
    assert ex.calls == ["train", "sample", "eval:main", "eval:quality", "tabpc"]
    ex2 = FakeExec()
    runner.run_job(_args(tmp_path, "--dataset", "shoppers", "--denoiser", "attngnn"), ex2)
    assert ex2.calls == []
    meta = json.load(open(os.path.join(C.run_dir(str(tmp_path), "tabdiff_shoppers", "attngnn", 0),
                                       "run_meta.json")))
    for k in ("denoiser", "variant", "gnn", "dim_t", "node_noise", "git_commit", "tabdiff", "tabpc",
              "train_seed", "deviations_from_tabdiff", "protocol"):
        assert k in meta, k
    assert meta["variant"] == "V2" and meta["tabdiff"]["commit"] == C.TABDIFF_COMMIT


def test_runner_resumes_at_the_failed_stage(tmp_path):
    runner = _load_route_a("runner")
    with pytest.raises(RuntimeError):
        runner.run_job(_args(tmp_path, "--dataset", "magic", "--denoiser", "transformer"),
                       FakeExec(fail_on="eval:quality"))
    ex = FakeExec()
    runner.run_job(_args(tmp_path, "--dataset", "magic", "--denoiser", "transformer"), ex)
    assert ex.calls == ["eval:quality", "tabpc"]


def test_runner_refuses_a_differently_trained_model(tmp_path):
    runner = _load_route_a("runner")
    runner.run_job(_args(tmp_path, "--dataset", "magic", "--denoiser", "attngnn"), FakeExec())
    run = C.run_dir(str(tmp_path), "tabdiff_magic", "attngnn", 0)
    spec = json.load(open(os.path.join(run, "route_a_spec.json")))
    spec["gnn"]["top_k"] = 99
    json.dump(spec, open(os.path.join(run, "route_a_spec.json"), "w"))
    with pytest.raises(runner.SpecMismatch):
        runner.run_job(_args(tmp_path, "--dataset", "magic", "--denoiser", "attngnn"), FakeExec())


def test_runner_chained_stages(tmp_path):
    runner = _load_route_a("runner")
    ex = FakeExec()
    runner.run_job(_args(tmp_path, "--dataset", "news", "--denoiser", "attngnn", "--stages", "train"), ex)
    assert ex.calls == ["train"]
    with pytest.raises(SystemExit, match="earlier job"):
        runner.run_job(_args(tmp_path, "--dataset", "adult", "--denoiser", "attngnn",
                             "--stages", "sample,eval_main,eval_quality,eval_tabpc"), FakeExec())
    ex = FakeExec()
    runner.run_job(_args(tmp_path, "--dataset", "news", "--denoiser", "attngnn",
                         "--stages", "sample,eval_main,eval_quality,eval_tabpc"), ex)
    assert ex.calls == ["sample", "eval:main", "eval:quality", "tabpc"]


def test_runner_arm_guards(tmp_path):
    runner = _load_route_a("runner")
    with pytest.raises(SystemExit):                     # no d_model override any more
        runner.make_parser().parse_args(["--dataset", "magic", "--denoiser", "attngnn", "--exp_root", "x",
                                         "--d_model", "384"])


@pytest.mark.parametrize("arm", [("diabetes", "attngnn"), ("magic", "transformer"), ("news", "attngnn")])
def test_dry_run_builds_the_denoiser(tmp_path, capsys, arm):
    _need_tabdiff()
    runner = _load_route_a("runner")
    ds, den = arm
    a = runner.make_parser().parse_args(["--dataset", ds, "--denoiser", den, "--exp_root", str(tmp_path),
                                         "--dry_run"])
    runner.run_job(a)
    out = capsys.readouterr().out
    assert "forward: ok" in out and ("AttnGNNDenoiser" if den == "attngnn" else "UniModMLP") in out
    if ds == "diabetes":
        assert "our choice" in out
    assert not os.path.exists(os.path.join(str(tmp_path), "exp_tabdiff_gnn"))   # dry run writes nothing


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def test_mle2_is_read_from_tabdiffs_mle_details():
    import tabdiff_eval as TE
    cls = {"best_auroc_scores": {"XGBClassifier": {"roc_auc": 0.91}},
           "best_f1_scores": {"XGBClassifier": {"binary_f1": 0.66}}}
    reg = {"best_rmse_scores": {"XGBRegressor": {"RMSE": 0.55}},
           "best_r2_scores": {"XGBRegressor": {"r2": 0.70}}}
    assert TE.mle2_from_extras(cls, "binclass") == {"auroc": 0.91, "f1": 0.66}
    assert TE.mle2_from_extras(reg, "regression") == {"rmse": 0.55, "r2": 0.70}


def test_tabpc_keys_are_the_article_keys():
    src = open(os.path.join(REPO, C.TABPC_REL, "scripts", "update_article_tables.py")).read()
    assert "nmi_l1_weighted_complement" in src and "xgboost(C2ST)" in src
    assert C.TABPC_KEYS["wnmis"][1] == "nmi_l1_weighted_complement"
    assert C.TABPC_KEYS["c2st_xgb"][1] == "xgboost (C2ST)"


# ---------------------------------------------------------------------------
# Jobs and analysis
# ---------------------------------------------------------------------------
JOBS = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff_gnn")


def _over_cap(ds):
    mj = _load_route_a("make_jobs")
    return mj.over_cap("gnn", ds)


REF_JOB = os.path.join(REPO, "hpc_transfer", "jobs", "king_transfer.sh")


def _blocks(text):
    """The verbatim blocks shared with hpc_transfer/jobs (everything up to the Paths section)."""
    return text.split("# Paths. Defined once")[0]


def test_job_scripts_follow_hpc_transfer_jobs():
    files = sorted(f for f in glob.glob(os.path.join(JOBS, "*.sh")) if not f.endswith("common_tabdiff_gnn.sh"))
    names = {os.path.basename(f)[:-3] for f in files}
    want = {f"{ds}_tabdiff_gnn" for ds in C.ORDER if ds != "news"} | {"news_tabdiff_gnn_train",
                                                                      "news_tabdiff_gnn_sample_eval",
                                                                      "beijing_tabdiff_gnn_ablation_topk_dense"}
    assert names == want                                     # one script per dataset, no per-seed copies
    ref = open(REF_JOB).read()
    norm = lambda t: re.sub(r"(--job-name=|jobs_out/)\S+|--time=\S+", "X", t)
    for f in files:
        t = open(f).read()
        stem = os.path.basename(f)[:-3]
        # header + TODO(GPU) block + account + template block: as in hpc_transfer/jobs, except
        # the job name/time and our 3-line note on per-seed logs
        ours = norm(_blocks(t)).replace(
            " For SEED=1 / 2\n# pass --job-name/--output/--error on the sbatch command line (see\n"
            "# hpc_transfer/submit_tabdiff_gnn.sh) so the seeds do not share one log.", "")
        assert ours == norm(_blocks(ref)), stem
        assert f"#SBATCH --job-name={stem}\n" in t and f"jobs_out/{stem}.out" in t and f"jobs_out/{stem}.err" in t
        assert "#SBATCH --mem=120G" in t and "#SBATCH --partition=gpu" in t
        assert "\n#SBATCH --gres" not in t and "##SBATCH --gres=gpu:1" in t     # TODO(GPU), commented
        h = int(re.search(r"#SBATCH --time=(\d+):00:00", t).group(1))
        assert 2 <= h <= 48
        for var in ("HPC_ROOT", "REPO_ROOT", "PYTHON_BIN", "TRANSFER_ROOT", "DEVICE", "SEED", "STAGES"):
            assert re.search(rf'^{var}="\${{{var}:-', t, re.M), (stem, var)  # overridable from the environment
        assert 'cd "${REPO_ROOT}"' in t and 'SEED="${SEED:-0}"' in t and 'DENOISER="attngnn"' in t
        assert t.rstrip().endswith("run_route_a") and "common_tabdiff_gnn.sh" in t
        stages = re.search(r'STAGES="\$\{STAGES:-([^}]*)\}"', t).group(1).split(",")
        assert stages in (["train", "sample", "eval_main", "eval_quality", "eval_tabpc"], ["train"],
                          ["sample", "eval_main", "eval_quality", "eval_tabpc"])
        assert "d256" in t


def test_over_cap_jobs_are_flagged_and_blocked():
    sub = open(os.path.join(REPO, "hpc_transfer", "submit_tabdiff_gnn.sh")).read()
    for ds in C.ORDER:
        first = sorted(glob.glob(os.path.join(JOBS, f"{ds}_tabdiff_gnn*.sh")))[-1]
        txt = open(first).read()
        lines = [ln for ln in sub.split("# === training seed 1")[0].splitlines()
                 if f"/{ds}_tabdiff_gnn" in ln and "sbatch" in ln]
        assert lines, ds
        if _over_cap(ds):
            assert "OVER_CAP" in txt and all(ln.startswith("### ") for ln in lines)
        else:
            assert "OVER_CAP" not in txt and all(ln.startswith("# ") for ln in lines)


def test_news_is_a_resumable_training_chain():
    train = open(os.path.join(JOBS, "news_tabdiff_gnn_train.sh")).read()
    post = open(os.path.join(JOBS, "news_tabdiff_gnn_sample_eval.sh")).read()
    assert 'STAGES="${STAGES:-train}"' in train and 'TRAIN_BUDGET_H="${TRAIN_BUDGET_H:-46.5}"' in train
    assert "#SBATCH --time=48:00:00" in train and "resumable copy of TabDiff's Trainer.run_loop" in train
    assert 'STAGES="${STAGES:-sample,eval_main,eval_quality,eval_tabpc}"' in post
    assert 'TRAIN_BUDGET_H="${TRAIN_BUDGET_H:-}"' in post
    for ds in C.ORDER:                                          # only news uses the resumable copy
        for f in glob.glob(os.path.join(JOBS, f"{ds}_tabdiff_gnn*.sh")):
            txt = open(f).read()
            assert ("resumable copy" in txt) == (ds == "news")
            if ds != "news":
                assert "TabDiff's Trainer, unchanged" in txt and 'TRAIN_BUDGET_H="${TRAIN_BUDGET_H:-}"' in txt
    sub = open(os.path.join(REPO, "hpc_transfer", "submit_tabdiff_gnn.sh")).read()
    seed0 = sub.split("# === training seed 1")[0]
    news = [ln for ln in seed0.splitlines() if "news_tabdiff_gnn" in ln and "sbatch" in ln]
    mj = _load_route_a("make_jobs")
    parts = mj.train_parts("gnn", "news")
    assert len(news) == parts + 1
    assert all("news_tabdiff_gnn_train.sh" in ln for ln in news[:parts]) and "sample_eval.sh" in news[-1]
    assert all("--dependency=afterok:${jid}" in ln for ln in news[1:])
    eff = [re.search(r"--job-name=(\S+)", ln).group(1) if "--job-name" in ln
           else re.search(r"jobs_tabdiff_gnn/(\w+)\.sh", ln).group(1) for ln in news]
    assert len(set(eff)) == len(news)                            # every part has its own job name / log
    assert "### " not in sub


def test_submission_loop_commented_and_ordered():
    s = open(os.path.join(REPO, "hpc_transfer", "submit_tabdiff_gnn.sh")).read()
    active = [ln for ln in s.splitlines() if "sbatch" in ln and not ln.lstrip().startswith("#")]
    assert active == []
    assert "exit 0" in s and 'export GRES=' in s and all('--gres="${GRES}"' in ln for ln in s.splitlines()
                                                         if "sbatch --" in ln and "hpc_transfer/jobs_tabdiff_gnn/" in ln)
    seed0 = s.split("# === training seed 1")[0]
    order = []
    for m in re.finditer(r"jobs_tabdiff_gnn/(\w+?)_tabdiff_gnn", seed0):
        if m.group(1) not in order:
            order.append(m.group(1))
    assert order == ["magic", "shoppers", "adult", "beijing", "default", "news", "diabetes"]
    later = s.split("# === training seed 1")[1]
    for ln in later.splitlines():
        if "sbatch" in ln:
            assert ln.startswith("## ") and re.search(r"SEED=[12] sbatch", ln) and "--job-name=" in ln


def test_route_a_report_is_reported_vs_computed(tmp_path):
    sys.path.insert(0, os.path.join(REPO, "analysis", "tabdiff"))
    import compare
    import pandas as pd
    rows = [dict(dataset="magic", metric="shape", method=compare.RA_ARM[1], kind="computed", mean=1.1,
                 std=0.05, std_seed_means=float("nan"), n=20, n_seeds=1, source="x")]
    md = compare.ra_markdown(pd.DataFrame(rows), {}, compare.ra_static_params(), str(tmp_path))
    assert "## Reference" in md and "reported, not reproduced" in md and "bit-identical" in md
    assert "GATD" in md and "primary" in md and "secondary" in md
    assert "REPRODUCTION" not in md and "repro check" not in md
    hdr = [ln for ln in md.splitlines() if ln.startswith("| dataset | reported:")]
    assert hdr and all("computed: TabDiff + AttnGNN (d_model 256)" in h for h in hdr)
    assert "1.10 ± 0.05 (n=20, s=1)" in md
    assert "| magic | 3,478,403 | 10,589,231 | 0.328 |" in md


def test_gatd_transcription_spot_checks():
    sys.path.insert(0, os.path.join(REPO, "analysis", "tabdiff"))
    import reported_gatd as RG
    assert RG.cell("shape", "magic") == (0.811, 0.108)
    assert RG.cell("trend", "default") == (4.71, 1.58)
    assert RG.cell("mle1", "news") == (0.0151, 0.0537)
    assert RG.cell("mle2", "beijing") == (0.564, 0.0131)
    assert set(RG.TABDIFF["shape"]) == set(C.ORDER)



# ---------------------------------------------------------------------------
# Resumable training (news): verbatim loop + bit-identical resume
# ---------------------------------------------------------------------------
def test_resumable_run_loop_is_tabdiffs_verbatim():
    _need_tabdiff()
    import ast
    import inspect
    import textwrap
    resumable = _load_route_a("resumable")
    from tabdiff.trainer import Trainer
    ours = inspect.getsource(resumable.ResumableTrainer.run_loop)
    theirs = inspect.getsource(Trainer.run_loop)
    stripped = re.sub(r"[ ]*# >>> ROUTE-A RESUME HOOK.*?# <<< ROUTE-A RESUME HOOK\n", "", ours, flags=re.S)
    assert stripped == theirs
    assert ours.count("# >>> ROUTE-A RESUME HOOK") == 2
    # the TabDiff source we copied is the pinned one
    assert C.sha16(os.path.join(TABDIFF, "tabdiff", "trainer.py")) == C.S.PINNED_TABDIFF_FILES["tabdiff/trainer.py"]
    # every other method is TabDiff's own (inherited, not overridden)
    overridden = {n for n, v in vars(resumable.ResumableTrainer).items() if callable(v)}
    assert overridden <= {"__init__", "run_loop", "_rng_state", "_save_run_state", "_restore_run_state",
                          "_after_epoch"}


def _magic_subset(tmp_path, rows=512):
    """data dir with the first `rows` training rows of magic (TabDiff layout)."""
    import numpy as np
    import shutil
    src = C.data_files("tabdiff_magic")
    d = tmp_path / "magic_subset"
    d.mkdir()
    for f in os.listdir(src["dir"]):
        p = os.path.join(src["dir"], f)
        if f.endswith("_train.npy"):
            np.save(d / f, np.load(p, allow_pickle=True)[:rows])
        elif f.endswith(".npy") or f in ("info.json", "test.csv"):
            shutil.copy(p, d / f)
    with open(src["real"]) as fh, open(d / "real.csv", "w") as out:
        for i, ln in enumerate(fh):
            if i > rows:
                break
            out.write(ln)
    return {"dir": str(d), "info": str(d / "info.json"), "real": str(d / "real.csv"),
            "test": str(d / "test.csv"), "val": None}


def _train(run_dir, files, steps, trainer, stop_after=None, save_every=10, seed=0, scramble=False,
           device="cpu"):
    import random
    import numpy as np
    import torch
    import entry
    resumable = _load_route_a("resumable")
    spec = entry.spec_for("tabdiff_magic", "attngnn", seed)
    spec["gnn"] = {**spec["gnn"], "d_model": 16, "n_layers": 1, "n_heads": 4}       # tiny, same type (gatv2)
    raw = entry.load_raw_config()
    raw["train"]["main"]["steps"] = steps
    entry.seed_everything(seed)
    kwargs, cls = None, None
    if trainer == "resumable":
        cls = resumable.ResumableTrainer
        kwargs = {"resume_path": os.path.join(run_dir, "ckpt", resumable.RESUME_FILE),
                  "save_every": save_every, "stop_after_epoch": stop_after}
    if scramble:                                       # a resumed process starts from unrelated RNG states
        torch.manual_seed(12345); np.random.seed(54321); random.seed(999)
    tr, diff, _, _ = entry.build(spec, raw, device, "train", run_dir, trainer_cls=cls, trainer_kwargs=kwargs,
                                 files=files, num_workers=0,
                                 log_append=bool(kwargs and os.path.exists(kwargs["resume_path"])))
    tr.run_loop()
    tr.logger.close()
    return tr


def _state(tr, run_dir):
    import glob as g
    import torch
    out = {"curr_epoch": tr.curr_epoch,
           "model": tr.diffusion._denoise_fn.state_dict(), "num_schedule": tr.diffusion.num_schedule.state_dict(),
           "cat_schedule": tr.diffusion.cat_schedule.state_dict(), "ema": tr.ema_model.state_dict(),
           "ema_num": tr.ema_num_schedule.state_dict(), "ema_cat": tr.ema_cat_schedule.state_dict(),
           "opt": tr.optimizer.state_dict()["state"], "lr": [p["lr"] for p in tr.optimizer.param_groups],
           "sched": {k: v for k, v in tr.scheduler.state_dict().items() if k != "optimizer"}}
    out["best_files"] = {os.path.basename(f): torch.load(f, map_location="cpu", weights_only=False)
                         for f in sorted(g.glob(os.path.join(run_dir, "ckpt", "best_*model_*")))}
    return out


def _assert_identical(a, b, path=""):
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, sorted(a), sorted(b))
        for k in a:
            _assert_identical(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_identical(x, y, f"{path}[{i}]")
    else:
        assert a == b or (a != a and b != b), (path, a, b)            # NaN-tolerant equality


def _resume_case(tmp_path, steps, stop_at, save_every):
    _need_tabdiff()
    files = _magic_subset(tmp_path)
    runs = {k: str(tmp_path / k) for k in ("tabdiff", "resumable", "interrupted")}
    a = _state(_train(runs["tabdiff"], files, steps, "tabdiff"), runs["tabdiff"])
    b = _state(_train(runs["resumable"], files, steps, "resumable", save_every=save_every), runs["resumable"])
    first = _train(runs["interrupted"], files, steps, "resumable", stop_after=stop_at, save_every=save_every)
    assert first.paused_at == stop_at and first.curr_epoch == stop_at
    second = _train(runs["interrupted"], files, steps, "resumable", save_every=save_every, scramble=True)
    assert second.resumed_from_epoch == stop_at and second.paused_at is None
    c = _state(second, runs["interrupted"])
    _assert_identical(b, a)        # the hooks change nothing (periodic saves included)
    _assert_identical(c, a)        # interrupted + resumed == uninterrupted == TabDiff's own Trainer
    return a


def test_resume_is_bit_identical(tmp_path):
    """CPU, 40 epochs, paused at 15 (between the periodic saves at 10 and 20): model, EMA,
    schedules, optimizer and LR-scheduler state identical to an uninterrupted run and to
    TabDiff's own Trainer."""
    a = _resume_case(tmp_path, steps=40, stop_at=15, save_every=10)
    assert a["curr_epoch"] == 40


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="slow (~10 min on CPU); set RUN_SLOW=1")
def test_resume_is_bit_identical_through_best_checkpoints(tmp_path):
    """Past TabDiff's epoch-4000 threshold, so best_model_* / best_ema_model_* are written:
    paused at 4004 (after some bests were saved), resumed to 4012 -- identical weights, EMA
    and the same best-checkpoint files (name = loss + epoch, and contents)."""
    a = _resume_case(tmp_path, steps=4012, stop_at=4004, save_every=1000)
    assert any(k.startswith("best_ema_model_") for k in a["best_files"])
    assert any(k.startswith("best_model_") for k in a["best_files"])


def test_runner_resumes_a_paused_training(tmp_path):
    runner = _load_route_a("runner")

    class PausingExec(FakeExec):
        def __init__(self):
            super().__init__()
            self.trains = 0

        def run(self, cmd):
            if self._stage(cmd) == "train" and self.trains == 0:
                self.trains += 1
                self.calls.append("train(paused)")
                run = cmd[cmd.index("--run_dir") + 1]
                os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
                open(os.path.join(run, "ckpt", "resume_state.pt"), "w").close()
                assert "--train_budget_h" in cmd                   # news gets its budget
                return runner.PAUSED_EXIT
            return super().run(cmd)

    ex = PausingExec()
    args = ["--dataset", "news", "--denoiser", "attngnn", "--stages", "train", "--train_budget_h", "46"]
    assert runner.run_job(_args(tmp_path, *args), ex) == 0                 # paused = clean exit
    run = C.run_dir(str(tmp_path), "tabdiff_news", "attngnn", 0)
    assert not os.path.exists(os.path.join(run, "stages", "train.done"))
    assert os.path.exists(os.path.join(run, "ckpt", "resume_state.pt"))    # kept for the next part
    runner.run_job(_args(tmp_path, *args), ex)
    assert ex.calls == ["train(paused)", "train"] and os.path.exists(os.path.join(run, "stages", "train.done"))
    meta = json.load(open(os.path.join(run, "run_meta.json")))
    assert meta["trainer"].startswith("resumable copy") and "test_resume_is_bit_identical" in meta["trainer_evidence"]
    assert any("news only" in d for d in meta["deviations_from_tabdiff"])
    other = C.run_dir(str(tmp_path), "tabdiff_magic", "attngnn", 0)
    runner.run_job(_args(tmp_path, "--dataset", "magic", "--denoiser", "attngnn"), FakeExec())
    m2 = json.load(open(os.path.join(other, "run_meta.json")))
    assert m2["trainer"] == "TabDiff's Trainer, unchanged" and m2["trainer_evidence"] is None
    assert C.TRAINER == {ds: ("resumable" if ds == "news" else "tabdiff") for ds in C.ORDER}


GPU_RESUME_FACTOR, GPU_RESUME_FLOOR = 10.0, 1e-6


@pytest.mark.skipif(os.environ.get("ROUTE_A_RESUME_DEVICE", "cpu") == "cpu",
                    reason="cluster check: ROUTE_A_RESUME_DEVICE=cuda:0")
def test_resume_on_gpu_within_run_to_run_tolerance(tmp_path):
    """On CUDA, TabDiff trains without deterministic algorithms, so even two uninterrupted
    runs need not be bit-identical. Tolerance used: the resumed run's max |difference| from an
    uninterrupted run (model, EMA, schedules) must be <= max(10 x the run-to-run max |difference|
    of two uninterrupted runs, 1e-6). Both numbers are printed for the record."""
    _need_tabdiff()
    import torch
    dev = os.environ["ROUTE_A_RESUME_DEVICE"]
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    files = _magic_subset(tmp_path)
    steps, stop = int(os.environ.get("ROUTE_A_RESUME_STEPS", 40)), int(os.environ.get("ROUTE_A_RESUME_STOP", 15))
    run = lambda name, **kw: _state(_train(str(tmp_path / name), files, steps, "resumable", device=dev, **kw),
                                    str(tmp_path / name))
    a1, a2 = run("a1"), run("a2")
    _train(str(tmp_path / "c"), files, steps, "resumable", stop_after=stop, device=dev)
    c = run("c", scramble=True)

    def maxdiff(x, y):
        out = 0.0
        for k in ("model", "ema", "num_schedule", "cat_schedule", "ema_num", "ema_cat"):
            for n in x[k]:
                out = max(out, (x[k][n].float() - y[k][n].float()).abs().max().item())
        return out
    run_to_run, resumed = maxdiff(a1, a2), maxdiff(a1, c)
    print(f"\n  GPU resume check ({dev}, {steps} epochs, paused at {stop}): run-to-run max|d| = {run_to_run:.3e}, "
          f"resumed vs uninterrupted max|d| = {resumed:.3e}")
    assert resumed <= max(GPU_RESUME_FACTOR * run_to_run, GPU_RESUME_FLOOR)
    assert sorted(a1["best_files"]) == sorted(c["best_files"]) or steps <= 4000


# ---------------------------------------------------------------------------
# Ablation (beijing, top_k = N-1) and the diagnosis helpers
# ---------------------------------------------------------------------------
def test_topk_dense_ablation_changes_only_top_k(tmp_path):
    main, _ = C.gnn_config("beijing")
    abl, src = C.gnn_config("beijing", ablation="topk_dense")
    assert C.n_nodes("beijing") == 12 and abl["top_k"] == 11 and main["top_k"] == 5
    assert {k: v for k, v in abl.items() if k != "top_k"} == {k: v for k, v in main.items() if k != "top_k"}
    assert "ABLATION" in src["top_k"]
    p_main = C.run_dir(str(tmp_path), "tabdiff_beijing", "attngnn", 0)
    p_abl = C.run_dir(str(tmp_path), "tabdiff_beijing", "attngnn", 0, ablation="topk_dense")
    assert "_topk11_" in p_abl and "_topk5_" in p_main and p_abl != p_main
    with pytest.raises(ValueError):
        C.gnn_config("magic", ablation="topk_dense")              # defined for beijing only
    with pytest.raises(ValueError):
        C.gnn_config("beijing", ablation="nope")


def test_ablation_run_meta_and_spec(tmp_path):
    runner = _load_route_a("runner")
    ex = FakeExec()
    runner.run_job(_args(tmp_path, "--dataset", "beijing", "--denoiser", "attngnn", "--ablation", "topk_dense"), ex)
    run = C.run_dir(str(tmp_path), "tabdiff_beijing", "attngnn", 0, ablation="topk_dense")
    meta = json.load(open(os.path.join(run, "run_meta.json")))
    assert meta["ablation"]["name"] == "topk_dense" and "not part of the main" in meta["ablation"]["note"]
    assert meta["ablation"]["main_run_dir"].endswith(os.path.join("v2_gatv2_layers3_dim256_topk5_heads8_dimt128_"
                                                                  "noise-shared", "seed0"))
    assert meta["gnn"]["top_k"] == 11
    assert "--ablation" in [c for call in [runner.stage_cmd(runner.Target("beijing", _args(
        tmp_path, "--dataset", "beijing", "--denoiser", "attngnn", "--ablation", "topk_dense"),
        REPO), "train")] for c in call]
    # the main run is untouched and unaffected
    assert not os.path.exists(C.run_dir(str(tmp_path), "tabdiff_beijing", "attngnn", 0))
    sys.path.insert(0, os.path.join(REPO, "analysis", "tabdiff"))
    import compare
    comp, prov, _ = compare.ra_computed(str(tmp_path))
    assert not prov                                               # ablations never enter ROUTE_A.md


def test_ablation_job_script():
    t = open(os.path.join(JOBS, "beijing_tabdiff_gnn_ablation_topk_dense.sh")).read()
    assert 'ABLATION="${ABLATION:-topk_dense}"' in t and "_topk11_" in t and "top-k 11" in t
    assert "#SBATCH --job-name=beijing_tabdiff_gnn_ablation_topk_dense" in t and "#SBATCH --time=48:00:00" in t
    for ds in C.ORDER:
        for f in glob.glob(os.path.join(JOBS, f"{ds}_tabdiff_gnn*.sh")):
            if "ablation" not in f:
                assert 'ABLATION="${ABLATION:-}"' in open(f).read()
    sub = open(os.path.join(REPO, "hpc_transfer", "submit_tabdiff_gnn.sh")).read()
    abl = [ln for ln in sub.splitlines() if "ablation_topk_dense" in ln and "sbatch" in ln]
    assert len(abl) == 1 and abl[0].startswith("# ")
    assert sub.index("=== ABLATION") > sub.index("block 2")       # after the main runs


def test_diagnosis_column_typing_and_eta():
    sys.path.insert(0, os.path.join(REPO, "analysis", "tabdiff"))
    import route_a_diagnosis as D
    info, real = D.info_and_real("beijing")
    num, cat = D.column_types(info)
    assert num == ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "pm2.5"]
    assert cat == ["year", "month", "day", "hour", "cbwd"]
    info, _ = D.info_and_real("magic")
    num, cat = D.column_types(info)
    assert len(num) == 10 and cat == ["class"]
    assert D.eta(["a", "a", "b", "b"], [1, 1, 3, 3]) == pytest.approx(1.0)
    assert D.eta(["a", "b", "a", "b"], [1, 1, 3, 3]) == pytest.approx(0.0)
