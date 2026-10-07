#!/usr/bin/env python3
"""TabDDPM's tuned configuration vs the GNN configuration actually run, per dataset.

    python3 analysis/config_comparison/config_comparison.py

Read-only: nothing is trained or sampled. Both denoisers are instantiated on CPU
(untrained) with each dataset's real shapes, only to count parameters.

TabDDPM side   exp/<ds>/ddpm_cb_best/config.toml (+ eval_catboost.json, info.json)
               and the defaults of the code that consumes it (scripts/pipeline.py,
               scripts/train.py, scripts/sample.py, scripts/eval_seeds.py, ...).
My side        the run used by the current analysis (analysis/topk_comparison/
               coverage.csv: *_catfix trees for categorical datasets,
               ablation_fixed_seed0 / original trees for numerical ones), or,
               for datasets that analysis does not cover (fb-comments,
               higgs-small, miniboone), the original transfer trees. Values come
               from that run's run_meta.json / denoiser_config.json, from
               ablation/<ds>_ablation.py DATASET_CFG and from the module
               constants / call sites of ablation/ablation_runner.py.

Every value carries a file:line located by searching the source (cite()), so a
moved line shows up as a failed lookup instead of a stale reference.

Outputs (this directory): CONFIG_COMPARISON.md, config_comparison.csv,
param_counts.csv, results_link.csv, spearman.csv, config_table.tex, figures/.
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "analysis" / "topk_comparison"))
sys.path.insert(0, str(REPO / "analysis" / "dependency_fidelity"))
import compare_topk as ct  # noqa: E402

RESULTS = REPO / "transfer_runs" / "exp"
COVERAGE = REPO / "analysis" / "topk_comparison" / "coverage.csv"
RUNNER = "ablation/ablation_runner.py"
RULES = ct.RULES
LEGACY_TREES = {"fixed": ("ablation",), "jenks": ("ablation_adaptive_jenks",),
                "entmax15": ("ablation_adaptive_entmax15",)}
GROUPS = ("selection", "a. architecture", "b. parameters", "c. training", "d. diffusion",
          "e. sampling", "f. data / preprocessing", "g. evaluation")
C_TAB, C_MINE = "#2a78d6", "#eb6834"          # reference palette slots 1-2
C_TEXT, C_TEXT2, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Source lookup
# ---------------------------------------------------------------------------
_LINES: dict = {}


def _lines(rel: str) -> list[str]:
    if rel not in _LINES:
        _LINES[rel] = (REPO / rel).read_text().splitlines()
    return _LINES[rel]


def cite(rel: str, pattern: str, after: str | None = None) -> str:
    """'rel:line' of the first line matching regex `pattern` (after the first
    line matching `after`, if given). Raises if absent: a missing line means
    the code changed and the reference must be re-checked."""
    lines, start = _lines(rel), 0
    if after:
        start = next(i for i, l in enumerate(lines) if re.search(after, l))
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return f"{rel}:{i + 1}"
    raise LookupError(f"{pattern!r} not found in {rel} (after {after!r})")


def toml_line(rel: str, section: str, key: str) -> str:
    cur = ""
    for i, l in enumerate(_lines(rel)):
        m = re.match(r"\s*\[([^\]]+)\]", l)
        if m:
            cur = m.group(1)
        elif cur == section and re.match(rf"\s*{re.escape(key)}\s*=", l):
            return f"{rel}:{i + 1}"
    return f"{rel} (absent)"


def module_constants(rel: str) -> dict:
    tree = ast.parse((REPO / rel).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = (ast.literal_eval(node.value), f"{rel}:{node.lineno}")
            except ValueError:
                pass
    return out


def dataset_cfg(ds: str) -> tuple[dict, str]:
    rel = f"ablation/{ds.replace('-', '_')}_ablation.py"
    for node in ast.parse((REPO / rel).read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id == "DATASET_CFG":
                d = node.value
                return ({ast.literal_eval(k): (ast.literal_eval(v), f"{rel}:{k.lineno}")
                         for k, v in zip(d.keys, d.values)}, rel)
    raise LookupError(rel)


def runner_reads(key: str) -> bool:
    """Does any consumer actually read DATASET_CFG[key]?"""
    pat = re.compile(rf"dataset_cfg(\[|\.get\()[\"']{re.escape(key)}[\"']")
    return any(pat.search(l) for f in (RUNNER, "hpc_transfer/split_runner.py") for l in _lines(f))


# ---------------------------------------------------------------------------
# Which run is "mine"
# ---------------------------------------------------------------------------
def my_runs(ds: str, cov: pd.DataFrame) -> dict:
    out = {}
    c = cov[cov.dataset == ds].set_index("rule") if ds in set(cov.dataset) else None
    for r in RULES:
        path, status, variant = None, "no run", "legacy"
        if c is not None and r in c.index and str(c.loc[r, "status"]).startswith("complete"):
            path, variant = REPO / c.loc[r, "source"], c.loc[r, "variant"]
        else:
            for t in LEGACY_TREES[r]:
                cand = sorted(p for p in (RESULTS / ds / t).glob("gnn_*") if (p / "run_meta.json").exists())
                if cand:
                    path = cand[0]
                    break
        if path is not None and path.is_dir():
            n = ct.load_run_dir(path)["n_gen"]
            status = "complete" if n >= ct.EXPECTED_GEN else f"partial ({n}/{ct.EXPECTED_GEN} gen seeds)"
            if c is not None and r in c.index and c.loc[r, "data_kind"] == "summary only (training seed 0)":
                status = "complete (summary only)"
        out[r] = {"path": path, "status": status, "variant": variant}
    return out


# ---------------------------------------------------------------------------
# Shapes and parameter counts
# ---------------------------------------------------------------------------
_SETUP = False


def shapes_and_params(ds: str, tcfg: dict, graph_by_rule: dict) -> dict:
    global _SETUP
    if not _SETUP:
        from cat_input_local_check import _setup
        _setup()
        _SETUP = True
    import lib
    from utils_train import get_model, make_dataset
    mp = tcfg["model_params"]
    T = lib.Transformations(**{k: (None if v == "__none__" else v) for k, v in tcfg["train"]["T"].items()})
    D = make_dataset(str(REPO / "data" / ds) + "/", T, num_classes=mp["num_classes"], is_y_cond=mp["is_y_cond"],
                     change_val=False)
    K = list(D.get_category_sizes("train"))
    d_num = D.X_num["train"].shape[1] if D.X_num is not None else 0
    d_in = int(sum(K) + d_num)

    def params(gp=None):
        m = get_model("mlp", {"num_classes": mp["num_classes"], "is_y_cond": mp["is_y_cond"], "d_in": d_in,
                              "rtdl_params": {"d_layers": list(mp["rtdl_params"]["d_layers"]),
                                              "dropout": mp["rtdl_params"]["dropout"]}},
                      d_num, K, graph_params=gp)
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        out = {"K": K, "d_num": d_num, "d_in": d_in, "n_nodes": d_num + len([k for k in K if k > 0]),
               "params_tabddpm": params()}
        for r, gp in graph_by_rule.items():
            out[f"params_{r}"] = params(gp) if gp else np.nan
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return str(a) == str(b)


def fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e-2 else f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ", ".join(map(str, v)) + "]"
    return str(v)


def build_dataset(ds: str, cov: pd.DataFrame, RC: dict, ablated: set) -> tuple[list, dict, dict]:
    trel = f"exp/{ds}/ddpm_cb_best/config.toml"
    tc = tomllib.load(open(REPO / trel, "rb"))
    info = json.load(open(REPO / "data" / ds / "info.json"))
    info_b = json.load(open(REPO / "exp" / ds / "ddpm_cb_best" / "info.json"))
    cfg, crel = dataset_cfg(ds)
    runs = my_runs(ds, cov)
    subset = "development" if ds in ablated else "held-out"
    rows = []

    def add(group, key, tv, tsrc, mv, msrc, rule="all", note="", compare=True):
        rows.append({"dataset": ds, "subset": subset, "rule": rule, "group": group, "key": key,
                     "tabddpm_value": fmt(tv), "tabddpm_source": tsrc, "my_value": fmt(mv), "my_source": msrc,
                     "differs": (not same(tv, mv)) if compare else None, "note": note})

    metas, dcs = {}, {}
    for r, run in runs.items():
        p = run["path"]
        metas[r] = json.load(open(p / "run_meta.json")) if p and (p / "run_meta.json").exists() else {}
        dcs[r] = json.load(open(p / "denoiser_config.json")) if p and (p / "denoiser_config.json").exists() else {}
    ref = next((r for r in RULES if metas[r]), None)
    meta = metas[ref] if ref else {}
    mrel = {r: (str(runs[r]["path"].relative_to(REPO)) + "/run_meta.json") if runs[r]["path"] else "—" for r in RULES}

    # ---- selection --------------------------------------------------------
    for r in RULES:
        add("selection", "run used", "exp/ddpm_cb_best (authors)", trel,
            str(runs[r]["path"].relative_to(REPO)) if runs[r]["path"] else "no run", "", rule=r,
            note=runs[r]["status"], compare=False)
    if subset == "development":
        how = ("argmax of fixed-mode TEST score.mean over own GNN grid "
               f"({meta.get('provenance', '?')})")
        src = f"{cite('hpc_transfer/best_manifest.py', 'ranks by')}; {mrel[ref]}"
    else:
        how = (f"transferred, {meta.get('strategy', '?')}: {meta.get('provenance', '?')}; neighbours "
               f"{meta.get('neighbours', '?')}; scaled alternative of the LOO-validated global best")
        src = f"{cite('hpc_transfer/predictions.md', 'scaled alternative')}; {mrel[ref]}"
    tune = "scripts/tune_ddpm.py"
    add("selection", "how chosen",
        "Optuna, 50 trials over lr, MLP layers, batch {256,4096}, steps {5k,20k,30k}, T {100,1000}, "
        "synthetic size train×2^{-2..1}; score = CatBoost val macro-F1/R2 (val re-split from train, --change_val)",
        "; ".join([cite(tune, r"n_trials=50"), cite(tune, r"suggest_loguniform\('lr'"),
                   cite(tune, r"suggest_categorical\('steps'"), cite(tune, r"num_samples = int\(train_size"),
                   cite(tune, r"metrics'\]\['val'\]\['macro avg'\]"), cite(tune, r"'--train', '--change_val'")]),
        how, src, compare=False)

    # ---- architecture -----------------------------------------------------
    gmp = "scripts/utils_train.py"
    add("a. architecture", "denoiser", "MLPDiffusion (rtdl MLP)", cite(gmp, r"model = MLPDiffusion"),
        "GraphAwareDenoiser", cite(gmp, r"model = GraphAwareDenoiser"))
    add("a. architecture", "d_layers", tc["model_params"]["rtdl_params"]["d_layers"],
        toml_line(trel, "model_params.rtdl_params", "d_layers"),
        "— (rtdl_params stripped)", cite(gmp, r"rtdl_params',\)"),
        note=f"DATASET_CFG copies rtdl_d_layers ({cfg['rtdl_d_layers'][1]}) but the graph branch drops it")
    add("a. architecture", "dropout", tc["model_params"]["rtdl_params"]["dropout"],
        toml_line(trel, "model_params.rtdl_params", "dropout"),
        0.0, cite(RUNNER, r"dropout=0\.0", after=r"graph_params = dict\("))
    add("a. architecture", "dim_t", 128, cite("tab_ddpm/modules.py", r"class MLPDiffusion", ) + " (default dim_t = 128)",
        128, cite("tab_ddpm/graph_denoiser.py", r"dim_t: int = 128") + "; not passed by get_model")
    for key, gkey in (("gnn_type", "gnn_type"), ("n_layers", "n_layers"), ("d_model", "d_model"),
                      ("n_heads", "n_heads"), ("top_k", "top_k"), ("use_attention", "use_attention")):
        vals = {r: metas[r].get(gkey) for r in RULES if metas[r]}
        add("a. architecture", key, "—", "", vals.get(ref), mrel[ref], compare=False,
            note="" if len(set(map(str, vals.values()))) <= 1 else f"differs across rules: {vals}")
    add("a. architecture", "graph_mode", "—", "", "dynamic", cite(RUNNER, r'mode="dynamic"'), compare=False)
    for r in RULES:
        if not metas[r]:
            continue
        ci = dcs[r].get("cat_input") or metas[r].get("cat_input") or "log"
        csrc = (str(runs[r]["path"].relative_to(REPO)) + "/denoiser_config.json") if "cat_input" in dcs[r] else \
            (mrel[r] if "cat_input" in metas[r] else cite(RUNNER, r'cat_input: str = "log",', after=r"^def _run_one\(") + " (default; run predates option)")
        add("a. architecture", "cat_input", "— (MLP: one-hot probabilities via proj)", "", ci, csrc, rule=r, compare=False)
        tm = dcs[r].get("topk_mode") or metas[r].get("topk_mode") or "fixed"
        add("a. architecture", "topk_mode", "—", "", tm, mrel[r], rule=r, compare=False)

    # ---- parameters (filled after counting) --------------------------------
    gps = {}
    for r in RULES:
        m = metas[r]
        gps[r] = None if not m else dict(
            enabled=True, mode="dynamic", gnn_type=m["gnn_type"], d_model=m["d_model"], n_layers=m["n_layers"],
            n_heads=m["n_heads"], sparsity_top_k=m["top_k"], dropout=0.0, use_attention=m["use_attention"],
            topk_mode=dcs[r].get("topk_mode") or m.get("topk_mode", "fixed"), k_min=m.get("k_min", 1),
            topk_warmup_steps=m.get("topk_warmup_steps", 0), num_timesteps=RC["ABLATION_NUM_TIMESTEPS"][0],
            cat_input=dcs[r].get("cat_input") or m.get("cat_input", "log"))
    sp = shapes_and_params(ds, tc, gps)
    add("a. architecture", "d_in (one-hot width)", sp["d_in"], "computed: sum(K) + d_num (scripts/train.py)",
        sp["d_in"], "same dataset object")
    add("a. architecture", "n_nodes (d_num + n_cat)", "—", "", sp["n_nodes"],
        cite("scripts/train.py", r"n_nodes=") + " (regression: y is a numerical node)", compare=False)
    for r in RULES:
        if gps[r]:
            add("b. parameters", "trainable parameters", sp["params_tabddpm"], "instantiated on CPU",
                sp[f"params_{r}"], "instantiated on CPU", rule=r,
                note=f"ratio mine/TabDDPM = {sp[f'params_{r}'] / sp['params_tabddpm']:.3f}")

    # ---- training ----------------------------------------------------------
    steps_m = RC["ABLATION_STEPS"]
    add("c. training", "steps", tc["train"]["main"]["steps"], toml_line(trel, "train.main", "steps"),
        steps_m[0], steps_m[1] + "; not overridden (" + cite("hpc_transfer/common.sh", r"# NOT passed, so the run inherits") + ")")
    lr_m = RC["ABLATION_LR"]
    add("c. training", "lr", tc["train"]["main"]["lr"], toml_line(trel, "train.main", "lr"),
        lr_m[0], lr_m[1] + "; used at " + cite(RUNNER, r"lr=ABLATION_LR"),
        note=("DATASET_CFG['lr'] = " + fmt(cfg["lr"][0]) + f" ({cfg['lr'][1]}) is never read")
        if "lr" in cfg and not runner_reads("lr") else "")
    add("c. training", "weight_decay", tc["train"]["main"]["weight_decay"], toml_line(trel, "train.main", "weight_decay"),
        cfg["weight_decay"][0], cfg["weight_decay"][1] + "; " + cite(RUNNER, r'weight_decay=dataset_cfg\["weight_decay"\]'))
    bs_m = RC["ABLATION_BATCH_SIZE"]
    add("c. training", "batch_size", tc["train"]["main"]["batch_size"], toml_line(trel, "train.main", "batch_size"),
        bs_m[0], bs_m[1] + "; used at " + cite(RUNNER, r"batch_size=ABLATION_BATCH_SIZE"),
        note=("DATASET_CFG['batch_size'] = " + fmt(cfg["batch_size"][0]) + " is never read")
        if "batch_size" in cfg and not runner_reads("batch_size") else "")
    opt = cite("scripts/train.py", r"torch\.optim\.AdamW")
    add("c. training", "optimizer", "AdamW", opt, "AdamW", opt + " (same Trainer)")
    ann = cite("scripts/train.py", r"lr = self\.init_lr \* \(1 - frac_done\)")
    add("c. training", "lr schedule", "linear decay to 0", ann, "linear decay to 0", ann)
    ema = cite("scripts/utils_train.py", r"def update_ema\(.*rate=0\.999")
    add("c. training", "EMA", "rate 0.999, every step", ema, "rate 0.999, every step", ema)
    add("c. training", "training seed", tc["seed"], toml_line(trel, "", "seed"),
        meta.get("seed"), mrel[ref])

    # ---- diffusion ---------------------------------------------------------
    gmd = "tab_ddpm/gaussian_multinomial_diffsuion.py"
    add("d. diffusion", "num_timesteps", tc["diffusion_params"]["num_timesteps"],
        toml_line(trel, "diffusion_params", "num_timesteps"),
        RC["ABLATION_NUM_TIMESTEPS"][0], RC["ABLATION_NUM_TIMESTEPS"][1])
    sch_t = tc["diffusion_params"].get("scheduler")
    add("d. diffusion", "scheduler", sch_t or "cosine",
        toml_line(trel, "diffusion_params", "scheduler") if sch_t else cite("scripts/train.py", r"scheduler = 'cosine'") + " (default; key absent)",
        cfg["scheduler"][0], cfg["scheduler"][1])
    add("d. diffusion", "gaussian_loss_type", tc["diffusion_params"]["gaussian_loss_type"],
        toml_line(trel, "diffusion_params", "gaussian_loss_type"),
        "mse", cite(RUNNER, r'gaussian_loss_type="mse"'))
    for key, pat in (("gaussian_parametrization", r"gaussian_parametrization='eps'"),
                     ("multinomial_loss_type", r"multinomial_loss_type='vb_stochastic'"),
                     ("parametrization", r"parametrization='x0'")):
        v = re.search(r"'(\w+)'", pat).group(1)
        src = cite(gmd, r"^\s+" + pat) + " (default; train() does not pass it)"
        add("d. diffusion", key, v, src, v, src)

    # ---- sampling ----------------------------------------------------------
    ns_t = tc["sample"]["num_samples"]
    add("e. sampling", "num_samples", ns_t, toml_line(trel, "sample", "num_samples"),
        cfg["num_samples"][0], cfg["num_samples"][1])
    add("e. sampling", "num_samples / train_size", round(ns_t / info["train_size"], 3), "data/<ds>/info.json",
        round(cfg["num_samples"][0] / info["train_size"], 3), "data/<ds>/info.json")
    add("e. sampling", "sample batch_size", tc["sample"]["batch_size"], toml_line(trel, "sample", "batch_size"),
        cfg["sample_batch_size"][0], cfg["sample_batch_size"][1])
    ycls = info["task_type"] != "regression"
    ydist_t = ("empirical class distribution (disbalance=None)" if ycls else "y generated as a numerical column")
    add("e. sampling", "y distribution", ydist_t,
        cite("scripts/pipeline.py", r"disbalance=raw_config\['sample'\]\.get\('disbalance', None\)"),
        ydist_t, cite("scripts/sample.py", r"disbalance = None") + " (not passed by the runner)")
    dd = cite("scripts/sample.py", r"x_gen, y_gen = diffusion\.sample_all\(num_samples, batch_size, empirical_class_dist\.float\(\), ddim=False\)",
              after=r"^\s+else:")
    add("e. sampling", "DDIM", False, dd, False, dd)
    add("e. sampling", "checkpoint sampled", "model.pt (raw weights)",
        cite("scripts/eval_seeds.py", r'shutil\.copy2\(parent_dir / "model\.pt"'),
        "model_ema.pt (EMA weights)", cite(RUNNER, r'model_path=os\.path\.join\(parent_dir, "model_ema\.pt"\)'))
    add("e. sampling", "gen seeds x clf seeds", "5 x 10", f"exp/{ds}/ddpm_cb_best/eval_catboost.json (count 50)",
        f"{RC['N_GEN_SEEDS'][0]} x {RC['N_CLF_SEEDS'][0]}", RC["N_GEN_SEEDS"][1])

    # ---- data --------------------------------------------------------------
    tT = tc["train"]["T"]
    add("f. data / preprocessing", "normalization (train)", tT["normalization"], toml_line(trel, "train.T", "normalization"),
        cfg["train_normalization"][0], cfg["train_normalization"][1])
    for key in ("cat_encoding", "cat_min_frequency", "num_nan_policy", "cat_nan_policy"):
        add("f. data / preprocessing", key, None if tT[key] == "__none__" else tT[key], toml_line(trel, "train.T", key),
            None, cite(RUNNER, rf"{key}=None", after=r"T_dict_train = dict\("))
    add("f. data / preprocessing", "y_policy", tT["y_policy"], toml_line(trel, "train.T", "y_policy"),
        "default", cite(RUNNER, r'y_policy="default"', after=r"T_dict_train = dict\("))
    add("f. data / preprocessing", "is_y_cond", tc["model_params"]["is_y_cond"], toml_line(trel, "model_params", "is_y_cond"),
        cfg["is_y_cond"][0], cfg["is_y_cond"][1])
    add("f. data / preprocessing", "num_classes", tc["model_params"]["num_classes"], toml_line(trel, "model_params", "num_classes"),
        cfg["num_classes"][0], cfg["num_classes"][1])
    add("f. data / preprocessing", "num_numerical_features", tc["num_numerical_features"],
        toml_line(trel, "", "num_numerical_features"), cfg["num_numerical_features"][0], cfg["num_numerical_features"][1])
    for k in ("n_num_features", "n_cat_features"):
        add("f. data / preprocessing", k, info_b.get(k), f"exp/{ds}/ddpm_cb_best/info.json", info.get(k), f"data/{ds}/info.json",
            note="local data/ is NOT the authors' layout: the TabDDPM run and its eval_catboost.json used other "
                 "column types / splits" if info_b.get(k) != info.get(k) else "")
    for part in ("train", "val", "test"):
        add("f. data / preprocessing", f"{part}_size", info_b[f"{part}_size"], f"exp/{ds}/ddpm_cb_best/info.json",
            info[f"{part}_size"], f"data/{ds}/info.json")

    # ---- evaluation --------------------------------------------------------
    cb = f"tuned_models/catboost/{ds}_cv.json"
    cbsrc = cite("scripts/eval_catboost.py", r"get_catboost_config\(real_data_path, is_cv=True\)") + "; " + \
        cite("lib/util.py", r"tuned_models' / 'catboost'")
    add("g. evaluation", "CatBoost config", cb + (" (exists)" if (REPO / cb).exists() else " (MISSING)"), cbsrc,
        cb + (" (exists)" if (REPO / cb).exists() else " (MISSING)"), cbsrc + " (same function)")
    add("g. evaluation", "normalization (eval)", None, cite("scripts/eval_seeds.py", r'T_dict\["normalization"\] = None'),
        cfg["eval_normalization"][0], cfg["eval_normalization"][1] + "; " + cite(RUNNER, r'normalization=dataset_cfg\.get\("eval_normalization"'),
        note="CatBoost is invariant to monotone rescaling up to border quantisation" if cfg["eval_normalization"][0] else "")
    met = "macro-F1" if ycls else "R2"
    msrc = cite("lib/metrics.py", r'\["macro avg"\]\["f1-score"\]') if ycls else cite("lib/metrics.py", r'"r2"\] = report')
    add("g. evaluation", "utility metric", met, msrc + " (SeedsMetricsReport)", met, msrc + " (same report dict)")

    extra = {"dataset": ds, "subset": subset, "task": info["task_type"], "train_size": info["train_size"],
             "steps_t": tc["train"]["main"]["steps"], "steps_m": steps_m[0],
             "lr_t": tc["train"]["main"]["lr"], "lr_m": lr_m[0],
             "batch_t": tc["train"]["main"]["batch_size"], "batch_m": bs_m[0],
             "samples_t": ns_t, "samples_m": cfg["num_samples"][0],
             "d_layers": tc["model_params"]["rtdl_params"]["d_layers"],
             "layout_differs": info_b.get("n_cat_features") != info.get("n_cat_features")
                               or info_b.get("test_size") != info.get("test_size"),
             "arch_m": (f"{meta['gnn_type']} L{meta['n_layers']} d{meta['d_model']} h{meta['n_heads']} k{meta['top_k']}"
                        if meta else "—"), "runs": runs, **sp}
    return rows, extra, {"tc": tc}


# ---------------------------------------------------------------------------
# Results link
# ---------------------------------------------------------------------------
def my_utility(path: Path, task: str) -> tuple[float, float, int]:
    import paper_comparison as pc
    metric = "r2" if task == "regression" else "macro avg/f1-score"
    if path is None or ct.load_run_dir(path)["n_gen"] < ct.EXPECTED_GEN:
        return np.nan, np.nan, 0
    v = pc.raw_runs(path, metric)
    return float(v.mean()), float(v.std()), len(v)


def results_link(extras: dict) -> pd.DataFrame:
    rows = []
    for ds, e in extras.items():
        ev = json.load(open(REPO / "exp" / ds / "ddpm_cb_best" / "eval_catboost.json"))["synthetic"]["test"]
        k = "r2" if e["task"] == "regression" else "f1"
        for r in RULES:
            run = e["runs"][r]
            m, s, n = my_utility(run["path"], e["task"]) if run["status"] == "complete" else (np.nan, np.nan, 0)
            rows.append({"dataset": ds, "subset": e["subset"], "rule": r, "metric": "R2" if k == "r2" else "macro-F1",
                         "tabddpm_eval_catboost": ev[f"{k}-mean"], "tabddpm_std": ev[f"{k}-std"],
                         "mine": m, "mine_std": s, "n_runs": n, "status": run["status"],
                         "delta": m - ev[f"{k}-mean"],
                         "steps_ratio": e["steps_m"] / e["steps_t"], "lr_ratio": e["lr_m"] / e["lr_t"],
                         "batch_ratio": e["batch_m"] / e["batch_t"], "samples_ratio": e["samples_m"] / e["samples_t"],
                         "param_ratio": e[f"params_{r}"] / e["params_tabddpm"]})
    return pd.DataFrame(rows)


RATIOS = ("steps_ratio", "lr_ratio", "batch_ratio", "samples_ratio", "param_ratio")


def spearman(rl: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in RULES:
        for sub in ("held-out", "pooled"):
            d = rl[(rl.rule == r) & rl.delta.notna() & ((rl.subset == sub) if sub != "pooled" else True)]
            for ratio in RATIOS:
                x = d[ratio].to_numpy()
                if len(d) < 3 or np.ptp(x) == 0:
                    rho = p = np.nan
                    note = "constant ratio" if len(d) and np.ptp(x) == 0 else "n < 3"
                else:
                    rho, p = stats.spearmanr(x, d.delta)
                    note = ""
                out.append({"rule": r, "subset": sub, "ratio": ratio, "n": len(d), "rho": rho, "p": p, "note": note})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Figure, LaTeX, report
# ---------------------------------------------------------------------------
def figure(pc_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    d = pc_df.assign(_o=pc_df.subset.map({"held-out": 0, "development": 1})).sort_values(["_o", "dataset"])
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(d) + 1.3))
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.params_tabddpm, r.params_mine], [i, i], color=C_GRID, lw=2, zorder=1)
    ax.scatter(d.params_tabddpm, y, s=46, marker="o", color=C_TAB, edgecolor="#fcfcfb", lw=1.5, zorder=3,
               label="TabDDPM best (MLP)")
    ax.scatter(d.params_mine, y, s=52, marker="D", color=C_MINE, edgecolor="#fcfcfb", lw=1.5, zorder=3,
               label="mine (GNN, all 3 rules)")
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{ds}{'  (held-out)' if s == 'held-out' else ''}" for ds, s in zip(d.dataset, d.subset)],
                  fontsize=8.5, color=C_TEXT)
    ax.invert_yaxis()
    n_ho = (d.subset == "held-out").sum()
    ax.axhline(n_ho - 0.5, color=C_TEXT2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("trainable parameters (log scale)", fontsize=9, color=C_TEXT)
    ax.set_title("Denoiser size: TabDDPM's tuned MLP vs the GNN actually run", fontsize=10, color=C_TEXT, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_TEXT2)
    ax.tick_params(colors=C_TEXT2)
    ax.grid(axis="x", color=C_GRID, lw=0.8, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "param_counts.png", dpi=180)
    fig.savefig(HERE / "figures" / "param_counts.pdf")
    plt.close(fig)


def short_layers(L: list) -> str:
    if len(set(L)) == 1:
        return f"{len(L)}×{L[0]}"
    mid = L[1:-1]
    if mid and len(set(mid)) == 1:
        return f"{L[0]}–{len(mid)}×{mid[0]}–{L[-1]}"
    return "–".join(map(str, L))


def latex(extras: dict, pc_df: pd.DataFrame) -> str:
    def n(x: int) -> str:
        return f"{x:,}".replace(",", "{,}")
    bad = {ds for ds, e in extras.items() if e.get("layout_differs")}
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{TabDDPM's tuned configuration (T, MLP) vs the GNN configuration run here (G). Params = trainable "
         r"parameters of the denoiser; synthetic size in multiples of the real train size. Held-out datasets above "
         r"the middle rule, development datasets below.}",
         r"\label{tab:config-comparison}",
         r"\begin{tabular}{lllrrrrrrr}", r"\toprule",
         r"dataset & T MLP layers & G architecture & T params & G params & G/T & T steps & T lr & T batch & synth. \\",
         r"\midrule"]
    order = sorted(extras, key=lambda d: (extras[d]["subset"] != "held-out", d))
    first_dev = True
    for ds in order:
        e = extras[ds]
        if e["subset"] == "development" and first_dev:
            L.append(r"\midrule")
            first_dev = False
        p = pc_df.set_index("dataset").loc[ds]
        mark = r"$^\dagger$" if ds in bad else ""
        tl = short_layers(e["d_layers"]).replace("–", "--").replace("×", r"$\times$")
        L.append(f"{ds}{mark} & {tl} & {e['arch_m']} & {p.params_tabddpm / 1e6:.2f}M & "
                 f"{p.params_mine / 1e6:.2f}M & {p.params_mine / p.params_tabddpm:.2f} & {n(e['steps_t'])} & "
                 f"{e['lr_t']:.1e} & {n(e['batch_t'])} & {e['samples_t'] / e['train_size']:.1f} \\\\")
    any_e = next(iter(extras.values()))
    L += [r"\midrule",
          rf"\multicolumn{{10}}{{l}}{{G, every dataset: {n(any_e['steps_m'])} steps, lr {any_e['lr_m']:.0e}, batch "
          rf"{n(any_e['batch_m'])}, same synthetic size as T; G samples the EMA weights, T the raw weights.}} \\"]
    if bad:
        L.append(rf"\multicolumn{{10}}{{l}}{{$^\dagger$ local data differs from the authors' (column types, "
                 rf"val/test split); T params use the local shapes.}} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def md_table(rows: pd.DataFrame) -> list[str]:
    L = ["| group | key | rule | TabDDPM | source | mine | source | note |", "|---|---|---|---|---|---|---|---|"]
    for _, r in rows.iterrows():
        mark = r.differs is True
        tv = f"**{r.tabddpm_value}**" if mark else r.tabddpm_value
        mv = f"**{r.my_value}** ≠" if mark else r.my_value
        L.append(f"| {r.group} | {r.key} | {r.rule} | {tv} | `{r.tabddpm_source}` | {mv} | `{r.my_source}` | {r.note} |")
    return L


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (g, k), d in df[df.differs.notna()].groupby(["group", "key"], sort=False):
        dif = d.differs.astype(bool)
        mine_const = d.my_value.nunique() == 1
        if not dif.any():
            kind = "identical for all datasets"
        elif dif.all() and mine_const:
            kind = "differs systematically (mine constant, theirs tuned per dataset)"
        elif dif.all():
            kind = "differs in every dataset (both sides vary)"
        else:
            kind = f"varies: differs in {d[dif].dataset.nunique()} of {d.dataset.nunique()} datasets"
        out.append({"group": g, "key": k, "kind": kind,
                    "datasets_differing": ";".join(sorted(d[dif].dataset.unique())),
                    "tabddpm_values": ";".join(sorted(d.tabddpm_value.unique())),
                    "my_values": ";".join(sorted(d.my_value.unique()))})
    return pd.DataFrame(out)


def step0() -> list[str]:
    def flat(d, p=""):
        for k, v in d.items():
            yield from (flat(v, f"{p}{k}.") if isinstance(v, dict) else [f"{p}{k}"])
    tk = list(flat(tomllib.load(open(REPO / "exp/cardio/ddpm_cb_best/config.toml", "rb"))))
    union = set()
    for f in sorted(REPO.glob("exp/*/ddpm_cb_best/config.toml")):
        union |= set(flat(tomllib.load(open(f, "rb"))))
    meta_f = sorted(RESULTS.glob("cardio/ablation_catfix/gnn_*/run_meta.json"))[0]
    mk = list(json.load(open(meta_f)))
    cfg, crel = dataset_cfg("cardio")
    M = [  # (toml key, my key, where my value lives)
        ("model_params.rtdl_params.d_layers", "gnn_type, n_layers, d_model, n_heads, top_k, use_attention",
         "run_meta.json (architecture replaced, not mapped)"),
        ("model_params.rtdl_params.dropout", "graph_params.dropout", f"{RUNNER} (hard-coded 0.0)"),
        ("model_params.is_y_cond", "DATASET_CFG.is_y_cond", crel),
        ("model_params.num_classes", "DATASET_CFG.num_classes", crel),
        ("num_numerical_features", "DATASET_CFG.num_numerical_features", crel),
        ("diffusion_params.num_timesteps", "ABLATION_NUM_TIMESTEPS (DATASET_CFG.num_timesteps unread)", RUNNER),
        ("diffusion_params.gaussian_loss_type", "literal \"mse\" at the train()/sample() call", RUNNER),
        ("diffusion_params.scheduler (some datasets)", "DATASET_CFG.scheduler", crel),
        ("train.main.steps", "ABLATION_STEPS (run_meta.steps = 'module default')", RUNNER),
        ("train.main.lr", "ABLATION_LR (DATASET_CFG.lr unread)", RUNNER),
        ("train.main.weight_decay", "DATASET_CFG.weight_decay", crel),
        ("train.main.batch_size", "ABLATION_BATCH_SIZE (DATASET_CFG.batch_size unread)", RUNNER),
        ("train.T.normalization", "DATASET_CFG.train_normalization", crel),
        ("train.T.{num_nan,cat_nan}_policy, cat_min_frequency, cat_encoding, y_policy", "T_dict_train literals", RUNNER),
        ("train.T.seed / seed", "run_meta.seed (--seed)", "run_meta.json"),
        ("sample.num_samples", "DATASET_CFG.num_samples", crel),
        ("sample.batch_size", "DATASET_CFG.sample_batch_size", crel),
        ("sample.seed", "gen_seed loop 0..N_GEN_SEEDS-1", RUNNER),
        ("eval.T.normalization", "DATASET_CFG.eval_normalization", crel),
        ("eval.type.eval_model", "EVALUATOR (run_meta.evaluator = 'module default')", RUNNER),
        ("eval.type.eval_type", "literal \"synthetic\"", RUNNER),
        ("real_data_path", "DATASET_CFG.real_data_path", crel),
        ("model_type", "literal \"mlp\" (graph_params override the model)", RUNNER),
    ]
    only_t = ["parent_dir", "device (run_meta.device exists, cosmetic)", "model_params.d_in (some files; computed anyway)",
              "eval.type.model (default only; duplicate of eval_model)"]
    only_m = ["gnn_type, n_layers, d_model, n_heads, top_k, use_attention, graph_mode, cat_input, topk_mode, k_min, "
              "topk_warmup_steps, entmax_alpha, topk_log_every (GNN only)",
              "study, provenance, strategy, neighbours, nearest_distance, confidence (selection record)",
              "n_gen_seeds, n_clf_seeds, evaluator (TabDDPM: eval_seeds.py CLI, not in config)",
              "hostname, gpu_name, git_commit, slurm_job_id, started_utc, top_k_role, exp_name (bookkeeping)"]
    L = ["## Step 0: key inventories and mapping\n",
         f"**TabDDPM** `exp/cardio/ddpm_cb_best/config.toml` ({len(tk)} keys): " + ", ".join(f"`{k}`" for k in tk) + ".",
         f"Keys present in other datasets' config.toml only: " + ", ".join(f"`{k}`" for k in sorted(union - set(tk))) + ".\n",
         f"**Mine** `{meta_f.relative_to(REPO)}` ({len(mk)} keys): " + ", ".join(f"`{k}`" for k in mk) + ".\n",
         "**run_meta.json records the architecture and the selection, but no training, diffusion or sampling "
         "value**: `steps`, `n_gen_seeds`, `n_clf_seeds`, `evaluator` are the string \"module default (…)\". Those "
         f"values come from `{crel}` (`DATASET_CFG`: " + ", ".join(cfg) + ") and from module constants / call-site "
         f"literals of `{RUNNER}`. Note that `DATASET_CFG` copies TabDDPM's `lr`, `batch_size` and `num_timesteps`, "
         "but the runner never reads them: it uses `ABLATION_LR`, `ABLATION_BATCH_SIZE`, `ABLATION_NUM_TIMESTEPS`.\n",
         "| TabDDPM config.toml key | my equivalent | where |", "|---|---|---|"]
    L += [f"| `{a}` | {b} | `{c}` |" for a, b, c in M]
    L.append("\nOnly in TabDDPM: " + "; ".join(only_t) + ".\n")
    L.append("Only in mine: " + "; ".join(only_m) + ".\n")
    L.append("Implicit on both sides (not in any config; code defaults, cited in the tables): optimizer, lr schedule, "
             "EMA, dim_t, gaussian/multinomial parametrization and loss type, DDIM, which checkpoint is sampled.\n")
    return L


def main():
    (HERE / "figures").mkdir(exist_ok=True)
    cov = pd.read_csv(COVERAGE)
    ablated = set(pd.read_csv(REPO / "hpc_transfer" / "best_per_dataset.csv").dataset)
    RC = module_constants(RUNNER)
    datasets = sorted(p.parent.parent.name for p in REPO.glob("exp/*/ddpm_cb_best/config.toml"))
    all_rows, extras = [], {}
    for ds in datasets:
        rows, e, _ = build_dataset(ds, cov, RC, ablated)
        all_rows += rows
        extras[ds] = e
        print(f"{ds}: {len(rows)} rows", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "config_comparison.csv", index=False)

    pc_df = pd.DataFrame([{"dataset": ds, "subset": e["subset"], "n_nodes": e["n_nodes"], "d_in": e["d_in"],
                           "d_num": e["d_num"], "category_sizes": ";".join(map(str, e["K"])),
                           "tabddpm_d_layers": "-".join(map(str, e["d_layers"])), "my_architecture": e["arch_m"],
                           "params_tabddpm": e["params_tabddpm"],
                           **{f"params_{r}": e[f"params_{r}"] for r in RULES}} for ds, e in extras.items()])
    pc_df["params_mine"] = pc_df[[f"params_{r}" for r in RULES]].max(axis=1)
    pc_df["rules_same_params"] = pc_df[[f"params_{r}" for r in RULES]].nunique(axis=1, dropna=True) <= 1
    pc_df["ratio_mine_over_tabddpm"] = pc_df.params_mine / pc_df.params_tabddpm
    pc_df.to_csv(HERE / "param_counts.csv", index=False)
    figure(pc_df)
    (HERE / "config_table.tex").write_text(latex(extras, pc_df))

    rl = results_link(extras)
    rl.to_csv(HERE / "results_link.csv", index=False)
    sp = spearman(rl)
    sp.to_csv(HERE / "spearman.csv", index=False)
    cls = classify(df)
    cls.to_csv(HERE / "differences_summary.csv", index=False)
    write_report(df, pc_df, rl, sp, cls, extras)
    print("wrote", HERE.relative_to(REPO))


def write_report(df, pc_df, rl, sp, cls, extras):
    L = ["# TabDDPM's tuned configuration vs the GNN configuration actually run\n",
         "_Generated by `config_comparison.py`; read-only (nothing trained or sampled; both denoisers are only "
         "instantiated, untrained, on CPU to count parameters). Every value carries the file:line it was read from._\n",
         "**Caveat.** TabDDPM's configurations were tuned per dataset with Optuna on the validation split "
         "(architecture *and* lr, steps, batch size, synthetic size, …). Mine come from a grid over GNN architecture "
         "only, selected on the test split (development datasets), or were transferred from other datasets "
         "(held-out); the training hyperparameters are not tuned per dataset at all. Differences in results below "
         "therefore confound architecture with training budget and tuning effort.\n"]
    L += step0()

    L.append("## Runs compared\n")
    L.append("| dataset | subset | fixed | Jenks | entmax-1.5 | my architecture | how chosen |")
    L.append("|---|---|---|---|---|---|---|")
    order = sorted(extras, key=lambda d: (extras[d]["subset"] != "held-out", d))
    for ds in order:
        e = extras[ds]
        st = [f"{e['runs'][r]['status']} ({e['runs'][r]['variant']})" if e["runs"][r]["path"] else "no run" for r in RULES]
        how = df[(df.dataset == ds) & (df.key == "how chosen")].my_value.iloc[0]
        L.append(f"| {ds} | {e['subset']} | " + " | ".join(st) + f" | {e['arch_m']} | {how} |")
    L.append("\nfb-comments: **no completed run** (fixed and Jenks have trained only; entmax never started); its "
             "configuration is still compared. higgs-small and miniboone: configuration compared; results only for "
             "cells with 5 complete gen seeds.\n")

    L.append("## Summary across datasets\n")
    def vals(v: str) -> str:
        u = v.split(";")
        return "{" + ", ".join(u) + "}" if len(u) <= 4 and len(v) < 90 else f"({len(u)} per-dataset values)"
    for kind, title in (("identical", "Identical for all datasets"),
                        ("differs systematically", "Differs systematically (mine one constant, theirs tuned per dataset)"),
                        ("differs in every", "Differs in every dataset (both sides vary)"),
                        ("varies", "Differs in some datasets only")):
        sub = cls[cls.kind.str.startswith(kind)]
        if sub.empty:
            continue
        L.append(f"**{title}:**\n")
        for _, r in sub.iterrows():
            if kind == "identical":
                L.append(f"- {r.group} / `{r.key}`: {vals(r.my_values)}")
            else:
                L.append(f"- {r.group} / `{r.key}`: TabDDPM {vals(r.tabddpm_values)} vs mine {vals(r.my_values)}"
                         + (f" — in {r.datasets_differing.replace(';', ', ')}" if kind == "varies" else ""))
        L.append("")
    L.append("Architecture keys (`gnn_type`, `d_model`, …) have no TabDDPM counterpart and are not classified; "
             "they are in the per-dataset tables.\n")

    L.append("### Parameter counts\n")
    L.append("Trainable parameters of the denoiser, instantiated with the dataset's real shapes (`make_dataset` with "
             "the training transformations; d_num includes y for regression). The count is identical for the three "
             "rules in every dataset" + ("" if pc_df.rules_same_params.all() else " **except where marked**") +
             ". Figure: `figures/param_counts.png`.\n")
    L.append("| dataset | subset | nodes | TabDDPM d_layers | TabDDPM params | my architecture | my params | mine / TabDDPM |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in order:
        p = pc_df.set_index("dataset").loc[ds]
        L.append(f"| {ds} | {p.subset} | {p.n_nodes} | {short_layers(extras[ds]['d_layers'])} | {p.params_tabddpm:,} | "
                 f"{p.my_architecture} | {int(p.params_mine):,} | {p.ratio_mine_over_tabddpm:.2f} |")
    L.append(f"\nMedian ratio {pc_df.ratio_mine_over_tabddpm.median():.2f} (range "
             f"{pc_df.ratio_mine_over_tabddpm.min():.2f}–{pc_df.ratio_mine_over_tabddpm.max():.2f}). The GNN's size "
             "barely depends on the number of nodes (weights are shared across nodes); the MLP's grows with its "
             "tuned width and depth.\n")

    L.append("## Link to results (descriptive; n is small)\n")
    L.append("Δ = my 50-run mean − TabDDPM's `eval_catboost.json` synthetic/test mean (macro-F1 or R², 50 runs). "
             "Ratios are mine / TabDDPM. Only cells with 5 complete gen seeds.\n")
    L.append("| dataset | subset | metric | TabDDPM | Δ fixed | Δ Jenks | Δ entmax | steps | lr | batch | synth. size | params |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in order:
        d = rl[rl.dataset == ds].set_index("rule")
        f0 = d.iloc[0]
        cells = [f"{d.loc[r, 'delta']:+.4f}" if np.isfinite(d.loc[r, "delta"]) else "—" for r in RULES]
        L.append(f"| {ds} | {f0.subset} | {f0.metric} | {f0.tabddpm_eval_catboost:.4f} | " + " | ".join(cells)
                 + f" | {f0.steps_ratio:.2f} | {f0.lr_ratio:.2f} | {f0.batch_ratio:.2f} | {f0.samples_ratio:.2f} | "
                 f"{d.loc['fixed', 'param_ratio']:.2f} |")
    L.append("\n**Spearman ρ of Δ with each ratio — exploratory**, no multiple-comparison correction, n ≤ "
             f"{int(sp.n.max())}:\n")
    L.append("| rule | subset | n | steps | lr | batch | synth. size | params |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in RULES:
        for sub in ("held-out", "pooled"):
            d = sp[(sp.rule == r) & (sp.subset == sub)].set_index("ratio")
            cells = [(f"{d.loc[x, 'rho']:+.2f} (p={d.loc[x, 'p']:.2f})" if np.isfinite(d.loc[x, "rho"])
                      else d.loc[x, "note"]) for x in RATIOS]
            L.append(f"| {ct.RULE_LABEL[r]} | {sub} | {int(d.n.iloc[0])} | " + " | ".join(cells) + " |")
    L.append("")
    L += house_paragraph(rl, pc_df, extras)

    L.append("## Per-dataset side-by-side tables\n")
    L.append("Bold with ≠ = the two sides differ. `—` = no counterpart. Rule `all` = identical for the three rules.\n")
    for ds in order:
        L.append(f"### {ds} ({extras[ds]['subset']})\n")
        L += md_table(df[df.dataset == ds])
        L.append("")

    L.append("## Caveats\n")
    L.append("- TabDDPM tuned per dataset on validation (Optuna); mine: architecture grid (development, selected on "
             "test) or transfer (held-out), training hyperparameters untuned. See the caveat at the top.")
    L.append("- The TabDDPM numbers are the authors' `eval_catboost.json` shipped in `exp/<ds>/ddpm_cb_best/` (the "
             "paper's Table 5 rounds them to 3 decimals); not re-run here.")
    L.append("- The two sides sample different checkpoints: TabDDPM's `eval_seeds.py` copies `model.pt` (raw weights), "
             "the runner samples `model_ema.pt`.")
    L.append("- My configuration values are read from the code as it is now; the runs record no git commit "
             "(`git_commit: unknown`). Checked with git: the assignments of `ABLATION_LR`, `ABLATION_STEPS`, "
             "`ABLATION_BATCH_SIZE` last changed in 220838d (2026-06-11), the earliest run here started 2026-09-24, "
             "and the uncommitted diff of `ablation_runner.py` does not touch them. One exception is recorded: "
             "fb-comments' `sample_batch_size` (150 000 → 50 000, commit 385a768), which changes only how sampling "
             "is batched.")
    L.append("- Parameter counts are for the denoiser only; the diffusion wrapper has no trainable parameters.")
    bad = df[(df.key == "n_cat_features") & (df.differs == True)].dataset.unique()  # noqa: E712
    for ds in bad:
        r = df[(df.dataset == ds)].set_index("key")
        L.append(f"- **{ds}: the local data is not the dataset the authors' config was tuned and evaluated on.** "
                 f"Their `info.json`: {r.loc['n_num_features', 'tabddpm_value']} numerical + "
                 f"{r.loc['n_cat_features', 'tabddpm_value']} categorical, val/test {r.loc['val_size', 'tabddpm_value']}/"
                 f"{r.loc['test_size', 'tabddpm_value']}; local `data/{ds}/`: {r.loc['n_num_features', 'my_value']} + "
                 f"{r.loc['n_cat_features', 'my_value']}, val/test {r.loc['val_size', 'my_value']}/"
                 f"{r.loc['test_size', 'my_value']} (`num_numerical_features` 20 → 14 in commit 385a768 to match the "
                 "local data). Its TabDDPM parameter count here uses the LOCAL shapes, and its Δ vs TabDDPM (here and "
                 "in the paper comparison) is not like-for-like: the test set and the column typing differ.")
    (HERE / "CONFIG_COMPARISON.md").write_text("\n".join(L) + "\n")


def house_paragraph(rl, pc_df, extras) -> list[str]:
    e = extras["house"]
    d = rl[rl.dataset == "house"].set_index("rule")
    p = pc_df.set_index("dataset").loc["house"]
    deltas = ", ".join(f"{ct.RULE_LABEL[r]} {d.loc[r, 'delta']:+.3f}" for r in RULES if np.isfinite(d.loc[r, "delta"]))
    ho = rl[rl.subset == "held-out"].drop_duplicates("dataset").set_index("dataset")
    with_res = set(rl[(rl.subset == "held-out") & rl.delta.notna()].dataset)
    rank_res = sorted(with_res, key=lambda x: ho.loc[x, "lr_ratio"])
    lr_txt = (f"{e['lr_m'] / e['lr_t']:.2f}×, "
              + ("the smallest ratio among held-out datasets with results" if rank_res[0] == "house"
                 else "held-out datasets with results")
              + ": " + ", ".join(f"{x} {ho.loc[x, 'lr_ratio']:.2f}×" for x in rank_res if x != "house")
              + "; without results: " + ", ".join(f"{x} {ho.loc[x, 'lr_ratio']:.2f}×" for x in ho.index if x not in with_res))
    return ["### house\n",
            f"house is numerical-only (16 features + y as a numerical node = {e['n_nodes']} nodes), so the "
            "categorical-input fix does not touch it. My runs are below TabDDPM's own result by "
            f"{deltas} R² (TabDDPM {d.iloc[0].tabddpm_eval_catboost:.4f}). The largest configuration differences there: "
            f"**{e['steps_t']:,} → {e['steps_m']:,} steps** ({e['steps_m'] / e['steps_t']:.2f}×), **lr "
            f"{e['lr_t']:.2e} → {e['lr_m']:.0e}** ({lr_txt}), and a denoiser "
            f"{p.ratio_mine_over_tabddpm:.2f}× TabDDPM's size ({int(p.params_mine):,} vs {p.params_tabddpm:,}). "
            + (f"Batch size ({e['batch_m']}), synthetic size" if e["batch_t"] == e["batch_m"]
               else f"Batch {e['batch_t']} → {e['batch_m']}; synthetic size")
            + ", normalization, diffusion and evaluation settings are the same (as everywhere, mine samples the "
            "EMA weights and TabDDPM the raw ones); "
            "both sides decay the lr linearly to 0, so steps × initial lr "
            f"is {e['steps_m'] * e['lr_m'] / (e['steps_t'] * e['lr_t']):.2f}× TabDDPM's. With n = 1 dataset this "
            "is a description, not a diagnosis.\n"]


if __name__ == "__main__":
    main()
