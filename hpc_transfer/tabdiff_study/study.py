"""
TabDiff-transfer study -- the one place that defines targets, sources, names,
paths and the transferred values. Everything else in hpc_transfer/tabdiff_study
imports from here.

What the study is
  The six TabDiff benchmark datasets (diabetes dropped: no TabDDPM source, by
  far the most expensive), on TabDiff's OWN splits (their process_dataset.py,
  run unchanged by prepare_data.py), evaluated with TabDiff's OWN evaluation
  code (third_party/TabDiff at TABDIFF_COMMIT, called unchanged by
  tabdiff_eval.py). Each dataset borrows, from its nearest TabDDPM dataset of
  the same task type ("source", Step 0.4 of the plan):
    * the TabDDPM authors' tuned training setup of the source
      (exp/<src>/ddpm_cb_best/config.toml, read by
      ablation_runner.training_setup(..., "tabddpm") exactly as the
      budget-matched runs do), plus its [train.T] normalization and
      [model_params] is_y_cond;
    * the source's GNN architecture (hpc_transfer/configs/<src>.sh, the
      configuration analysis/paper_tables reports for it).

Names and trees -- enforced, never mixed with the TabDDPM datasets
  data      data/tabdiff_<ds>[_dcr]/                  (prepare_data.py)
  results   <exp_root>/exp_tabdiff/tabdiff_<ds>[_dcr]/gnn/<combo>/
  The "tabdiff_" prefix is mandatory (parse_target refuses anything else) and
  the results root is always the dedicated "exp_tabdiff" directory, so a
  tabdiff_adult run can never land in, or be read as, exp/adult/.

Not a TabDiff run: the old *_tabdiff_split folders, ablation/eval_xgboost.py,
ablation/eval_tabdiff_fidelity.py and notebooks_tabdiff/ are NOT used here.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

STUDY = "tabdiff_transfer"

# ---------------------------------------------------------------------------
# TabDiff code: pinned commit + digests of every file this study executes.
# The cluster upload has no git history, so tabdiff_eval.py proves the code is
# unchanged by hashing these files before it runs anything.
# ---------------------------------------------------------------------------
TABDIFF_REL = os.path.join("third_party", "TabDiff")
TABDIFF_URL = "https://github.com/MinkaiXu/TabDiff"
TABDIFF_COMMIT = "5ecdb3356261aea72716cc9a779f31d7ad083bf4"
PINNED_TABDIFF_FILES = {   # sha256, first 16 hex digits
    "tabdiff/metrics.py":         "06bb3be588cdc7f1",
    "tabdiff/trainer.py":         "45c177d86a96e7a8",
    "utils_train.py":             "c1e8927f7781cc39",
    "eval/mle/mle.py":            "7d1d87113590a939",
    "eval/visualize_density.py":  "e5564eae40da4c29",
    "eval/eval_quality.py":       "377eab8a08fb9dec",
    "process_dataset.py":         "5a8d1d3630576ba4",
    "download_dataset.py":        "506c4e102db3944a",
    "data/Info/adult.json":       "f83fcc5a962c99fc",
    "data/Info/default.json":     "2669efb6d250a6f0",
    "data/Info/shoppers.json":    "89121f783371b0c4",
    "data/Info/magic.json":       "eaad447cc8d44307",
    "data/Info/beijing.json":     "9db9fdbf0db77f46",
    "data/Info/news.json":        "c9696c824190abbd",
    "data/Info/adult_dcr.json":   "efb5d750127d044a",
    "data/Info/default_dcr.json": "24922fcc1495ede0",
    "data/Info/shoppers_dcr.json": "56b154d60388a691",
    "data/Info/beijing_dcr.json": "15ddeda8df286d1d",
    "data/Info/news_dcr.json":    "4d51c5dea9aca4ba",
}

# Evaluator package pins (TabDiff's tabdiff.yaml pins neither). Chosen as the
# releases current when TabDiff's evaluation code was committed (2025-02-16/19):
# sdmetrics 0.18.0 (2024-12-13; 0.19.0 came 2025-02-25), xgboost 2.1.4
# (2025-02-06; 3.x removed the tree_method='gpu_hist' that eval/mle/mle.py
# uses). sdmetrics 0.28 breaks LogisticDetection on TabDiff's integer column
# labels and changes Trend (shoppers real-vs-real 99.23 vs 98.76 on 0.18.0).
EVAL_PINS = {"sdmetrics": "0.18.0", "xgboost": "2.1.4"}
QUALITY_PINS = {"synthcity": "0.2.12"}

# ---------------------------------------------------------------------------
# Targets, sources, protocol
# ---------------------------------------------------------------------------
PREFIX = "tabdiff_"
DCR_SUFFIX = "_dcr"
RESULTS_DIRNAME = "exp_tabdiff"

# TabDiff dataset -> TabDDPM source (nearest same-task neighbour in the
# existing transfer space; confirmed by the user 2026-10-06).
SOURCES = {
    "magic":    "wilt",
    "shoppers": "adult",
    "adult":    "adult",
    "default":  "default",
    "beijing":  "king",
    "news":     "king",
}
# Cheapest first, so a problem shows up early (submission order).
ORDER = ["magic", "shoppers", "adult", "default", "beijing", "news"]
TASK = {"magic": "binclass", "shoppers": "binclass", "adult": "binclass",
        "default": "binclass", "beijing": "regression", "news": "regression"}
# TabDiff reports DCR for every dataset but magic (no data/Info/magic_dcr.json,
# no Magic column in their Table 10).
HAS_DCR = {ds: ds != "magic" for ds in SOURCES}

# TabDiff protocol (main.py --num_runs default 20; paper Sec. 4.1 "the average
# of 20 random sampled synthetic data"); each table has n_train rows
# (trainer.py evaluate_generation: num_samples = real_data_size).
N_SAMPLE_RUNS = 20

# Model, as in the final runs (the GNN denoiser is the only model of the study).
MODEL_DIR = "gnn"
CAT_INPUT = "prob"
TOPK_MODE = "fixed"
GRAPH_MODE = "dynamic"
USE_ATTENTION = False
TRAIN_SETUP = "tabddpm"          # ablation_runner.training_setup mode

# Deviations from TabDiff's own protocol, recorded in every run_meta.json and
# in the analysis output.
DEVIATIONS = [
    "checkpoint: our final EMA weights (model_ema.pt after the last step), not "
    "TabDiff's 'best EMA training loss after epoch 4000' (tabdiff/trainer.py run_loop)",
    "training setup: the source's TabDDPM-tuned lr/steps/batch/T/scheduler, not "
    "TabDiff's tabdiff_configs.toml",
    "MLE val split: TabDiff's prepare_ml_problem shuffles with the global numpy RNG "
    "unseeded; tabdiff_eval.py seeds it with the run index before each table so the "
    "evaluation is reproducible (the code itself is unchanged)",
    "our loader needs val/test files: data/tabdiff_<ds>/X_*_val.npy is a copy of "
    "TabDiff's test split; training and normalisation read only the train split",
]

_TARGET_RE = re.compile(r"^tabdiff_(%s)(_dcr)?$" % "|".join(SOURCES))


def target_name(ds: str, dcr: bool = False) -> str:
    if ds not in SOURCES:
        raise ValueError(f"{ds!r} is not a TabDiff-study dataset ({sorted(SOURCES)})")
    if dcr and not HAS_DCR[ds]:
        raise ValueError(f"{ds} has no DCR split in TabDiff's protocol")
    return f"{PREFIX}{ds}{DCR_SUFFIX if dcr else ''}"


def parse_target(name: str):
    """'tabdiff_adult_dcr' -> ('adult', True). Refuses an unprefixed name, so a
    TabDDPM dataset name can never be passed where a study target is meant."""
    m = _TARGET_RE.match(name or "")
    if not m:
        raise ValueError(
            f"{name!r} is not a TabDiff-study target: names must be "
            f"'{PREFIX}<ds>' or '{PREFIX}<ds>{DCR_SUFFIX}' with <ds> in {sorted(SOURCES)}")
    ds, dcr = m.group(1), bool(m.group(2))
    if dcr and not HAS_DCR[ds]:
        raise ValueError(f"{name}: {ds} has no DCR split in TabDiff's protocol")
    return ds, dcr


def tabdiff_dataname(target: str) -> str:
    """TabDiff's own name for the split: 'adult', 'adult_dcr', ..."""
    ds, dcr = parse_target(target)
    return ds + (DCR_SUFFIX if dcr else "")


def data_dir(target: str, repo_root: str = REPO) -> str:
    parse_target(target)
    return os.path.join(repo_root, "data", target)


def eval_inputs(target: str, repo_root: str = REPO) -> dict:
    """TabDiff's processed files for the target (what their evaluator reads)."""
    d = os.path.join(data_dir(target, repo_root), "tabdiff")
    return {"info": os.path.join(d, "info.json"),
            "real": os.path.join(d, "real.csv"),
            "test": os.path.join(d, "test.csv")}


def results_root(exp_root: str) -> str:
    return os.path.join(exp_root, RESULTS_DIRNAME)


def combo_name(ds: str, repo_root: str = REPO) -> str:
    """Run directory name: the GNN combo name as everywhere else
    (ablation_runner.exp_dir_name)."""
    g = gnn_config(SOURCES[ds], repo_root)[0]
    ar = _ablation_runner(repo_root)
    return ar.exp_dir_name(g["gnn_type"], g["n_layers"], g["d_model"], g["top_k"],
                           g["n_heads"], g["use_attention"])


def run_dir(exp_root: str, target: str, repo_root: str = REPO) -> str:
    """<exp_root>/exp_tabdiff/<target>/gnn/<combo>. The only way the study
    builds a results path."""
    ds, _ = parse_target(target)
    path = os.path.join(results_root(exp_root), target, MODEL_DIR, combo_name(ds, repo_root))
    assert_in_study_tree(path, exp_root)
    return path


def assert_in_study_tree(path: str, exp_root: str) -> None:
    root = os.path.realpath(results_root(exp_root))
    real = os.path.realpath(path)
    if not (real == root or real.startswith(root + os.sep)):
        raise ValueError(f"{path} is outside the study tree {root}")
    rel = os.path.relpath(real, root).split(os.sep)
    if rel[0] != ".":
        parse_target(rel[0])


# ---------------------------------------------------------------------------
# Transferred values (value + file:line for each)
# ---------------------------------------------------------------------------
def _ablation_runner(repo_root: str = REPO):
    for p in (os.path.join(repo_root, "ablation"), os.path.join(repo_root, "scripts"), repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("ablation_runner")


def _rel(path: str, repo_root: str) -> str:
    return os.path.relpath(path, repo_root)


def _toml_lookup(repo_root: str, src: str, section: str, key: str):
    ar = _ablation_runner(repo_root)
    rel = os.path.join(*ar.TABDDPM_CONFIG_REL).format(dataset=src)
    path = os.path.join(repo_root, rel)
    cfg = ar._load_toml(path)
    node = cfg
    for part in section.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    if key not in node:
        raise KeyError(f"{rel} lacks [{section}] {key}; refusing to guess")
    with open(path) as fh:
        line = ar._toml_line(fh.read(), section, key)
    return node[key], f"{rel}:{line} [{section}] {key}"


def train_config(src: str, repo_root: str = REPO):
    """The TabDDPM authors' training setup of `src`, through the SAME function
    the budget-matched runs use (ablation_runner.training_setup, "tabddpm"),
    plus the preprocessing / conditioning keys of the same config.toml."""
    ar = _ablation_runner(repo_root)
    mod = importlib.import_module(src.replace("-", "_") + "_ablation")
    values, sources = ar.training_setup(src, mod.DATASET_CFG, TRAIN_SETUP, repo_root)
    values, sources = dict(values), dict(sources)
    norm, s = _toml_lookup(repo_root, src, "train.T", "normalization")
    values["normalization"], sources["normalization"] = (None if norm == "__none__" else norm), s
    ycond, s = _toml_lookup(repo_root, src, "model_params", "is_y_cond")
    values["is_y_cond"], sources["is_y_cond"] = bool(ycond), s
    return values, sources


_GNN_KEYS = {"GNN_TYPE": ("gnn_type", str), "N_LAYERS": ("n_layers", int),
             "D_MODEL": ("d_model", int), "TOP_K": ("top_k", int),
             "N_HEADS": ("n_heads", int), "USE_ATTENTION": ("use_attention", int)}


def gnn_config(src: str, repo_root: str = REPO):
    """The source's GNN architecture from hpc_transfer/configs/<src>.sh -- the
    configuration analysis/paper_tables reports for it (checked by the tests)."""
    path = os.path.join(repo_root, "hpc_transfer", "configs", f"{src}.sh")
    values, sources = {}, {}
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            m = re.match(r'^([A-Z_]+)=("?)([^"#\s]*)\2', line)
            if m and m.group(1) in _GNN_KEYS:
                name, typ = _GNN_KEYS[m.group(1)]
                values[name] = typ(m.group(3))
                sources[name] = f"{_rel(path, repo_root)}:{i} {m.group(1)}"
    missing = [n for n, _ in _GNN_KEYS.values() if n not in values]
    if missing:
        raise KeyError(f"{path} lacks {missing}")
    values["use_attention"] = bool(values["use_attention"])
    if values["use_attention"] != USE_ATTENTION:
        raise ValueError(f"{path}: USE_ATTENTION must be 0 for this study")
    return values, sources


def transferred(ds: str, repo_root: str = REPO):
    """(values, sources) of everything this target borrows from its source."""
    src = SOURCES[ds]
    values, sources = train_config(src, repo_root)
    v, s = gnn_config(src, repo_root)
    values.update(v)
    sources.update(s)
    return values, sources


# ---------------------------------------------------------------------------
# TabDiff code integrity
# ---------------------------------------------------------------------------
def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_tabdiff(tabdiff_root: str) -> dict:
    """Refuse to run on a modified or incomplete TabDiff checkout."""
    bad = []
    for rel, want in PINNED_TABDIFF_FILES.items():
        p = os.path.join(tabdiff_root, rel)
        got = sha16(p) if os.path.exists(p) else "missing"
        if got != want:
            bad.append(f"{rel}: {got} != pinned {want}")
    if bad:
        raise RuntimeError(
            f"{tabdiff_root} is not TabDiff @ {TABDIFF_COMMIT} unchanged:\n  " + "\n  ".join(bad)
            + f"\nupload third_party/TabDiff from the submodule at that commit")
    return {"commit": TABDIFF_COMMIT, "url": TABDIFF_URL, "files_sha256_16": dict(PINNED_TABDIFF_FILES)}
