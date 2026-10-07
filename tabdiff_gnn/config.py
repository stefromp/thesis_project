"""
Route A -- AttnGNN as the denoiser INSIDE TabDiff's framework. The one place
that defines datasets, arms, names, paths, pins and deviations; everything
else in tabdiff_gnn/ imports from here.

What the study is
  TabDiff (third_party/TabDiff @ TABDIFF_COMMIT, never modified) is run through
  its own classes -- UnifiedCtimeDiffusion (EDM numerical + masked categorical
  diffusion, learnable per-column schedules), Model/Precond, Trainer (training
  loop, EMA, best-EMA checkpoint after epoch 4000, sampler), TabDiffDataset,
  TabMetrics -- with ONE substitution: the denoiser.

    transformer  TabDiff's own UniModMLP -- code path kept and tested (bit-identity
                 with TabDiff's model; the mirrored entry point builds main.py's
                 model) as evidence that the framework is TabDiff's; NOT run:
                 TabDiff numbers are the reported ones (GATD's 3x20 re-run first,
                 the TabDiff paper second)
    attngnn      V2, d_model 256 on every dataset: TabDiff Tokenizer at d_model ->
                 one shared noise embedding
                 added to every node -> AttnGNN blocks (tab_ddpm.graph_denoiser.
                 AttnGNNBlock, imported unchanged: dynamic adjacency, fixed top-k,
                 no global attention) -> TabDiff Reconstructor at d_model.
                 Every cross-column interaction goes through the learned sparse
                 graph; TabDiff's transformers and flattened MLP are gone.

Protocol (GATD's extension of TabDiff's): TabDiff's tabdiff_configs.toml
unchanged (8000 epochs, batch 4096, lr 1e-3 + ReduceLROnPlateau, EMA 0.997,
best EMA loss after epoch 4000, 50 sampling steps), 3 training seeds x 20
tables (seed 0 first; seeds 1-2 only after confirmation). No DCR in this study
(GATD's protocol does not use it; no _dcr models are trained). Evaluation:
TabDiff's evaluator through
hpc_transfer/tabdiff_study/tabdiff_eval.py (digests checked) + wNMIS and
C2ST (XGBoost) from april-tools/tabpc (third_party/tabpc @ TABPC_COMMIT).

Trees -- dedicated, never mixed with the other studies
  data     data/tabdiff_gnn/tabdiff_<ds>/                (prepare_data.py)
  results  <exp_root>/exp_tabdiff_gnn/tabdiff_<ds>/<denoiser>/<arm>/seed<s>/
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STUDY_DIR = os.path.join(REPO, "hpc_transfer", "tabdiff_study")
if STUDY_DIR not in sys.path:
    # appended, not prepended: the study has its own runner.py / make_jobs.py /
    # prepare_data.py, which must never shadow this package's modules
    sys.path.append(STUDY_DIR)
import study as S  # noqa: E402  (the TabDiff-protocol study: pins, GNN configs)

STUDY = "tabdiff_gnn_route_a"

# ---------------------------------------------------------------------------
# Datasets. diabetes is last in every order (largest, most expensive).
# ---------------------------------------------------------------------------
ORDER = ["magic", "shoppers", "adult", "default", "beijing", "news", "diabetes"]
TASK = {**S.TASK, "diabetes": "binclass"}
# GNN hyper-parameters come from the TabDiff study's source mapping
# (study.SOURCES -> hpc_transfer/configs/<src>.sh). diabetes has no source
# there (the TabDiff study dropped it); adult -- the categorical-heavy
# binary-classification source -- is used for it.
SOURCES = {**S.SOURCES, "diabetes": "adult"}
SOURCE_NOTES = {"diabetes": "our choice: diabetes has no TabDDPM source (not in the TabDiff "
                            "study's mapping); adult's GNN configuration is used "
                            "(categorical-heavy binary classification, as diabetes)"}
# No DCR in this study: only the main model of each dataset is trained (TabDiff's
# _dcr splits and the TabDiff study's DCR pipeline are not used here).
PREFIX = "tabdiff_"
_TARGET_RE = re.compile(r"^tabdiff_(%s)$" % "|".join(ORDER))

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
DENOISERS = ("transformer", "attngnn")
VARIANT = "V2"
DIM_T = 128
NODE_NOISE_CHOICES = ("shared", "per_column")
NODE_NOISE = "shared"                    # main runs; per_column implemented, off
D_MODEL = 256                            # one AttnGNN width for every dataset (replaces the
                                         # source d_model and the former d384 capacity arm)
TRAIN_SEEDS = (0, 1, 2)                  # GATD: 3 training seeds ...
N_TABLES = 20                            # ... x 20 tables (generation seeds 0..19)
# Trainer per dataset: TabDiff's own Trainer everywhere, except news, whose 8000
# epochs do not fit one 48 h job -> resumable.ResumableTrainer (TabDiff's run_loop
# copied verbatim + save/restore hooks; bit-identity proven by the tests).
RESUMABLE_DATASETS = ("news",)
TRAINER = {ds: ("resumable" if ds in RESUMABLE_DATASETS else "tabdiff") for ds in ORDER}
TRAINER_EVIDENCE = ("tests/test_tabdiff_gnn.py::test_resumable_run_loop_is_tabdiffs_verbatim, "
                    "::test_resume_is_bit_identical (CPU; 40 epochs), "
                    "::test_resume_is_bit_identical_through_best_checkpoints (CPU; past epoch 4000, RUN_SLOW=1)")
GNN_FIXED = {"graph_mode": "dynamic", "topk_mode": "fixed", "use_attention": False,
             "dropout": 0.0}

RESULTS_DIRNAME = "exp_tabdiff_gnn"
DATA_REL = os.path.join("data", "tabdiff_gnn")

# ---------------------------------------------------------------------------
# TabDiff code: the study's evaluator pins + every file Route A EXECUTES
# (model, diffusion, schedules, entry point mirrored, config, src/).
# ---------------------------------------------------------------------------
TABDIFF_REL = S.TABDIFF_REL
TABDIFF_COMMIT = S.TABDIFF_COMMIT
ROUTE_A_TABDIFF_FILES = {   # sha256, first 16 hex digits
    "tabdiff/modules/main_modules.py":           "9d9ee6fb344cffdb",
    "tabdiff/modules/transformer.py":            "f3197ebb0af083c2",
    "tabdiff/models/unified_ctime_diffusion.py": "aa341c5c31e02fbe",
    "tabdiff/models/noise_schedule.py":          "50006962741c2922",
    "tabdiff/main.py":                           "1d437caf45b2d17e",
    "tabdiff/configs/tabdiff_configs.toml":      "38eff74ecb572c9b",
    "src/__init__.py":                           "9eb081650d6c073f",
    "src/data.py":                               "4a0ac7eda23e3968",
    "src/env.py":                                "a3a9452d30393fd3",
    "src/metrics.py":                            "fb9a08b226a8d0c2",
    "src/util.py":                               "0420fd21738f9706",
    "data/Info/diabetes.json":                   "19012a8ebf0b184c",
}
ALL_TABDIFF_FILES = {**S.PINNED_TABDIFF_FILES, **ROUTE_A_TABDIFF_FILES}

# tabpc (april-tools): wNMIS = MI_l1()["nmi_l1_weighted_complement"],
# C2ST (XGB) = C2ST()["xgboost (C2ST)"] -- the keys its article tables use
# (scripts/update_article_tables.py, scripts/plot_main_experiment.py).
TABPC_REL = os.path.join("third_party", "tabpc")
TABPC_URL = "https://github.com/april-tools/tabpc"
TABPC_COMMIT = "b821a47a9ca9f674f1083d1309b1e7e548657c0d"
PINNED_TABPC_FILES = {
    "src/metrics/metrics.py":                "124580caaa19a4cd",
    "src/util.py":                           "c5bd1d2435928567",
    "tabdiff_eval/__init__.py":              "e3b0c44298fc1c14",
    "tabdiff_eval/mle/mle.py":               "81a6540be194531b",
    "tabdiff_eval/mle/tabular_dataload.py":  "092d55ad5df6452f",
    "tabdiff_eval/mle/tabular_transformer.py": "756d04737c8a0914",
}
TABPC_KEYS = {"wnmis": ("mi_l1", "nmi_l1_weighted_complement"),
              "c2st_xgb": ("C2ST", "xgboost (C2ST)"),
              "c2st_lr_tabpc": ("C2ST", "logistic_regression (C2ST)")}
# tabpc's requirements.txt pins sdmetrics==0.24.0 and torch==2.9.1 (CPU wheel
# used: the MI matrix is device-independent); the rest are what pip resolved
# with those pins (recorded per run in tabpc_meta.json).
TABPC_PINS = {"sdmetrics": "0.24.0", "torch": "2.9.1"}

# ---------------------------------------------------------------------------
# Deviations from TabDiff's own protocol (run_meta.json + analysis)
# ---------------------------------------------------------------------------
DEVIATIONS = [
    "routine evaluation every check_val_every=2000 epochs disabled (no effect on "
    "training or checkpoint selection; it only sampled/scored intermediate models)",
    "training seed set explicitly (torch/numpy/random) -- TabDiff seeds only under "
    "--deterministic; deterministic CUDA algorithms are NOT enabled (as TabDiff's default)",
    "each of the 20 tables is drawn after seeding with its generation seed k=0..19 "
    "(TabDiff's report_test draws them back-to-back from one RNG stream)",
    "entry point mirrors tabdiff/main.py (same classes and calls) instead of calling it: "
    "main.py hard-codes UniModMLP (main.py:185), writes checkpoints into the submodule "
    "and needs wandb; a test proves the transformer model is identical to main.py's",
    "evaluation adaptations of hpc_transfer/tabdiff_study/tabdiff_eval.py (MLE global "
    "RNG seeded per table, paths); MLE-2 (F1 / R2) read from the same MLE call",
    "wandb replaced by a no-op logger (Trainer only calls define_metric/log)",
]
DEVIATIONS_RESUMABLE = [
    "news only: trained with resumable.ResumableTrainer -- TabDiff's Trainer.run_loop copied verbatim "
    "plus two save/restore hooks -- across chained jobs (8000 epochs do not fit 48 h); evidence: "
    + TRAINER_EVIDENCE,
]
DEVIATIONS_ATTNGNN = [
    "denoiser: AttnGNN V2 instead of UniModMLP (the variable under study)",
    f"AttnGNN width: d_model {D_MODEL} on every dataset (gnn type, layers, top_k, heads from the "
    "source mapping; the source runs used d_model 64/128)",
]


# ---------------------------------------------------------------------------
# Names and trees
# ---------------------------------------------------------------------------
def target_name(ds: str) -> str:
    if ds not in ORDER:
        raise ValueError(f"{ds!r} is not a Route-A dataset ({ORDER})")
    return f"{PREFIX}{ds}"


def parse_target(name: str) -> str:
    """'tabdiff_adult' -> 'adult'. Refuses unprefixed names and *_dcr (no DCR here)."""
    m = _TARGET_RE.match(name or "")
    if not m:
        raise ValueError(f"{name!r} is not a Route-A target: 'tabdiff_<ds>', <ds> in {ORDER} "
                         f"(no _dcr targets: DCR is not part of this study)")
    return m.group(1)


def tabdiff_dataname(target: str) -> str:
    return parse_target(target)


def all_targets():
    return [target_name(ds) for ds in ORDER]


def data_dir(target: str, repo_root: str = REPO) -> str:
    parse_target(target)
    return os.path.join(repo_root, DATA_REL, target)


def data_files(target: str, repo_root: str = REPO) -> dict:
    """What TabDiff reads: the processed npy directory (TabDiffDataset) and the
    CSVs TabMetrics reads (real = train split, test, and val only where
    TabDiff has one -- diabetes)."""
    d = data_dir(target, repo_root)
    val = os.path.join(d, "val.csv")
    return {"dir": d, "info": os.path.join(d, "info.json"), "real": os.path.join(d, "real.csv"),
            "test": os.path.join(d, "test.csv"), "val": val if os.path.exists(val) else None}


def results_root(exp_root: str) -> str:
    return os.path.join(exp_root, RESULTS_DIRNAME)


def gnn_config(ds: str, repo_root: str = REPO):
    """(values, sources) of the AttnGNN for ds: the source's GNN configuration
    (study.gnn_config: gnn type, layers, top_k, heads) with d_model = D_MODEL."""
    v, s = S.gnn_config(SOURCES[ds], repo_root)
    v, s = dict(v), dict(s)
    s["d_model"] = f"Route A: d_model {D_MODEL} on every dataset (source value {v['d_model']} at {s['d_model']})"
    v["d_model"] = D_MODEL
    if v["d_model"] % v["n_heads"]:
        raise ValueError(f"{ds}: d_model {v['d_model']} not divisible by n_heads {v['n_heads']}")
    return v, s


def arm_name(denoiser: str, ds: str | None = None, node_noise: str = NODE_NOISE,
             repo_root: str = REPO) -> str:
    if denoiser == "transformer":
        return "unimodmlp_tabdiff_config"
    if denoiser != "attngnn":
        raise ValueError(f"denoiser must be one of {DENOISERS}")
    g, _ = gnn_config(ds, repo_root)
    return (f"v2_{g['gnn_type']}_layers{g['n_layers']}_dim{g['d_model']}_topk{g['top_k']}"
            f"_heads{g['n_heads']}_dimt{DIM_T}_noise-{node_noise}")


def run_dir(exp_root: str, target: str, denoiser: str, seed: int,
            node_noise: str = NODE_NOISE, repo_root: str = REPO) -> str:
    ds = parse_target(target)
    path = os.path.join(results_root(exp_root), target, denoiser,
                        arm_name(denoiser, ds, node_noise, repo_root), f"seed{seed}")
    assert_in_tree(path, exp_root)
    return path


def assert_in_tree(path: str, exp_root: str) -> None:
    root = os.path.realpath(results_root(exp_root))
    real = os.path.realpath(path)
    if not (real == root or real.startswith(root + os.sep)):
        raise ValueError(f"{path} is outside the Route-A tree {root}")
    rel = os.path.relpath(real, root).split(os.sep)
    if rel[0] != ".":
        parse_target(rel[0])
        if len(rel) > 1 and rel[1] not in DENOISERS:
            raise ValueError(f"{path}: second level must be a denoiser {DENOISERS}")


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _verify(root: str, pins: dict, what: str, commit: str) -> dict:
    bad = []
    for rel, want in pins.items():
        p = os.path.join(root, rel)
        got = sha16(p) if os.path.exists(p) else "missing"
        if got != want:
            bad.append(f"{rel}: {got} != pinned {want}")
    if bad:
        raise RuntimeError(f"{root} is not {what} @ {commit} unchanged:\n  " + "\n  ".join(bad))
    return {"commit": commit, "files_sha256_16": dict(pins)}


def verify_tabdiff(repo_root: str = REPO) -> dict:
    """Every TabDiff file the study pins AND every file Route A executes."""
    out = _verify(os.path.join(repo_root, TABDIFF_REL), ALL_TABDIFF_FILES, "TabDiff", TABDIFF_COMMIT)
    out["url"] = S.TABDIFF_URL
    return out


def verify_tabpc(repo_root: str = REPO) -> dict:
    out = _verify(os.path.join(repo_root, TABPC_REL), PINNED_TABPC_FILES, "tabpc", TABPC_COMMIT)
    out["url"] = TABPC_URL
    return out
