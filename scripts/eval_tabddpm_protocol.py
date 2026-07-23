"""
Paper-faithful implementations of the original TabDDPM (Kotelnikov et al., 2023)
evaluation protocol, as GENERIC functions over any dataset in this pipeline.

Every public function takes ``dataset_name`` + a ``real_data_path`` (data/<name>/)
and one or more generated directories (each containing X_num_train.npy etc., as
written by sample.py). Task type (regression vs classification) is detected from
``info.json`` — nothing is hardcoded to a single dataset.

These live ALONGSIDE the existing metrics in eval_fidelity.py and the DCR code in
ablation_runner.py; none of that is modified.

Metrics
-------
1. compute_ml_efficiency_catboost : ML efficiency (F1 / R2), CatBoost, 5x10 seeds,
   synthetic-trained vs real-trained (both evaluated on the real test split).
2. compute_dcr_tabddpm_mean       : MEAN nearest-neighbour L2 distance, synthetic
   -> real TRAIN only (distinct from the median-distance and TabDiff-probability
   versions elsewhere in the repo).
3. compute_wasserstein_mean       : mean 1D Wasserstein distance over numerical
   columns (TabDDPM Table 9 reports numerical only).
4. compute_membership_inference_auc : black-box membership-inference ROC-AUC
   (Sec. 5.3). Best-faith reconstruction — see the function docstring for the
   assumptions, since the paper under-specifies the exact attack.

A per-dataset reference scaffold ``TABDDPM_PAPER_REFERENCE`` and a comparison
printer are provided; reference values are left empty until filled in.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict

import numpy as np


# ── dataset abbreviations (TabDDPM paper) ───────────────────────────────────────

NAME_TO_ABBREV = {
    "abalone": "AB", "adult": "AD", "buddy": "BU", "california": "CA",
    "cardio": "CAR", "churn2": "CH", "default": "DE", "diabetes": "DI",
    "fb-comments": "FB", "gesture": "GE", "higgs-small": "HI", "house": "HO",
    "insurance": "IN", "king": "KI", "miniboone": "MI", "wilt": "WI",
}

# Per-dataset published reference values. Keyed by the paper's abbreviation and
# left EMPTY until actual numbers are filled in (do not fabricate). Expected
# inner keys once populated:
#     "ml_f1" / "ml_r2", "ml_real"   (utility of synthetic vs real, mean)
#     "dcr", "wasserstein", "mia_auc"
# plus optional "*_std". See print_tabddpm_comparison() for how they are shown.
TABDDPM_PAPER_REFERENCE: dict[str, dict] = {ab: {} for ab in NAME_TO_ABBREV.values()}


# ── generic loading / encoding ──────────────────────────────────────────────────

def _load_info(real_data_path):
    import lib
    return lib.load_json(os.path.join(real_data_path, "info.json"))


def _read(path, split):
    """(X_num, X_cat, y) for a split, via the pipeline's standard loader."""
    import lib
    return lib.read_pure_data(path, split)


def _encode(fit_num, fit_cat, *transform_sets, normalize="minmax"):
    """Fit scaler/OHE on (fit_num, fit_cat); return list of encoded matrices for
    each (num, cat) pair in ``transform_sets``. Numeric -> min-max or standardize;
    categorical -> one-hot / sqrt(2) (matching the repo's DCR encoding)."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

    scaler = None
    if fit_num is not None and fit_num.shape[1] > 0:
        scaler = (MinMaxScaler() if normalize == "minmax" else StandardScaler())
        scaler.fit(fit_num.astype(float))

    ohe = None
    if fit_cat is not None and fit_cat.shape[1] > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(fit_cat)

    out = []
    for Xn, Xc in transform_sets:
        parts = []
        if scaler is not None:
            parts.append(scaler.transform(Xn.astype(float)))
        if ohe is not None:
            parts.append(ohe.transform(Xc) / np.sqrt(2))
        out.append(np.concatenate(parts, axis=1) if parts else np.empty((len(Xn), 0)))
    return out


# ── array-level cores (unit-testable without the pipeline) ───────────────────────

def _dcr_from_encoded(X_real_train, X_syn, aggregation="mean"):
    """Distance-to-closest-record: nearest-neighbour L2 from each synthetic row to
    the real TRAIN set, aggregated by ``aggregation`` ("mean" or "median")."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_real_train)
    d = nn.kneighbors(X_syn)[0][:, 0]
    return float(np.median(d) if aggregation == "median" else d.mean())


# Backwards-compatible alias (mean aggregation).
def _dcr_mean_from_encoded(X_real_train, X_syn):
    return _dcr_from_encoded(X_real_train, X_syn, aggregation="mean")


def _append_label(num, cat, y, task_type):
    """Fold the target into the (num, cat) matrices, matching the baseline DCR
    (resample_privacy.privacy_metrics / ablation_runner._compute_dcr): regression
    -> extra numeric column; classification -> extra string categorical column."""
    if task_type == "regression":
        yn = np.asarray(y, dtype=float)[:, None]
        num = yn if num is None or num.shape[1] == 0 else np.concatenate([num, yn], axis=1)
    else:
        yc = np.asarray(y).astype(int).astype(str)[:, None]
        cat = yc if cat is None or cat.shape[1] == 0 else np.concatenate([cat, yc], axis=1)
    return num, cat


def wasserstein_mean_arrays(real_num, syn_num, normalize="standard"):
    """Mean 1D Wasserstein distance across numerical columns."""
    from scipy.stats import wasserstein_distance
    if real_num is None or real_num.shape[1] == 0:
        return None
    dists = []
    for j in range(real_num.shape[1]):
        r = real_num[:, j].astype(float)
        s = syn_num[:, j].astype(float)
        if normalize == "standard":
            mu, sd = float(r.mean()), float(r.std()) or 1.0
            r, s = (r - mu) / sd, (s - mu) / sd
        elif normalize == "minmax":
            lo, hi = float(r.min()), float(r.max())
            rng = (hi - lo) or 1.0
            r, s = (r - lo) / rng, (s - lo) / rng
        dists.append(wasserstein_distance(r, s))
    return float(np.mean(dists))


def mia_auc_arrays(X_train, X_test, X_syn, mode="mia_proximity", seed=0):
    """Membership-inference ROC-AUC on encoded matrices.

    mode="mia_proximity" (default, privacy-faithful): members = train records,
      non-members = test records; the attacker scores each real record by
      closeness to the synthetic set (score = -distance to nearest synthetic).
      AUC of that score predicting membership. ~0.5 = good privacy, ->1.0 =
      the training set is distinguishable via the synthetic data (memorisation).
      Train/test are balanced by subsampling the larger to the smaller size.

    mode="detection": the literal "classifier that distinguishes real-train from
      synthetic" (a C2ST-style detector), reported as cross-val ROC-AUC. NOTE:
      its privacy interpretation is the *opposite* (0.5 = real~synthetic), so it
      measures distinguishability/fidelity, not membership leakage.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    if mode == "mia_proximity":
        rng = np.random.default_rng(seed)
        n = min(len(X_train), len(X_test))
        Xtr = X_train[rng.choice(len(X_train), n, replace=False)]
        Xte = X_test[rng.choice(len(X_test), n, replace=False)]
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_syn)
        d_mem = nn.kneighbors(Xtr)[0][:, 0]
        d_non = nn.kneighbors(Xte)[0][:, 0]
        y = np.r_[np.ones(n), np.zeros(n)]
        score = -np.r_[d_mem, d_non]  # closer to synthetic -> predict member
        return float(roc_auc_score(y, score))

    if mode == "detection":
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.vstack([X_train, X_syn])
        y = np.r_[np.ones(len(X_train)), np.zeros(len(X_syn))]
        clf = HistGradientBoostingClassifier(random_state=seed)
        proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba",
                                  n_jobs=-1)[:, 1]
        return float(roc_auc_score(y, proba))

    raise ValueError(f"unknown mode: {mode}")


# ── public, dataset-generic entry points ─────────────────────────────────────────

def compute_dcr_tabddpm_mean(dataset_name, real_data_path, gen_dir,
                             normalize="minmax", aggregation="mean",
                             include_label=False):
    """TabDDPM DCR: nearest-neighbour L2 distance, synthetic -> real TRAIN only.
    Numeric min-max normalised, categoricals one-hot/√2 (generic; works with 0 or
    many categorical columns).

    ``aggregation``   : "mean" (default, TabDDPM Table-9 style) or "median".
    ``include_label`` : if True, fold the target into the distance (regression ->
                        numeric col, classification -> categorical col).

    NOTE: the pipeline BASELINE (exp/<ds>/ddpm_cb_best/privacy.json, written by
    resample_privacy.privacy_metrics) uses aggregation="median" AND
    include_label=True. To compare an ablation run against that baseline on equal
    footing use compute_dcr_baseline() (or pass those two args), NOT the defaults —
    the mean/median gap is large on datasets with many one-hot categorical dims."""
    rn, rc, ry = _read(real_data_path, "train")
    sn, sc, sy = _read(gen_dir, "train")
    if include_label:
        task_type = _load_info(real_data_path)["task_type"]
        rn, rc = _append_label(rn, rc, ry, task_type)
        sn, sc = _append_label(sn, sc, sy, task_type)
    X_real, X_syn = _encode(rn, rc, (rn, rc), (sn, sc), normalize=normalize)
    return _dcr_from_encoded(X_real, X_syn, aggregation=aggregation)


def compute_dcr_baseline(dataset_name, real_data_path, gen_dir):
    """DCR computed with the SAME recipe as the pipeline baseline
    (exp/<ds>/ddpm_cb_best/privacy.json, from resample_privacy.privacy_metrics):
    MEDIAN nearest-neighbour L2, target label INCLUDED, minmax num + onehot/√2 cat.
    This is the number to compare ablation runs against privacy.json directly."""
    return compute_dcr_tabddpm_mean(dataset_name, real_data_path, gen_dir,
                                    aggregation="median", include_label=True)


def compute_wasserstein_mean(dataset_name, real_data_path, gen_dir,
                             normalize="standard"):
    """Mean 1D Wasserstein distance over numerical columns (TabDDPM Table 9).
    Categorical columns are ignored by design."""
    rn, _, _ = _read(real_data_path, "train")
    sn, _, _ = _read(gen_dir, "train")
    return wasserstein_mean_arrays(rn, sn, normalize=normalize)


def compute_membership_inference_auc(dataset_name, real_data_path, gen_dir,
                                     normalize="minmax", mode="mia_proximity",
                                     seed=0):
    """Black-box membership-inference ROC-AUC (TabDDPM Sec. 5.3).

    ASSUMPTIONS (paper under-specifies the exact attack):
      * We use a distance-based attacker (nearest-synthetic proximity) with the
        TRAIN split as members and the TEST split as non-members. This is the
        standard reproducible construction and gives the ~0.5-good / ->1.0-bad
        semantics the protocol expects.
      * All features are put in a shared encoded space (numeric min-max, cats
        one-hot). Label is excluded.
      * mode="detection" offers the literal train-vs-synthetic classifier
        instead; see mia_auc_arrays for why its privacy reading differs.
    """
    rn, rc, _ = _read(real_data_path, "train")
    tn, tc, _ = _read(real_data_path, "test")
    sn, sc, _ = _read(gen_dir, "train")
    X_tr, X_te, X_sy = _encode(rn, rc, (rn, rc), (tn, tc), (sn, sc),
                               normalize=normalize)
    return mia_auc_arrays(X_tr, X_te, X_sy, mode=mode, seed=seed)


def compute_ml_efficiency_catboost(dataset_name, real_data_path, gen_dirs,
                                   n_clf_seeds=10, normalization=None,
                                   params=None, include_real=True):
    """ML efficiency under TabDDPM's protocol, generic over task type.

    - Detects regression vs classification from info.json.
    - For each generated dir in ``gen_dirs`` (typically the 5 gen_seed_* dirs),
      trains CatBoost on the synthetic data with ``n_clf_seeds`` random seeds and
      evaluates on the REAL test split -> 5x10 = 50 runs.
    - Reports mean +/- std of F1 (classification) or R2 (regression).
    - ``include_real`` adds a Real reference: CatBoost trained on real train,
      evaluated on real test, ``n_clf_seeds`` seeds.

    Hyperparameters: ``params=None`` uses this repo's tuned CatBoost config
    (tuned_models/catboost/<ds>_cv.json), matching the rest of the pipeline and
    TabDDPM's own tuned setup. Pass an explicit ``params`` dict to force defaults.
    NOTE: relies on the pipeline's relative path to the tuned config, so run with
    the repository root as the current working directory.

    F1 AVERAGING ASSUMPTION: we report sklearn's macro-average F1 (the
    'macro avg'/'f1-score' entry of classification_report). If the paper used
    weighted or binary F1 the absolute value shifts slightly.
    """
    import lib
    from eval_catboost import train_catboost

    info = _load_info(real_data_path)
    regression = info["task_type"] == "regression"

    def _T(seed):
        return dict(seed=seed, normalization=normalization, num_nan_policy=None,
                    cat_nan_policy=None, cat_min_frequency=None, cat_encoding=None,
                    y_policy="default")

    def _metric(report):
        m = report["metrics"]["test"]
        if regression:
            return m.get("r2")
        return m.get("macro avg", {}).get("f1-score")

    synth_vals = []
    for gd in gen_dirs:
        for s in range(n_clf_seeds):
            train_catboost(parent_dir=gd, real_data_path=real_data_path,
                           eval_type="synthetic", T_dict=_T(s), seed=s,
                           params=params, change_val=False)
            rep = lib.load_json(os.path.join(gd, "results_catboost.json"))
            synth_vals.append(_metric(rep))

    real_vals = []
    if include_real:
        tmp = tempfile.mkdtemp(prefix="mleff_real_")
        for s in range(n_clf_seeds):
            train_catboost(parent_dir=tmp, real_data_path=real_data_path,
                           eval_type="real", T_dict=_T(s), seed=s,
                           params=params, change_val=False)
            rep = lib.load_json(os.path.join(tmp, "results_catboost.json"))
            real_vals.append(_metric(rep))

    metric_name = "r2" if regression else "macro_f1"

    def _agg(vals):
        arr = np.array([v for v in vals if v is not None], dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}

    out = {"metric": metric_name, "synthetic": _agg(synth_vals)}
    if include_real:
        out["real"] = _agg(real_vals)
    return out


# ── run-all + comparison table ───────────────────────────────────────────────────

def _find_gen_dirs(gen_dir):
    if os.path.exists(os.path.join(gen_dir, "X_num_train.npy")):
        return [gen_dir]
    subs = sorted(
        os.path.join(gen_dir, d) for d in os.listdir(gen_dir)
        if d.startswith("gen_seed_")
        and os.path.exists(os.path.join(gen_dir, d, "X_num_train.npy"))
    )
    if not subs:
        raise FileNotFoundError(f"No generated arrays under {gen_dir}")
    return subs


def run_tabddpm_protocol(dataset_name, real_data_path, gen_dir,
                         n_clf_seeds=10, normalization=None, run_mia=True):
    """Run all four TabDDPM-protocol metrics and return a results dict.

    DCR / Wasserstein / MIA are averaged across the generated dirs; ML efficiency
    already aggregates over them internally.
    """
    gen_dirs = _find_gen_dirs(gen_dir)

    ml = compute_ml_efficiency_catboost(
        dataset_name, real_data_path, gen_dirs,
        n_clf_seeds=n_clf_seeds, normalization=normalization,
    )

    dcr, wass, mia = [], [], []
    for gd in gen_dirs:
        dcr.append(compute_dcr_tabddpm_mean(dataset_name, real_data_path, gd))
        w = compute_wasserstein_mean(dataset_name, real_data_path, gd)
        if w is not None:
            wass.append(w)
        if run_mia:
            mia.append(compute_membership_inference_auc(
                dataset_name, real_data_path, gd))

    def _agg(vals):
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    return {
        "dataset": dataset_name,
        "ml_efficiency": ml,
        "dcr_tabddpm_mean": _agg(dcr),
        "wasserstein_mean": _agg(wass),
        "mia_auc": _agg(mia),
    }


def print_tabddpm_comparison(dataset_name, results):
    """Print our TabDDPM-protocol numbers, auto-looking-up the paper reference by
    dataset abbreviation. Datasets with no filled-in reference are labelled
    explicitly rather than omitted."""
    abbr = NAME_TO_ABBREV.get(dataset_name)
    ref = TABDDPM_PAPER_REFERENCE.get(abbr, {}) if abbr else {}

    print(f"\n{'=' * 82}\n  TabDDPM protocol — {dataset_name} "
          f"({abbr or 'no abbrev'})\n{'=' * 82}")

    ml = results.get("ml_efficiency", {})
    if ml:
        syn, real = ml.get("synthetic", {}), ml.get("real", {})
        print(f"  ML efficiency ({ml.get('metric')}):")
        print(f"    synthetic : {syn.get('mean'):.4f} +/- {syn.get('std'):.4f} "
              f"(n={syn.get('n')})")
        if real:
            print(f"    real ref  : {real.get('mean'):.4f} +/- {real.get('std'):.4f} "
                  f"(n={real.get('n')})")

    def _line(label, entry):
        if entry is None:
            print(f"  {label:22s} : n/a")
        else:
            print(f"  {label:22s} : {entry['mean']:.4f} +/- {entry['std']:.4f}")

    _line("DCR (mean L2)", results.get("dcr_tabddpm_mean"))
    _line("Wasserstein (num mean)", results.get("wasserstein_mean"))
    _line("Membership-inf AUC", results.get("mia_auc"))

    print("\n  Published TabDDPM reference:")
    if not ref:
        print(f"    no published reference available for {dataset_name} "
              f"({abbr}) — populate TABDDPM_PAPER_REFERENCE['{abbr}'].")
    else:
        for k, v in ref.items():
            if not k.endswith("_std"):
                std = ref.get(f"{k}_std")
                print(f"    {k:22s} : {v}" + (f" +/- {std}" if std is not None else ""))


# ── CLI ──────────────────────────────────────────────────────────────────────────

def _main():
    import argparse
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.dirname(here)):
        if p not in sys.path:
            sys.path.insert(0, p)

    p = argparse.ArgumentParser(description="TabDDPM-protocol metrics (generic).")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--real_data_path", required=True)
    p.add_argument("--gen_dir", required=True,
                   help="a gen_seed dir or a run dir containing gen_seed_*")
    p.add_argument("--n_clf_seeds", type=int, default=10)
    p.add_argument("--normalization", default=None)
    p.add_argument("--no_mia", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = run_tabddpm_protocol(
        args.dataset_name, args.real_data_path, args.gen_dir,
        n_clf_seeds=args.n_clf_seeds, normalization=args.normalization,
        run_mia=not args.no_mia,
    )
    print_tabddpm_comparison(args.dataset_name, res)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    _main()
