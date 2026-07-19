"""TabDiff's exact XGBoost ML-efficiency evaluator.

Faithful reimplementation of ``eval/mle/mle.py`` from MinkaiXu/TabDiff
(the evaluator behind the paper's MLE numbers), exposed through the same
interface as ``scripts/eval_catboost.train_catboost`` so ``ablation_runner``
can call either one interchangeably.

Protocol (per synthetic sample):
  1. TabDiff's ``feat_transform``: numerical columns are min-max scaled to
     [0, 5] (or log-transformed when ``cmin >= 0 and cmax >= 1e3``);
     categorical columns are one-hot encoded per column
     (``handle_unknown='ignore'``); the target is label-encoded for
     classification. All encoders/statistics are fitted on the TRAIN
     (synthetic) split and reused for the real val/test splits.
  2. Fit every combination of the fixed 36-point XGBoost grid on train.
  3. Score each combination on the real VALIDATION split.
  4. Independently per metric, pick the best combination, refit it on
     train, and report its score on the real TEST split.

Two TabDiff quirks are reproduced on purpose so numbers stay directly
comparable to the paper:
  * ``cmin``/``cmax`` are shared across numerical columns — every numerical
    column is scaled with the min/max of the first numerical column
    (``if not cmin: cmin = col.min()`` in the original).
  * For regression, the train and test targets are ``log(clip(y, 1, 20000))``
    transformed but the validation target is left raw, exactly as in the
    original code. Reported RMSE is therefore on the log scale.

Output: ``results_xgboost.json`` in ``parent_dir``, with the same
``metrics['test']`` layout ``ablation_runner`` already flattens — the
existing keys ('accuracy', 'roc_auc', 'macro avg/f1-score', 'r2', 'rmse')
plus TabDiff-style scalars (binary_f1, weighted_f1, macro_f1, mae,
explained_variance).
"""

import os

import numpy as np
import pandas as pd
import zero
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

import lib
from lib import read_changed_val, read_pure_data

# ---------------------------------------------------------------------------
# TabDiff's fixed hyper-parameter grid (36 combinations)
# ---------------------------------------------------------------------------

_TABDIFF_GRID = {
    "n_estimators":     [10, 50, 100],
    "min_child_weight": [1, 10],
    "max_depth":        [5, 10, 20],
    "gamma":            [0.0, 1.0],
    "nthread":          [-1],
}


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _xgb_param_grid(task_type: str, device=None) -> dict:
    """TabDiff's exact grid, with tree_method/objective adapted to the
    installed XGBoost version (2.x removed 'gpu_hist' and renamed
    'reg:linear' -> 'reg:squarederror'; both replacements are exact)."""
    import xgboost

    grid = dict(_TABDIFF_GRID)
    major = int(xgboost.__version__.split(".")[0])
    use_gpu = _has_cuda() if device is None else str(device).startswith("cuda")

    if major >= 2:
        grid["tree_method"] = ["hist"]
        if use_gpu:
            grid["device"] = ["cuda"]
        grid["objective"] = (["reg:squarederror"] if task_type == "regression"
                             else ["binary:logistic"])
    else:
        grid["tree_method"] = ["gpu_hist"] if use_gpu else ["hist"]
        grid["objective"] = (["reg:linear"] if task_type == "regression"
                             else ["binary:logistic"])
    return grid


# ---------------------------------------------------------------------------
# TabDiff feat_transform (ported from mle.py to X_num/X_cat/y arrays)
# ---------------------------------------------------------------------------

def _make_ohe():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def _feat_transform(X_num, X_cat, y, task_type,
                    label_encoder=None, encoders=None, cmax=None, cmin=None):
    """Port of TabDiff's feat_transform. Numerical columns first, then
    categorical, matching this repo's data layout. The shared cmin/cmax and
    the ``if not cmin`` re-trigger on 0-valued minima are reproduced
    verbatim from the original."""
    if encoders is None:
        encoders = dict()

    if task_type != "regression":
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(y)
        labels = label_encoder.transform(y)
    else:
        labels = np.asarray(y, dtype=np.float32)

    features = []

    if X_num is not None:
        for i in range(X_num.shape[1]):
            col = X_num[:, i].astype(np.float32)
            if not cmin:
                cmin = col.min()
            if not cmax:
                cmax = col.max()
            if cmin >= 0 and cmax >= 1e3:
                feature = np.log(np.maximum(col, 1e-2))
            else:
                feature = (col - cmin) / (cmax - cmin) * 5
            features.append(feature)

    if X_cat is not None:
        for i in range(X_cat.shape[1]):
            col = X_cat[:, i].astype(str).reshape(-1, 1)
            encoder = encoders.get(i)
            if encoder is not None:
                feature = encoder.transform(col)
            else:
                encoder = _make_ohe()
                encoders[i] = encoder
                feature = encoder.fit_transform(col)
            features.append(feature)

    features = np.column_stack(features)
    return features, labels, label_encoder, encoders, cmax, cmin


# ---------------------------------------------------------------------------
# TabDiff metric helpers (ported verbatim)
# ---------------------------------------------------------------------------

def _weighted_f1(y_true, pred):
    report = classification_report(y_true, pred, output_dict=True)
    classes = list(report.keys())[:-3]
    proportion = [report[i]["support"] / len(y_true) for i in classes]
    weighted_f1 = np.sum(
        list(map(lambda i, prop: report[i]["f1-score"] * (1 - prop) / (len(classes) - 1),
                 classes, proportion)))
    return weighted_f1


def _roc_auc_padded(y_true, pred_prob, unique_labels, size):
    """TabDiff's AUROC with zero-padding for labels absent from train."""
    rest_label = set(range(size)) - set(int(u) for u in unique_labels)
    tmp = []
    j = 0
    for i in range(size):
        if i in rest_label:
            tmp.append(np.array([0] * y_true.shape[0])[:, np.newaxis])
        else:
            try:
                tmp.append(pred_prob[:, [j]])
            except Exception:
                tmp.append(pred_prob[:, np.newaxis])
            j += 1
    onehot = np.eye(size)[np.asarray(y_true, dtype=int)]
    if size == 2:
        return roc_auc_score(onehot, np.hstack(tmp))
    return roc_auc_score(onehot, np.hstack(tmp), multi_class="ovr")


# ---------------------------------------------------------------------------
# Grid search: fit all combos, select best per metric on val, refit, test
# ---------------------------------------------------------------------------

def _classification_scores(y_true, pred, pred_prob, unique_labels, n_classes,
                           binary: bool) -> dict:
    scores = {
        "weighted_f1": _weighted_f1(y_true, pred),
        "roc_auc":     _roc_auc_padded(y_true, pred_prob, unique_labels, n_classes),
        "accuracy":    accuracy_score(y_true, pred),
        "macro_f1":    f1_score(y_true, pred, average="macro"),
    }
    if binary:
        scores["binary_f1"] = f1_score(y_true, pred, average="binary")
        scores["precision"] = precision_score(y_true, pred, average="binary")
        scores["recall"]    = recall_score(y_true, pred, average="binary")
    return scores


def _evaluate_classification(x_train, y_train, x_val, y_val, x_test, y_test,
                             n_classes, param_set, binary):
    unique_labels = np.unique(y_train)

    def _predict(model, x, y_ref):
        # TabDiff's degenerate path when train lacks some classes.
        degenerate = (len(unique_labels) == 1 if binary
                      else len(unique_labels) != len(np.unique(y_ref)))
        if degenerate:
            pred = np.array([unique_labels[0]] * len(x))
            pred_prob = np.array([1.0] * len(x))
        else:
            pred = model.predict(x)
            pred_prob = model.predict_proba(x)
            if binary:
                pred_prob = pred_prob  # (n, 2); padding helper handles layout
        return pred, pred_prob

    results = []
    for param in param_set:
        model = XGBClassifier(**param)
        try:
            model.fit(x_train, y_train)
        except ValueError:
            pass
        pred, pred_prob = _predict(model, x_val, y_val)
        row = _classification_scores(y_val, pred, pred_prob, unique_labels,
                                     n_classes, binary)
        row["param"] = param
        results.append(row)

    df = pd.DataFrame(results)
    primary = "binary_f1" if binary else "macro_f1"
    df["avg"] = df[[primary, "weighted_f1", "roc_auc"]].mean(axis=1)

    # Best-per-metric selection on validation, exactly as TabDiff does.
    # macro_f1 selection is an addition for binclass (TabDiff computes but
    # does not select on it) so the summary's macro-F1 column stays a true
    # best-per-metric number.
    select_metrics = ["weighted_f1", "roc_auc", "accuracy", "macro_f1"]
    if binary:
        select_metrics.append("binary_f1")

    def _test_scores(param):
        model = XGBClassifier(**param)
        try:
            model.fit(x_train, y_train)
        except ValueError:
            pass
        pred, pred_prob = _predict(model, x_test, y_test)
        return _classification_scores(y_test, pred, pred_prob, unique_labels,
                                      n_classes, binary)

    val_best  = {m: float(df[m].max()) for m in select_metrics}
    test_best = {m: float(_test_scores(df.param[df[m].idxmax()])[m])
                 for m in select_metrics}
    return val_best, test_best


def _evaluate_regression(x_train, y_train, x_val, y_val, x_test, y_test,
                         param_set):
    # TabDiff transforms train/test targets but NOT the validation target —
    # reproduced verbatim for comparability with the published numbers.
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test  = np.log(np.clip(y_test, 1, 20000))

    def _scores(y_true, pred):
        return {
            "r2":                 r2_score(y_true, pred),
            "explained_variance": explained_variance_score(y_true, pred),
            "mae":                mean_absolute_error(y_true, pred),
            "rmse":               mean_squared_error(y_true, pred) ** 0.5,
        }

    results = []
    for param in param_set:
        model = XGBRegressor(**param)
        model.fit(x_train, y_train)
        row = _scores(y_val, model.predict(x_val))
        row["param"] = param
        results.append(row)

    df = pd.DataFrame(results)

    higher_better = {"r2": True, "explained_variance": True,
                     "mae": False, "rmse": False}

    def _test_scores(param):
        model = XGBRegressor(**param)
        model.fit(x_train, y_train)
        return _scores(y_test, model.predict(x_test))

    val_best, test_best = {}, {}
    for m, hb in higher_better.items():
        idx = df[m].idxmax() if hb else df[m].idxmin()
        val_best[m]  = float(df[m][idx])
        test_best[m] = float(_test_scores(df.param[idx])[m])
    return val_best, test_best


# ---------------------------------------------------------------------------
# Report layout compatible with eval_catboost / ablation_runner
# ---------------------------------------------------------------------------

def _to_report_metrics(scores: dict, task_type: str) -> dict:
    if task_type == "regression":
        out = dict(scores)
        out["score"] = -scores["rmse"]
        return out
    out = {k: v for k, v in scores.items()}
    out["macro avg"] = {"f1-score": scores["macro_f1"]}
    out["score"] = scores["accuracy"]
    return out


# ---------------------------------------------------------------------------
# Entry point — mirrors eval_catboost.train_catboost
# ---------------------------------------------------------------------------

def train_xgboost(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed=0,
    params=None,
    change_val=True,
    device=None,
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = lib.load_json(os.path.join(real_data_path, "info.json"))
    task_type = info["task_type"]

    if change_val:
        (X_num_real, X_cat_real, y_real,
         X_num_val, X_cat_val, y_val) = read_changed_val(real_data_path, val_size=0.2)

    print("-" * 100)
    if eval_type == "merged":
        print("loading merged data...")
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)

        y = np.concatenate([y_real, y_fake], axis=0)
        X_num = (np.concatenate([X_num_real, X_num_fake], axis=0)
                 if X_num_real is not None else None)
        X_cat = (np.concatenate([X_cat_real, X_cat_fake], axis=0)
                 if X_cat_real is not None else None)
    elif eval_type == "synthetic":
        print(f"loading synthetic data: {parent_dir}")
        X_num, X_cat, y = read_pure_data(synthetic_data_path)
    elif eval_type == "real":
        print("loading real data...")
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path)
        else:
            X_num, X_cat, y = X_num_real, X_cat_real, y_real
    else:
        raise ValueError("Choose eval method")

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, "val")
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, "test")

    # TabDiff feature transform: fit on train, reuse on val/test.
    x_train, y_train, label_enc, encoders, cmax, cmin = _feat_transform(
        X_num, X_cat, y, task_type)
    x_val, y_val_t, _, _, _, _ = _feat_transform(
        X_num_val, X_cat_val, y_val, task_type, label_enc, encoders, cmax, cmin)
    x_test, y_test_t, _, _, _, _ = _feat_transform(
        X_num_test, X_cat_test, y_test, task_type, label_enc, encoders, cmax, cmin)

    grid = params if params is not None else _xgb_param_grid(task_type, device)
    param_set = list(ParameterGrid(grid))
    print(f"Train size: {x_train.shape}, Val size {x_val.shape}")
    print(f"TabDiff XGBoost grid: {len(param_set)} combinations")
    print("-" * 100)

    if task_type == "regression":
        val_best, test_best = _evaluate_regression(
            x_train, y_train, x_val, y_val_t, x_test, y_test_t, param_set)
    else:
        n_classes = info.get("n_classes") or len(np.unique(y_train))
        binary = task_type == "binclass"
        val_best, test_best = _evaluate_classification(
            x_train, y_train, x_val, y_val_t, x_test, y_test_t,
            n_classes, param_set, binary)

    report = {
        "eval_type": eval_type,
        "dataset":   real_data_path,
        "evaluator": "xgboost_tabdiff",
        "grid_size": len(param_set),
        "metrics": {
            "val":  _to_report_metrics(val_best, task_type),
            "test": _to_report_metrics(test_best, task_type),
        },
    }

    metrics_report = lib.MetricsReport(report["metrics"], lib.TaskType(task_type))
    metrics_report.print_metrics()

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_xgboost.json"))

    return metrics_report
