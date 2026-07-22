"""TabDiff's paper metric suite beyond ML efficiency.

Faithful ports of the evaluators behind the TabDiff paper's fidelity /
privacy tables (MinkaiXu/TabDiff):

  * density  -> ``tabdiff/metrics.py::evaluate_density``
               Shape / Trend via ``sdmetrics`` QualityReport — the same
               library call the original makes, not a reimplementation.
  * c2st     -> ``tabdiff/metrics.py::evaluate_c2st``
               ``sdmetrics`` LogisticDetection.
  * dcr      -> ``tabdiff/metrics.py::evaluate_dcr``
               Fraction of synthetic rows whose L1-nearest real neighbour
               is in TRAIN rather than TEST (~0.5 is ideal). Numerical
               columns are scaled by the train range, categoricals one-hot
               encoded with the encoder fitted on train+test.
  * quality  -> ``eval/eval_quality.py`` (alpha-precision / beta-recall)
               TabDiff calls synthcity's ``AlphaPrecision`` and keeps only
               the *naive* keys. The naive path is pure numpy/sklearn, so
               it is ported here verbatim (from
               ``synthcity.metrics.eval_statistical.AlphaPrecision``)
               instead of pulling in the heavy synthcity dependency.

Column layout replicates TabDiff's ``reorder``: numerical block first, then
categorical, with the target appended to the numerical block for regression
and to the categorical block for classification. Reproduced synthcity quirk:
``GenericDataLoader`` defaults ``target_column`` to the LAST column, which
``_normalize_covariates`` then drops — so the last one-hot column is excluded
from the alpha/beta computation, exactly as in the original pipeline.

Deliberate deviations (numerics preserved, robustness added):
  * DCR distances run in float32 (original: float64) with an OOM fallback
    that halves the batch; distance *comparisons* are unaffected.
  * Zero-range numerical columns are scaled by 1 instead of 0 (the original
    divides by zero and NaNs out).
"""

import json
import os

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def _read_split(path: str, split: str):
    """Load one split's raw (data-space) arrays: (X_num, X_cat, y)."""
    def _maybe(name):
        p = os.path.join(path, f"{name}_{split}.npy")
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    return _maybe("X_num"), _maybe("X_cat"), np.load(
        os.path.join(path, f"y_{split}.npy"), allow_pickle=True)


def _blocks(X_num, X_cat, y, task_type):
    """TabDiff's `reorder` layout: (num_block float, cat_block str-or-None).

    Regression appends the target to the numerical block, classification to
    the categorical block.
    """
    num_parts = [] if X_num is None else [np.asarray(X_num, dtype=np.float64)]
    cat_parts = [] if X_cat is None else [np.asarray(X_cat).astype(str)]

    if task_type == "regression":
        num_parts.append(np.asarray(y, dtype=np.float64).reshape(-1, 1))
    else:
        cat_parts.append(np.asarray(y).astype(str).reshape(-1, 1))

    num_block = np.column_stack(num_parts) if num_parts else None
    cat_block = np.column_stack(cat_parts) if cat_parts else None
    return num_block, cat_block


def _to_frame_and_metadata(num_block, cat_block):
    """DataFrame with integer columns (num first, then cat) + sdmetrics
    metadata, mirroring what TabDiff's `reorder` hands to sdmetrics."""
    import pandas as pd

    parts, sdtypes = [], []
    if num_block is not None:
        parts.append(pd.DataFrame(num_block))
        sdtypes += ["numerical"] * num_block.shape[1]
    if cat_block is not None:
        parts.append(pd.DataFrame(cat_block))
        sdtypes += ["categorical"] * cat_block.shape[1]

    df = pd.concat(parts, axis=1)
    # TabDiff uses integer column labels; current sdmetrics (LogisticDetection)
    # requires strings. Labels only — metric values are unaffected.
    df.columns = [str(i) for i in range(len(df.columns))]
    metadata = {"columns": {str(i): {"sdtype": s}
                            for i, s in enumerate(sdtypes)}}
    return df, metadata


# ---------------------------------------------------------------------------
# density (Shape / Trend)  +  c2st  — sdmetrics, as in the original
# ---------------------------------------------------------------------------

def compute_density(real_blocks, syn_blocks) -> dict:
    from sdmetrics.reports.single_table import QualityReport

    real_df, metadata = _to_frame_and_metadata(*real_blocks)
    syn_df, _ = _to_frame_and_metadata(*syn_blocks)

    report = QualityReport()
    report.generate(real_df, syn_df, metadata, verbose=False)
    props = report.get_properties()
    shape = float(props["Score"][0])
    trend = float(props["Score"][1])
    return {"shape": shape, "trend": trend,
            "density_overall": (shape + trend) / 2}


def compute_c2st(real_blocks, syn_blocks) -> dict:
    from sdmetrics.single_table import LogisticDetection

    real_df, metadata = _to_frame_and_metadata(*real_blocks)
    syn_df, _ = _to_frame_and_metadata(*syn_blocks)

    score = LogisticDetection.compute(
        real_data=real_df, synthetic_data=syn_df, metadata=metadata)
    return {"c2st": float(score)}


# ---------------------------------------------------------------------------
# dcr — port of tabdiff/metrics.py::evaluate_dcr
# ---------------------------------------------------------------------------

def _encode_dcr(num_block, cat_block, num_ranges, encoder):
    parts = []
    if num_block is not None:
        parts.append(num_block / num_ranges)
    if cat_block is not None:
        parts.append(encoder.transform(cat_block).toarray())
    return np.concatenate(parts, axis=1).astype(np.float32)


def compute_dcr_tabdiff(train_blocks, test_blocks, syn_blocks,
                        device="cpu") -> dict:
    num_train, cat_train = train_blocks
    num_test, cat_test = test_blocks
    num_syn, cat_syn = syn_blocks

    if num_train is not None:
        num_ranges = num_train.max(axis=0) - num_train.min(axis=0)
        num_ranges[num_ranges == 0] = 1.0  # original divides by zero here
    else:
        num_ranges = None

    n_onehot = 0
    encoder = None
    if cat_train is not None:
        encoder = OneHotEncoder()
        encoder.fit(np.concatenate([cat_train, cat_test], axis=0))
        n_onehot = sum(len(c) for c in encoder.categories_)

    train_np = _encode_dcr(num_train, cat_train, num_ranges, encoder)
    test_np = _encode_dcr(num_test, cat_test, num_ranges, encoder)
    syn_np = _encode_dcr(num_syn, cat_syn, num_ranges, encoder)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    train_th = torch.tensor(train_np, device=dev)
    test_th = torch.tensor(test_np, device=dev)
    syn_th = torch.tensor(syn_np, device=dev)

    # Original heuristic ("fit into 10GB"); the OOM loop below is the safety
    # net for smaller GPUs.
    batch_size = max(1, 10000 // max(1, n_onehot))

    closer_to_train = 0
    i = 0
    while i < syn_th.shape[0]:
        batch = syn_th[i:i + batch_size]
        try:
            dcr_train = (batch[:, None] - train_th).abs().sum(dim=2).min(dim=1).values
            dcr_test = (batch[:, None] - test_th).abs().sum(dim=2).min(dim=1).values
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            continue
        closer_to_train += int((dcr_train < dcr_test).sum().item())
        i += batch.shape[0]

    return {"dcr_tabdiff": closer_to_train / syn_th.shape[0]}


# ---------------------------------------------------------------------------
# quality — alpha-precision / beta-recall (naive), port of synthcity's
# AlphaPrecision.metrics + TabDiff's eval/eval_quality.py driver
# ---------------------------------------------------------------------------

def _alpha_precision_metrics(X, X_syn):
    """Verbatim port of synthcity AlphaPrecision.metrics (emb_center=None)."""
    if len(X) != len(X_syn):
        raise RuntimeError("The real and synthetic data must have the same length")

    emb_center = np.mean(X, axis=0)

    n_steps = 30
    alphas = np.linspace(0, 1, n_steps)

    Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)

    synth_center = np.mean(X_syn, axis=0)

    alpha_precision_curve = []
    beta_coverage_curve = []

    synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
    real_to_real, _ = nbrs_real.kneighbors(X)

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

    real_to_real = real_to_real[:, 1].squeeze()
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()

    real_synth_closest = X_syn[real_to_synth_args]

    real_synth_closest_d = np.sqrt(
        np.sum((real_synth_closest - synth_center) ** 2, axis=1))
    closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

    for k in range(len(Radii)):
        precision_audit_mask = synth_to_center <= Radii[k]
        alpha_precision = np.mean(precision_audit_mask)

        beta_coverage = np.mean(
            (real_to_synth <= real_to_real)
            * (real_synth_closest_d <= closest_synth_Radii[k]))

        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)

    Delta_precision_alpha = 1 - np.sum(
        np.abs(np.array(alphas) - np.array(alpha_precision_curve))
    ) / np.sum(alphas)
    Delta_coverage_beta = 1 - np.sum(
        np.abs(np.array(alphas) - np.array(beta_coverage_curve))
    ) / np.sum(alphas)

    return Delta_precision_alpha, Delta_coverage_beta


def compute_alpha_beta(train_blocks, syn_blocks) -> dict:
    num_real, cat_real = train_blocks
    num_syn, cat_syn = syn_blocks

    parts_real, parts_syn = [], []
    if num_real is not None:
        parts_real.append(num_real)
        parts_syn.append(num_syn)
    if cat_real is not None:
        # eval_quality.py fits the encoder on the REAL data only (unlike DCR).
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(cat_real)
        parts_real.append(encoder.transform(cat_real).toarray())
        parts_syn.append(encoder.transform(cat_syn).toarray())

    X_real = np.concatenate(parts_real, axis=1).astype(float)
    X_syn = np.concatenate(parts_syn, axis=1).astype(float)

    # synthcity's GenericDataLoader defaults target_column to the last
    # column, which _normalize_covariates drops before scaling.
    X_real = X_real[:, :-1]
    X_syn = X_syn[:, :-1]

    scaler = MinMaxScaler().fit(X_real)
    alpha, beta = _alpha_precision_metrics(
        scaler.transform(X_real), scaler.transform(X_syn))
    return {"alpha_precision": float(alpha), "beta_recall": float(beta)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_tabdiff_fidelity(
    synthetic_dir: str,
    real_data_path: str,
    device: str = "cpu",
    metrics=("density", "c2st", "dcr", "quality"),
) -> dict:
    """Compute the TabDiff-paper fidelity metrics for one synthetic sample.

    ``synthetic_dir`` must contain X_num_train / X_cat_train / y_train npy
    files in raw data space (as written by sample.py). Real train and test
    come from ``real_data_path``. Metrics that fail (e.g. sdmetrics not
    installed) are skipped with a warning so the rest still land.
    """
    with open(os.path.join(real_data_path, "info.json")) as fh:
        task_type = json.load(fh)["task_type"]

    train_blocks = _blocks(*_read_split(real_data_path, "train"), task_type)
    test_blocks = _blocks(*_read_split(real_data_path, "test"), task_type)
    syn_blocks = _blocks(*_read_split(synthetic_dir, "train"), task_type)

    out: dict = {}
    runners = {
        "density": lambda: compute_density(train_blocks, syn_blocks),
        "c2st":    lambda: compute_c2st(train_blocks, syn_blocks),
        "dcr":     lambda: compute_dcr_tabdiff(train_blocks, test_blocks,
                                               syn_blocks, device=device),
        "quality": lambda: compute_alpha_beta(train_blocks, syn_blocks),
    }
    for name in metrics:
        try:
            out.update(runners[name]())
        except Exception as exc:
            print(f"    [WARN] TabDiff fidelity metric '{name}' failed: {exc}")
    return out
