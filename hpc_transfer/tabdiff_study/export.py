"""
Our sampler's output -> one CSV in exactly TabDiff's raw table format.

scripts/sample.py writes, per generation seed, X_num_train.npy (numerical
columns, already inverse-transformed to raw units; regression target split
off into y), X_cat_train.npy (the categorical tokens, inverse of the ordinal
encoding) and y_train.npy (class index or regression value). This module puts
them back into TabDiff's column order with TabDiff's column names, maps class
indices back to the original label tokens, and writes the CSV the same way
TabDiff's real.csv was written (pandas to_csv, index=False), so that
pd.read_csv parses both tables into the same columns, dtypes and category
labels. check_against_real() asserts that.

No value is changed: numerical columns are exported as sampled (sample.py's
own rounding of discrete columns included, nothing added), categorical tokens
are the strings the model was trained on, which prepare_data.py read verbatim
from TabDiff's real.csv.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def load_info(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "info.json")) as fh:
        return json.load(fh)


def to_frame(gen_dir: str, data_dir: str) -> pd.DataFrame:
    info = load_info(data_dir)
    cols = info["column_names"]
    num_idx, cat_idx = info["num_col_idx"], info["cat_col_idx"]
    tgt_idx = info["target_col_idx"][0]

    def maybe(name):
        p = os.path.join(gen_dir, name)
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    X_num, X_cat, y = maybe("X_num_train.npy"), maybe("X_cat_train.npy"), maybe("y_train.npy")
    if y is None:
        raise FileNotFoundError(f"{gen_dir}/y_train.npy missing")
    n = len(y)
    if num_idx and (X_num is None or X_num.shape != (n, len(num_idx))):
        raise ValueError(f"{gen_dir}: X_num shape {None if X_num is None else X_num.shape}, "
                         f"expected ({n}, {len(num_idx)})")
    if cat_idx and (X_cat is None or X_cat.shape != (n, len(cat_idx))):
        raise ValueError(f"{gen_dir}: X_cat shape {None if X_cat is None else X_cat.shape}, "
                         f"expected ({n}, {len(cat_idx)})")

    if info["task_type"] == "regression":
        target = np.asarray(y, dtype=np.float64)
    else:
        labels = info["label_tokens"]
        yi = np.asarray(y).astype(np.int64)
        if yi.min() < 0 or yi.max() >= len(labels):
            raise ValueError(f"{gen_dir}: class index outside 0..{len(labels) - 1}")
        target = np.asarray(labels, dtype=object)[yi]

    data = {}
    for i, name in enumerate(cols):
        if i in num_idx:
            data[name] = np.asarray(X_num[:, num_idx.index(i)], dtype=np.float64)
        elif i in cat_idx:
            data[name] = np.asarray(X_cat[:, cat_idx.index(i)]).astype(str)
        elif i == tgt_idx:
            data[name] = target
        else:
            raise ValueError(f"column {i} ({name}) is in no index list")
    return pd.DataFrame(data, columns=cols)


def write_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def export_seed(gen_dir: str, data_dir: str, out_csv: str) -> str:
    df = to_frame(gen_dir, data_dir)
    write_csv(df, out_csv)
    check_against_real(out_csv, data_dir)
    return out_csv


def check_against_real(csv_path: str, data_dir: str) -> None:
    """The exported table parses like TabDiff's real.csv: same columns in the
    same order, same dtypes, categorical / label values inside the real
    categories, no missing values the real table does not have."""
    info = load_info(data_dir)
    real = pd.read_csv(os.path.join(data_dir, info["tabdiff"]["real"]))
    syn = pd.read_csv(csv_path)
    if list(syn.columns) != list(real.columns):
        raise ValueError(f"{csv_path}: columns {list(syn.columns)} != real {list(real.columns)}")
    cols = info["column_names"]
    cat_like = [cols[i] for i in info["cat_col_idx"]]
    if info["task_type"] != "regression":
        cat_like.append(cols[info["target_col_idx"][0]])
    problems = []
    for c in real.columns:
        rk, sk = real[c].dtype.kind, syn[c].dtype.kind
        if c in cat_like:
            if rk != sk:
                problems.append(f"{c}: dtype {syn[c].dtype} != real {real[c].dtype}")
            extra = set(syn[c].dropna().unique()) - set(real[c].dropna().unique())
            if extra:
                problems.append(f"{c}: categories not in real: {sorted(map(str, extra))[:5]}")
        elif not (sk in "fi" and rk in "fi"):
            problems.append(f"{c}: numerical column parsed as {syn[c].dtype} (real {real[c].dtype})")
        if syn[c].isna().any() and not real[c].isna().any():
            problems.append(f"{c}: {int(syn[c].isna().sum())} missing values")
    if problems:
        raise ValueError(f"{csv_path} does not match TabDiff's raw format:\n  " + "\n  ".join(problems))
