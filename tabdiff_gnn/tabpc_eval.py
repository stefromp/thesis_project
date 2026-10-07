#!/usr/bin/env python3
"""
wNMIS and C2ST (XGBoost) of every table, computed by april-tools/tabpc's own
metric classes (third_party/tabpc @ config.TABPC_COMMIT, digests checked,
called unchanged). Runs in the tabpc venv (tabpc's pins: sdmetrics 0.24.0,
torch 2.9.1).

    python tabdiff_gnn/tabpc_eval.py --run_dir RUN --target tabdiff_adult [--device cpu]

Per table k (RUN/samples/samples_<k>.csv), exactly as tabpc's
src/evaluation.generative_performance_metric(original=<dataset folder>, ...)
calls a metric: metric(df_train, df_generated, df_test, info, device=...),
with df_train = TabDiff's train split (real.csv, byte-identical to
data/<name>/train.csv), df_test = test.csv, info = TabDiff's info.json:

  wnmis          MI_l1()  ["nmi_l1_weighted_complement"]   (tabpc's "wNMIS")
  c2st_xgb       C2ST()   ["xgboost (C2ST)"]               (tabpc's "C2ST (XGB)")
  + the other keys the same two calls return (nmi_l1, nmi_l1_weighted,
    logistic_regression (C2ST), AUC versions), kept for completeness.
numpy's global RNG is seeded with k before each table (XGBoostDetection fixes
random_state=0; the folds of sdmetrics' detection metrics use their own seed).

Outputs RUN/tabpc_eval/: tabpc_results.csv (one row per table), avg_std.csv
(mean, std ddof=1), tabpc_meta.json (commit, digests, versions).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

OUT_DIR = "tabpc_eval"
COLUMNS = ["wnmis", "c2st_xgb"]


def _versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for d in ("sdmetrics", "sdv", "rdt", "xgboost", "torch", "torchmetrics", "pandas", "numpy",
              "scikit-learn", "scipy"):
        try:
            out[d] = metadata.version(d)
        except Exception as exc:
            out[d] = f"unavailable ({type(exc).__name__})"
    return out


def check_pins(versions: dict) -> None:
    bad = {k: (versions.get(k), v) for k, v in C.TABPC_PINS.items()
           if not str(versions.get(k, "")).startswith(v)}
    if bad:
        raise RuntimeError("tabpc venv differs from its pins: "
                           + ", ".join(f"{k} {a} != {b}" for k, (a, b) in bad.items()))


def import_tabpc(repo: str = C.REPO):
    root = os.path.join(repo, C.TABPC_REL)
    # tabpc first: its metrics.py imports `src.*` (TabDiff has a `src` too) and
    # `tabdiff_eval.mle.mle` (hpc_transfer/tabdiff_study/tabdiff_eval.py, on the
    # path through config, has the same top-level name).
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for m in [m for m in list(sys.modules)
              if m in ("src", "tabdiff_eval") or m.startswith(("src.", "tabdiff_eval."))]:
        del sys.modules[m]
    from src.metrics import metrics as M
    if not os.path.realpath(M.__file__).startswith(os.path.realpath(root)):
        raise RuntimeError(f"imported {M.__file__}, not tabpc's metrics")
    return M


def flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": float(v) for k, v in d.items() if isinstance(v, (int, float))}


def evaluate(run_dir: str, target: str, device: str = "cpu", n_runs: int = C.N_TABLES,
             check: bool = True, repo: str = C.REPO) -> str:
    import numpy as np
    import pandas as pd
    C.parse_target(target)
    tp = C.verify_tabpc(repo)
    versions = _versions()
    if check:
        check_pins(versions)
    M = import_tabpc(repo)
    files = C.data_files(target, repo)
    info = json.load(open(files["info"]))
    df_train = pd.read_csv(files["real"])
    df_test = pd.read_csv(files["test"])
    out = os.path.join(run_dir, OUT_DIR)
    os.makedirs(out, exist_ok=True)
    rows = []
    for k in range(n_runs):
        path = os.path.join(run_dir, "samples", f"samples_{k}.csv")
        syn = pd.read_csv(path)
        np.random.seed(k)
        mi = M.MI_l1()(df_train, syn, df_test, info, device=device)
        c2 = M.C2ST()(df_train, syn, df_test, info, device=device)
        row = {"run": k, "wnmis": float(mi[C.TABPC_KEYS["wnmis"][1]]),
               "c2st_xgb": float(c2[C.TABPC_KEYS["c2st_xgb"][1]])}
        row.update(flat("mi_l1", mi))
        row.update(flat("C2ST", c2))
        rows.append(row)
        print(f"  [tabpc] table {k}: wNMIS={row['wnmis']:.4f}  C2ST(XGB)={row['c2st_xgb']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "tabpc_results.csv"), index=False)
    vals = df.drop(columns=["run"])
    pd.DataFrame({"avg": vals.mean(axis=0), "std": vals.std(axis=0, ddof=1)}).to_csv(
        os.path.join(out, "avg_std.csv"))
    with open(os.path.join(out, "tabpc_meta.json"), "w") as fh:
        json.dump({"finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "n_runs": n_runs, "tabpc": tp, "versions": versions, "device": device,
                   "keys": C.TABPC_KEYS, "real": files["real"], "test": files["test"]}, fh, indent=2)
    return out


def done(run_dir: str, n_runs: int) -> bool:
    p = os.path.join(run_dir, OUT_DIR, "tabpc_results.csv")
    m = os.path.join(run_dir, OUT_DIR, "tabpc_meta.json")
    if not (os.path.exists(p) and os.path.exists(m)):
        return False
    if json.load(open(m)).get("n_runs") != n_runs:
        return False
    with open(p) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    return len(lines) == n_runs + 1 and all(c in lines[0].split(",") for c in COLUMNS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_runs", type=int, default=C.N_TABLES)
    ap.add_argument("--no_pin_check", action="store_true")
    a = ap.parse_args(argv)
    out = evaluate(a.run_dir, a.target, a.device, a.n_runs, not a.no_pin_check)
    print(f"  [OK] tabpc -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
