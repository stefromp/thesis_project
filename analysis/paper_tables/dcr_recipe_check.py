#!/usr/bin/env python3
"""Is TabDDPM's DCR (exp/<ds>/ddpm_cb_best/privacy.json) computed like ours?

    python3 analysis/paper_tables/dcr_recipe_check.py

Runs the authors' scripts/resample_privacy.py::privacy_metrics VERBATIM (its
source is extracted with ast and executed, nothing re-typed) on our own
generation-seed-0 samples, and compares
    np.median(privacy_metrics(...))    <- what calc_privacy writes to privacy.json
with the values our runs stored per generation seed:
    dcr_baseline       (compute_dcr_baseline: median, target included)
    dcr_tabddpm_mean   (compute_dcr_tabddpm_mean defaults: mean, no target)

privacy_metrics builds a dense synthetic x real distance matrix, so the
verbatim run is limited to cells where that matrix fits (MAX_PAIRS). For the
other datasets the one recipe difference that can matter -- the authors
subsample the synthetic (and real) rows to 50,000 -- is quantified with an
exact nearest-neighbour search: median over a random 50k subsample vs all rows.

Writes DCR_RECIPE_CHECK.md and dcr_recipe_check.csv.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "analysis" / "dependency_fidelity"))
sys.path.insert(0, str(HERE))

from cat_input_local_check import _setup  # noqa: E402  (stubs heavy optional imports of lib)

_setup()
import lib  # noqa: E402
import eval_tabddpm_protocol as ep  # noqa: E402

import paper_tables as pt  # noqa: E402  (same run cells as the tables)

SRC = REPO / "scripts" / "resample_privacy.py"
MAX_PAIRS = 1.5e8            # dense float64 distance matrix <= 1.2 GB
CAP = 50_000                 # resample_privacy.py:56-64


def authors_privacy_metrics():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "privacy_metrics")
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.metrics import pairwise_distances
    # Only shim: the authors' sklearn accepted the np.matrix that .todense()
    # returns (resample_privacy.py:75-76); current sklearn rejects it. np.asarray
    # gives the same values.
    def pairwise_distances_shim(X, Y=None, **kw):
        return pairwise_distances(np.asarray(X), None if Y is None else np.asarray(Y), **kw)
    ns = {"np": np, "lib": lib, "MinMaxScaler": MinMaxScaler, "OneHotEncoder": OneHotEncoder,
          "pairwise_distances": pairwise_distances_shim}
    exec(compile(ast.Module([fn], []), str(SRC), "exec"), ns)
    return ns["privacy_metrics"], fn.lineno, fn.end_lineno


def stored(run: Path, metric: str, seed: int = 0) -> float:
    raw = pd.concat([pd.read_csv(f) for f in sorted(run.glob("per_seed_scores*.csv"))])
    v = raw[(raw.metric == metric) & (raw.gen_seed == seed)].value
    return float(v.iloc[-1]) if len(v) else np.nan


def nn_dists(real_dir: Path, gen_dir: Path, task: str, n_syn: int | None, n_real: int | None, rng):
    """Our encoding (= the authors', see the side-by-side), exact NN, optional
    50k subsampling like resample_privacy.py:56-64."""
    from sklearn.neighbors import NearestNeighbors
    rn, rc, ry = ep._read(str(real_dir), "train")
    sn, sc, sy = ep._read(str(gen_dir), "train")
    rn, rc = ep._append_label(rn, rc, ry, task)
    sn, sc = ep._append_label(sn, sc, sy, task)
    if n_real and len(ry) > n_real:
        ix = rng.choice(len(ry), n_real, replace=False)
        rn, rc = rn[ix], (rc[ix] if rc is not None else None)
    if n_syn and len(sy) > n_syn:
        ix = rng.choice(len(sy), n_syn, replace=False)
        sn, sc = sn[ix], (sc[ix] if sc is not None else None)
    Xr, Xs = ep._encode(rn, rc, (rn, rc), (sn, sc))
    return NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xr).kneighbors(Xs)[0][:, 0]


def main():
    pm, l0, l1 = authors_privacy_metrics()
    pt.TASK.update({ds: pt.task_of(ds) for ds in pt.DS})
    ours, _ = pt.load_ours()
    ours = ours[ours["mean"].notna()]
    rows = []
    for _, r in ours.iterrows():
        run = REPO / r.source
        gen = run / "gen_seed_0"
        n_syn = len(np.load(gen / "y_train.npy", allow_pickle=True))
        n_real = len(np.load(REPO / "data" / r.dataset / "y_train.npy", allow_pickle=True))
        row = {"dataset": r.dataset, "method": r.method, "n_syn": n_syn, "n_real": n_real,
               "stored_dcr_baseline": stored(run, "dcr_baseline"),
               "stored_dcr_tabddpm_mean": stored(run, "dcr_tabddpm_mean")}
        if min(n_syn, CAP) * min(n_real, CAP) <= MAX_PAIRS:
            np.random.seed(0)                       # resample_privacy.py:22 zero.improve_reproducibility(0)
            d = pm(str(REPO / "data" / r.dataset), str(gen))
            row.update(authors_median=float(np.median(d)), authors_mean_with_label=float(np.mean(d)),
                       check="verbatim")
        else:
            rng = np.random.default_rng(0)
            full = nn_dists(REPO / "data" / r.dataset, gen, pt.TASK[r.dataset], None, None, rng)
            sub = [np.median(nn_dists(REPO / "data" / r.dataset, gen, pt.TASK[r.dataset], CAP, CAP,
                                      np.random.default_rng(s))) for s in range(3)]
            row.update(full_median=float(np.median(full)), sub50k_median_mean=float(np.mean(sub)),
                       sub50k_median_min=float(np.min(sub)), sub50k_median_max=float(np.max(sub)),
                       check="50k-subsample effect")
        rows.append(row)
        print(row)
    df = pd.DataFrame(rows)
    v = df[df.check == "verbatim"]
    s = df[df.check != "verbatim"]
    df.to_csv(HERE / "dcr_recipe_check.csv", index=False)

    rel_b = (v.authors_median / v.stored_dcr_baseline - 1).abs()
    rel_m = (v.authors_median / v.stored_dcr_tabddpm_mean - 1)
    rel_s = ((s.sub50k_median_mean / s.full_median - 1).abs()) if len(s) else pd.Series(dtype=float)
    rel_full = ((s.full_median / s.stored_dcr_baseline - 1).abs()) if len(s) else pd.Series(dtype=float)
    L = ["# Does TabDDPM's privacy.json compute DCR like our runs?\n",
         "_Generated by `dcr_recipe_check.py`._\n",
         f"The authors' `privacy_metrics` ({ct_rel(SRC)}:{l0}-{l1}) was executed verbatim on our gen-seed-0 "
         f"samples (only shim: `np.asarray` around the `np.matrix` inputs of `pairwise_distances`, which current "
         f"sklearn rejects; same values).\n",
         f"- **vs `dcr_baseline`** ({len(v)} cells, verbatim): max |relative difference| = {rel_b.max():.2e}.",
         f"- **vs `dcr_tabddpm_mean`** (same cells): authors / ours − 1 ranges from {rel_m.min():+.1%} to "
         f"{rel_m.max():+.1%}.",
         (f"- **50k subsampling** (resample_privacy.py:56-64), on the {len(s)} cells too large for the verbatim run: "
          f"the median over a random 50k subsample differs from the all-rows median by at most "
          f"{rel_s.max():.2%} (mean of 3 subsamples), and the all-rows median reproduces the stored "
          f"`dcr_baseline` to {rel_full.max():.1e}.") if len(s) else "",
         "", "Per cell: `dcr_recipe_check.csv`.", ""]
    (HERE / "DCR_RECIPE_CHECK.md").write_text("\n".join(L))
    print("\n".join(L))


def ct_rel(p: Path) -> str:
    return str(p.relative_to(REPO))


if __name__ == "__main__":
    main()
