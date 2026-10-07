#!/usr/bin/env python3
"""
Build TabDiff's OWN splits for the TabDiff-transfer study, then convert them
into this repo's npy layout. Run locally (needs internet for adult/default);
upload the resulting data/tabdiff_* folders.

    python3 hpc_transfer/tabdiff_study/prepare_data.py [--raw_dir ~/Desktop/Thesis/new_datasets]

1. Work directory data/tabdiff_raw/ is laid out like a TabDiff checkout: the
   UNCHANGED download_dataset.py, process_dataset.py and data/Info/*.json are
   copied in from third_party/TabDiff (digests verified first), because
   process_dataset.py works on relative paths and rewrites data/Info/news*.json
   -- running it inside the submodule would modify it.
2. Raw files
     adult, default (+ _dcr)  TabDiff's own download_dataset.download_from_uci
                              (UCI zips; the only download)
     shoppers, magic, beijing, news (+ _dcr)
                              the local raw UCI files in --raw_dir, copied to
                              the paths TabDiff's Info files expect
3. `python process_dataset.py --dataname <name>` (unchanged, cwd = work dir):
   TabDiff's split (seed 1234, +1 until train holds every category; 90/10, or
   50/50 for *_dcr; adult keeps the official test file).
4. Conversion into data/tabdiff_<name>/ (prefix: never collides with data/<ds>/):
     tabdiff/info.json, real.csv, test.csv   TabDiff's processed files, copied
                                             verbatim -- what their evaluator reads
     X_num_*.npy, X_cat_*.npy, y_*.npy       our layout, built FROM real.csv /
                                             test.csv as text, so categorical
                                             tokens round-trip exactly
                                             (export.py writes them back)
     val split                               a copy of test (our loader needs a
                                             val file; training/normalisation
                                             read only train)
     info.json                               our metadata + column map + label
                                             tokens + provenance
   and data/tabdiff_manifest.json (sha256 of every raw and produced file).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study as S  # noqa: E402

DOWNLOAD = ("adult", "default")
# TabDiff dataname -> (file in --raw_dir, path TabDiff's Info expects, relative to the work dir)
LOCAL_RAW = {
    "shoppers": ("online_shoppers_intention.csv", "online_shoppers_intention.csv"),
    "magic":    ("magic+gamma+telescope/magic04.data", "magic04.data"),
    "beijing":  ("PRSA_data_2010.1.1-2014.12.31.csv", "PRSA_data_2010.1.1-2014.12.31.csv"),
    "news":     ("OnlineNewsPopularity/OnlineNewsPopularity.csv",
                 "OnlineNewsPopularity/OnlineNewsPopularity.csv"),
}


def all_names():
    out = []
    for ds in S.ORDER:
        out.append(ds)
        if S.HAS_DCR[ds]:
            out.append(ds + S.DCR_SUFFIX)
    return out


def setup_workdir(work: str, tabdiff: str) -> None:
    S.verify_tabdiff(tabdiff)
    os.makedirs(os.path.join(work, "data", "Info"), exist_ok=True)
    for f in ("download_dataset.py", "process_dataset.py"):
        shutil.copy2(os.path.join(tabdiff, f), os.path.join(work, f))
    for name in all_names():
        shutil.copy2(os.path.join(tabdiff, "data", "Info", f"{name}.json"),
                     os.path.join(work, "data", "Info", f"{name}.json"))


def fetch_raw(work: str, name: str, raw_dir: str) -> None:
    base = name[:-len(S.DCR_SUFFIX)] if name.endswith(S.DCR_SUFFIX) else name
    if base in DOWNLOAD:
        # TabDiff's own downloader; it skips a folder that already exists.
        code = ("import download_dataset as d; d.download_from_uci(%r)" % name)
        subprocess.run([sys.executable, "-c", code], cwd=work, check=True)
        if base == "default":
            # The UCI zip nests the xls in another zip in some mirrors; TabDiff's
            # Info expects it at data/<name>/default of credit card clients.xls.
            target = os.path.join(work, "data", name, "default of credit card clients.xls")
            if not os.path.exists(target):
                for root, _, files in os.walk(os.path.join(work, "data", name)):
                    for f in files:
                        if f.endswith(".zip") and f != f"{name}.zip":
                            with zipfile.ZipFile(os.path.join(root, f)) as z:
                                z.extractall(os.path.join(work, "data", name))
            if not os.path.exists(target):
                raise FileNotFoundError(target)
        return
    src_rel, dst_rel = LOCAL_RAW[base]
    src = os.path.join(raw_dir, src_rel)
    dst = os.path.join(work, "data", name, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def process(work: str, name: str) -> None:
    log = os.path.join(work, f"process_{name}.log")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "process_dataset.py", "--dataname", name],
                           cwd=work, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"TabDiff process_dataset.py failed for {name}; see {log}")


def convert(work: str, name: str, out_root: str) -> dict:
    """data/tabdiff_<name>/ from TabDiff's processed output for <name>."""
    target = S.PREFIX + name
    S.parse_target(target)
    info = json.load(open(os.path.join(work, "data", name, "info.json")))
    real_p = os.path.join(work, "synthetic", name, "real.csv")
    test_p = os.path.join(work, "synthetic", name, "test.csv")
    out = os.path.join(out_root, target)
    os.makedirs(os.path.join(out, "tabdiff"), exist_ok=True)
    shutil.copy2(os.path.join(work, "data", name, "info.json"), os.path.join(out, "tabdiff", "info.json"))
    shutil.copy2(real_p, os.path.join(out, "tabdiff", "real.csv"))
    shutil.copy2(test_p, os.path.join(out, "tabdiff", "test.csv"))

    cols = list(info["column_names"])
    num = [cols[i] for i in info["num_col_idx"]]
    cat = [cols[i] for i in info["cat_col_idx"]]
    tgt = cols[info["target_col_idx"][0]]
    task = info["task_type"]
    # Text tokens exactly as the CSV holds them (what pd.read_csv will parse
    # for both real and synthetic tables), and floats as pandas parses them.
    txt = {s: pd.read_csv(p, dtype=str, keep_default_na=False) for s, p in (("train", real_p), ("test", test_p))}
    parsed = {s: pd.read_csv(p) for s, p in (("train", real_p), ("test", test_p))}
    if list(txt["train"].columns) != cols:
        raise ValueError(f"{name}: real.csv header {list(txt['train'].columns)} != info column_names")

    labels = None
    if task != "regression":
        labels = sorted(txt["train"][tgt].unique().tolist())
    for split, src in (("train", "train"), ("val", "test"), ("test", "test")):
        X_num = parsed[src][num].to_numpy(np.float32) if num else None
        X_cat = txt[src][cat].to_numpy(dtype=str) if cat else None
        if task == "regression":
            y = parsed[src][tgt].to_numpy(np.float32)
        else:
            idx = {t: i for i, t in enumerate(labels)}
            y = np.array([idx[t] for t in txt[src][tgt]], dtype=np.int64)
        if X_num is not None:
            np.save(os.path.join(out, f"X_num_{split}.npy"), X_num)
        if X_cat is not None:
            np.save(os.path.join(out, f"X_cat_{split}.npy"), X_cat)
        np.save(os.path.join(out, f"y_{split}.npy"), y)

    n = {s: len(parsed[s]) for s in parsed}
    ours = {
        "name": f"{name} (TabDiff split)",
        "id": f"{target}--tabdiff",
        "task_type": task,
        "n_num_features": len(num),
        "n_cat_features": len(cat),
        "train_size": n["train"], "val_size": n["test"], "test_size": n["test"],
        "val_is_test_copy": True,
        "num_col_names": num, "cat_col_names": cat, "target_col": tgt,
        "column_names": cols,
        "num_col_idx": info["num_col_idx"], "cat_col_idx": info["cat_col_idx"],
        "target_col_idx": info["target_col_idx"],
        "label_tokens": labels,
        "tabdiff": {"dataname": name, "commit": S.TABDIFF_COMMIT,
                    "processed_by": "third_party/TabDiff/process_dataset.py (unchanged)",
                    "info": "tabdiff/info.json", "real": "tabdiff/real.csv", "test": "tabdiff/test.csv"},
    }
    if labels is not None:
        ours["n_classes"] = len(labels)
    with open(os.path.join(out, "info.json"), "w") as fh:
        json.dump(ours, fh, indent=2)
    return {"target": target, "train": n["train"], "test": n["test"], "num": len(num),
            "cat": len(cat), "task": task, "labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default=os.path.expanduser("~/Desktop/Thesis/new_datasets"))
    ap.add_argument("--repo_root", default=S.REPO)
    ap.add_argument("--only", nargs="*", default=None, help="subset of TabDiff datanames")
    args = ap.parse_args()

    tabdiff = os.path.join(args.repo_root, S.TABDIFF_REL)
    work = os.path.join(args.repo_root, "data", "tabdiff_raw")
    setup_workdir(work, tabdiff)
    names = args.only or all_names()
    rows = []
    for name in names:
        print(f"== {name}", flush=True)
        fetch_raw(work, name, args.raw_dir)
        process(work, name)
        rows.append(convert(work, name, os.path.join(args.repo_root, "data")))
        print("   ", rows[-1], flush=True)

    manifest_p = os.path.join(args.repo_root, "data", "tabdiff_manifest.json")
    manifest = json.load(open(manifest_p)) if os.path.exists(manifest_p) else {}
    manifest["tabdiff"] = S.verify_tabdiff(tabdiff)
    manifest.setdefault("datasets", {})
    for r in rows:
        d = os.path.join(args.repo_root, "data", r["target"])
        files = sorted(os.path.relpath(os.path.join(dp, f), d)
                       for dp, _, fs in os.walk(d) for f in fs)
        name = r["target"][len(S.PREFIX):]
        raw = os.path.join(work, "data", name)
        produced = {"train.csv", "test.csv", "val.csv", "info.json", f"{name}.csv", "test.data"}
        raw_files = sorted(os.path.relpath(os.path.join(dp, f), work)
                           for dp, _, fs in os.walk(raw) for f in fs
                           if f not in produced and not f.endswith(".npy"))
        manifest["datasets"][r["target"]] = {
            **r,
            "files_sha256_16": {f: S.sha16(os.path.join(d, f)) for f in files},
            "raw_sha256_16": {f: S.sha16(os.path.join(work, f)) for f in raw_files},
        }
    with open(manifest_p, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {manifest_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
