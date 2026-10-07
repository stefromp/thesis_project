#!/usr/bin/env python3
"""
Build data/tabdiff_gnn/tabdiff_<name>/ -- exactly what TabDiff's own pipeline
reads -- from TabDiff's processed output. Run locally (diabetes needs internet
once); upload data/tabdiff_gnn/.

    python3 tabdiff_gnn/prepare_data.py [--only diabetes]

Work directory = data/tabdiff_raw/, the TabDiff-layout copy the TabDiff study
already uses (hpc_transfer/tabdiff_study/prepare_data.py): the UNCHANGED
download_dataset.py / process_dataset.py and data/Info/*.json copied from
third_party/TabDiff (digests verified), so nothing runs inside the submodule.

No DCR in this study: only the main splits are built (no _dcr targets).

  6 datasets of the TabDiff study: their processed output is reused
      as it is (the same split the TabDiff study evaluates on -- checked by
      the tests); processed again only if missing.
  diabetes: Info copied in, TabDiff's download_dataset.download_from_uci
      ('diabetes' -> UCI 296), then `python process_dataset.py --dataname <name>`
      unchanged (CDTD-aligned preprocessing; 60/20/20 split, random_state 42 --
      the only TabDiff dataset with a val split).

Per target, copied verbatim:
  X_num_{train,test}.npy X_cat_{train,test}.npy y_{train,test}.npy info.json
      (data/<name>/ -- read by TabDiffDataset / src.make_dataset)
  real.csv test.csv [val.csv]
      (synthetic/<name>/ -- read by TabMetrics; val.csv only where TabDiff
      writes one, so their MLE uses it exactly when theirs would)
and data/tabdiff_gnn/manifest.json (sha256/16 of every file + the TabDiff pins).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

NPY = [f"{k}_{s}.npy" for k in ("X_num", "X_cat", "y") for s in ("train", "test")]
CSV = ["real.csv", "test.csv", "val.csv"]
DOWNLOAD_HERE = ("diabetes",)            # fetched by this script (TabDiff's downloader)


def workdir(repo: str) -> str:
    return os.path.join(repo, "data", "tabdiff_raw")


def ensure_workdir(repo: str, names) -> str:
    work = workdir(repo)
    tabdiff = os.path.join(repo, C.TABDIFF_REL)
    C.verify_tabdiff(repo)
    os.makedirs(os.path.join(work, "data", "Info"), exist_ok=True)
    for f in ("download_dataset.py", "process_dataset.py"):
        dst = os.path.join(work, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(tabdiff, f), dst)
        if C.sha16(dst) != C.sha16(os.path.join(tabdiff, f)):
            raise RuntimeError(f"{dst} differs from TabDiff's {f}")
    for name in names:
        dst = os.path.join(work, "data", "Info", f"{name}.json")
        if not os.path.exists(dst):   # process_dataset.py rewrites its Info later: copy once
            shutil.copy2(os.path.join(tabdiff, "data", "Info", f"{name}.json"), dst)
    return work


def processed(work: str, name: str) -> bool:
    d = os.path.join(work, "data", name)
    return (os.path.exists(os.path.join(d, "info.json"))
            and os.path.exists(os.path.join(d, "y_train.npy"))
            and os.path.exists(os.path.join(work, "synthetic", name, "real.csv")))


def fetch(work: str, name: str) -> None:
    base = name
    if base not in DOWNLOAD_HERE:
        raise RuntimeError(f"{name} is not processed in {work}: run "
                           f"hpc_transfer/tabdiff_study/prepare_data.py --only {name} first")
    code = "import download_dataset as d; d.download_from_uci(%r)" % name
    subprocess.run([sys.executable, "-c", code], cwd=work, check=True)
    want = os.path.join(work, "data", name, "diabetic_data.csv")
    if not os.path.exists(want):           # the UCI zip may nest the csv in a folder
        for root, _, files in os.walk(os.path.join(work, "data", name)):
            if "diabetic_data.csv" in files:
                shutil.copy2(os.path.join(root, "diabetic_data.csv"), want)
                break
    if not os.path.exists(want):
        raise FileNotFoundError(want)


def process(work: str, name: str) -> None:
    log = os.path.join(work, f"process_{name}.log")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "process_dataset.py", "--dataname", name], cwd=work,
                           stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"TabDiff process_dataset.py failed for {name}; see {log}")


def copy_target(work: str, name: str, repo: str) -> dict:
    target = C.PREFIX + name
    C.parse_target(target)
    src_d = os.path.join(work, "data", name)
    syn_d = os.path.join(work, "synthetic", name)
    out = C.data_dir(target, repo)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    files = {}
    for f in NPY + ["info.json"]:
        p = os.path.join(src_d, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, f))
    for f in CSV:
        p = os.path.join(syn_d, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, f))
    for f in sorted(os.listdir(out)):
        files[f] = C.sha16(os.path.join(out, f))
    info = json.load(open(os.path.join(out, "info.json")))
    return {"target": target, "tabdiff_dataname": name, "task_type": info["task_type"],
            "train_num": info.get("train_num"), "test_num": info.get("test_num"),
            "has_val_csv": "val.csv" in files, "files_sha256_16": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_root", default=C.REPO)
    ap.add_argument("--only", nargs="*", default=None, help="subset of TabDiff datanames")
    args = ap.parse_args(argv)
    names = args.only or [C.tabdiff_dataname(t) for t in C.all_targets()]
    work = ensure_workdir(args.repo_root, names)
    rows = {}
    for name in names:
        if not processed(work, name):
            print(f"== {name}: download + TabDiff process_dataset.py", flush=True)
            fetch(work, name)
            process(work, name)
        rows[C.PREFIX + name] = copy_target(work, name, args.repo_root)
        r = rows[C.PREFIX + name]
        print(f"   {r['target']:22s} {r['task_type']:10s} train={r['train_num']} test={r['test_num']} "
              f"val.csv={r['has_val_csv']}", flush=True)
    man_p = os.path.join(args.repo_root, C.DATA_REL, "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    man["tabdiff"] = C.verify_tabdiff(args.repo_root)
    man.setdefault("datasets", {}).update(rows)
    man["datasets"] = {k: v for k, v in man["datasets"].items() if k in C.all_targets()}
    with open(man_p, "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"wrote {man_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
