#!/usr/bin/env python3
"""PYTHON_BIN stand-in for hpc_transfer/dryrun/dryrun_job.sh.

`mock_python.py - ARGS` (a preflight heredoc on stdin) runs the real
interpreter unchanged. `mock_python.py <repo>/ablation/<ds>_ablation.py ARGS`
runs that script for real -- argparse, run_ablation, _run_one, the
training_setup read of config.toml, the result tree, run_meta.json -- but with
train / sample / CatBoost / metrics replaced by the stubs of
hpc_transfer/test_split_runner_backcompat.py. Nothing is trained or sampled.
Every stubbed call is printed with the arguments that matter here.
"""
import os
import runpy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

if len(sys.argv) < 2 or sys.argv[1] == "-" or not sys.argv[1].endswith("_ablation.py"):
    sys.exit(subprocess.call([sys.executable] + sys.argv[1:]))

sys.path.insert(0, os.path.join(REPO, "hpc_transfer"))
import test_split_runner_backcompat as bc  # noqa: E402

ar = bc._import_runner()


class Rec(bc._Recorder):
    def record(self, name, **kw):
        super().record(name, **kw)
        keys = {"train": ("lr", "steps", "batch_size", "weight_decay", "num_timesteps", "scheduler",
                          "divergence_check", "seed"),
                "sample": ("seed", "num_timesteps", "scheduler", "num_samples", "batch_size", "model_path"),
                "eval": ("seed",)}[name]
        shown = {k: kw.get(k) for k in keys}
        if name == "train":
            shown["cat_input"] = kw["graph_params"]["cat_input"]
            shown["parent_dir"] = kw["parent_dir"]
        if name != "eval" or kw.get("seed") == 0:
            print(f"  [MOCK {name}] {shown}", flush=True)


bc._stub_world(Rec(), None)
ar._load_raw_real = lambda p: (None, None, [], None)
ar._load_raw_syn = lambda *a, **k: None
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
