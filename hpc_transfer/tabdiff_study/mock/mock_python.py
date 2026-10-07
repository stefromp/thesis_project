"""
MOCK interpreter bootstrap for the local dry run (dryrun_mock.sh). NEVER used
by the cluster jobs. Runs a script (or stdin with '-') exactly as python would,
after three loudly-labelled substitutions so a real job script finishes on a
laptop in minutes:

  MOCK_STEPS=<n>    training steps (study.train_config's value is replaced and
                    its source string tagged "[MOCK]")
  MOCK_RUNS=<n>     tables per target instead of 20 (study.N_SAMPLE_RUNS)
  MOCK_CPU_XGB=1    xgboost tree_method 'gpu_hist' -> 'hist' (macOS wheels have
                    no CUDA build; TabDiff's eval/mle/mle.py asks for gpu_hist)
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # hpc_transfer/tabdiff_study

if os.environ.get("MOCK_CPU_XGB") == "1":
    try:
        import xgboost.sklearn as xs
        _init = xs.XGBModel.__init__

        def _patched(self, *a, **kw):
            if kw.get("tree_method") == "gpu_hist":
                kw["tree_method"] = "hist"
            _init(self, *a, **kw)
        # keep the signature xgboost's sklearn wrapper introspects
        _patched.__signature__ = __import__("inspect").signature(_init)
        xs.XGBModel.__init__ = _patched
    except ImportError:
        pass

if any(k in os.environ for k in ("MOCK_STEPS", "MOCK_RUNS")):
    import study as S
    if "MOCK_RUNS" in os.environ:
        S.N_SAMPLE_RUNS = int(os.environ["MOCK_RUNS"])
    if "MOCK_STEPS" in os.environ:
        _tc = S.train_config

        def train_config(src, repo_root=S.REPO):
            v, s = _tc(src, repo_root)
            v, s = dict(v), dict(s)
            v["steps"] = int(os.environ["MOCK_STEPS"])
            s["steps"] = s["steps"] + " [MOCK: overridden by MOCK_STEPS]"
            return v, s
        S.train_config = train_config

script = sys.argv[1]
sys.argv = sys.argv[1:]
if script == "-":
    code = sys.stdin.read()
    sys.argv[0] = "-"
    exec(compile(code, "<stdin>", "exec"), {"__name__": "__main__"})
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
