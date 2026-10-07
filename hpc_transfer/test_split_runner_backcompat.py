#!/usr/bin/env python3
"""
Guards for hpc_transfer/split_runner.py.

Two jobs:

1. EQUIVALENCE -- with no split flags, split_runner.run_split must produce the
   same call sequence, the same seeds and the same output file names as
   ablation_runner._run_one. Every heavy function is stubbed, so this runs
   anywhere and needs no GPU, no data and no working `lib`.

2. DRIFT GUARD -- split_runner mirrors four fragments that are inline in
   _run_one and cannot be imported (T_dict_eval, the tabddpm_sample call, the
   train_eval_model call, the classifier metric wiring). Each is pinned by a
   source digest. If ablation_runner changes, these fail loudly and name the
   fragment to re-check, instead of the split runs silently diverging.

Run:  python3 hpc_transfer/test_split_runner_backcompat.py
      pytest hpc_transfer/test_split_runner_backcompat.py -v
"""
from __future__ import annotations

import hashlib
import inspect
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Digests of the fragments split_runner.py mirrors. Update ONLY together with
# the corresponding code in split_runner.py, after re-reading the fragment.
PINNED = {
    "subdir_rule":  "d1045adce726d56e",
    "T_dict_eval":  "9d23d9584aca7af0",
    "sample_call":  "c18ac58d3cdece22",
    "eval_call":    "0915b770103cf5fa",
    "metric_wire":  "26a14158efc36bb1",
}
# 2026-10: re-pinned for cat_input (results_subdir call, denoiser_config.json
# record after training, cat_input in graph_params). split_runner.parent_dir_for
# and the checkpoint check were updated in the same change; the sample / eval /
# metric fragments are unchanged.
# 2026-10-05: re-pinned for train_setup (training_setup() lookup; setup values
# replace ABLATION_* / dataset_cfg in the train and sample calls; results_subdir
# gets train_setup; denoiser_record()). split_runner.parent_dir_for, the
# checkpoint check and the sample call's batch_size were updated in the same
# change and read the same functions; T_dict_eval / eval / metric fragments
# are unchanged.
PINNED_RUN_ONE = "7fb213566b45b4e4"

# _run_one's keyword-only parameters, as split_runner builds them.
PINNED_RUN_ONE_PARAMS = [
    "gnn_type", "n_layers", "d_model", "top_k", "n_heads", "use_attention",
    "dataset_name", "dataset_cfg", "repo_root", "data_root", "exp_root",
    "device", "seed", "skip_if_done", "topk_mode", "k_min",
    "topk_warmup_steps", "topk_log_every", "cat_input", "train_setup",
]

FRAG_RE = {
    # Since cat_input: both _run_one and split_runner.parent_dir_for call
    # ablation_runner.results_subdir, so this pins the call, not a mirror.
    "subdir_rule": r'    ablation_subdir = results_subdir\(topk_mode, cat_input, train_setup\)',
    "T_dict_eval": r"    T_dict_eval = dict\(.*?\n    \)",
    "sample_call": r"        tabddpm_sample\(.*?\n        \)",
    "eval_call":   r"            train_eval_model\(.*?\n            \)",
    "metric_wire": r'            for metric, value in run_report\["metrics"\]\["test"\].*?clf_len_before\)',
}


def _runner_source() -> str:
    with open(os.path.join(REPO, "ablation", "ablation_runner.py")) as fh:
        return fh.read()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. Drift guard
# ---------------------------------------------------------------------------
def test_mirrored_fragments_unchanged():
    src = _runner_source()
    bad = []
    for name, pattern in FRAG_RE.items():
        m = re.search(pattern, src, re.S)
        if m is None:
            bad.append(f"{name}: fragment no longer found in ablation_runner.py")
            continue
        got = _digest(m.group(0))
        if got != PINNED[name]:
            bad.append(f"{name}: digest {got} != pinned {PINNED[name]}")
    assert not bad, (
        "ablation_runner.py changed in a place hpc_transfer/split_runner.py "
        "mirrors:\n  " + "\n  ".join(bad) +
        "\n\nRe-read those fragments, update split_runner.py to match, then "
        "update PINNED in this file. Do NOT just bump the digests.")


def test_run_one_source_unchanged():
    src = _runner_source()
    m = re.search(r"\ndef _run_one\(\n(.*?)\n# -{20,}\n# Main entry-point", src, re.S)
    assert m, "could not locate _run_one in ablation_runner.py"
    got = _digest(m.group(1))
    assert got == PINNED_RUN_ONE, (
        f"ablation_runner._run_one changed (digest {got} != {PINNED_RUN_ONE}).\n"
        "split_runner.py reproduces its orchestration; re-verify that the "
        "capture still works, that the mirrored fragments still match, and "
        "that the generation-seed derivation is unchanged. Then update "
        "PINNED_RUN_ONE.")


def test_run_one_signature_unchanged():
    ar = _import_runner()
    params = [p.name for p in inspect.signature(ar._run_one).parameters.values()]
    assert params == PINNED_RUN_ONE_PARAMS, (
        f"ablation_runner._run_one signature changed.\n"
        f"  now:    {params}\n  pinned: {PINNED_RUN_ONE_PARAMS}\n"
        "split_runner.main() builds these kwargs by name -- update it too.")


def test_delegated_function_signatures():
    """split_runner calls train.train and sample.sample with explicit kwargs."""
    ar = _import_runner()
    import train as train_mod
    import sample as sample_mod
    for mod, fn_name, needed in (
        (train_mod, "train", ["parent_dir", "real_data_path", "model_params",
                              "T_dict", "num_numerical_features", "seed",
                              "graph_params", "steps"]),
        (sample_mod, "sample", ["parent_dir", "real_data_path", "model_path",
                                "model_params", "T_dict", "seed",
                                "graph_params", "num_samples", "batch_size"]),
    ):
        sig = inspect.signature(getattr(mod, fn_name))
        missing = [n for n in needed if n not in sig.parameters]
        assert not missing, (
            f"{mod.__name__}.{fn_name} no longer accepts {missing}; "
            f"split_runner.py passes them by keyword.")


def test_gen_seed_derivation_is_the_original_call():
    """The seed passed to sample() must be the bare generation-seed integer.

    This is what makes a split run reproduce the monolithic one: sample.py
    opens with zero.improve_reproducibility(seed), a full RNG reset from that
    value alone, so seed k never depends on seeds 0..k-1.
    """
    src = _runner_source()
    call = re.search(FRAG_RE["sample_call"], src, re.S).group(0)
    assert "seed=gen_seed" in call, (
        "ablation_runner no longer passes seed=gen_seed to tabddpm_sample; "
        "split_runner's equivalence argument depends on it.")
    split = open(os.path.join(HERE, "split_runner.py")).read()
    assert "seed=gen_seed" in split, \
        "split_runner no longer passes seed=gen_seed to sample()"
    # and the reset itself
    with open(os.path.join(REPO, "scripts", "sample.py")) as fh:
        assert "zero.improve_reproducibility(seed)" in fh.read(), \
            "scripts/sample.py no longer re-seeds from its seed argument"


# ---------------------------------------------------------------------------
# 1. Equivalence, with every heavy function stubbed
# ---------------------------------------------------------------------------
def _import_runner():
    for sub in ("ablation", "scripts"):
        p = os.path.join(REPO, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    _install_lib_stub()
    import ablation_runner as ar
    ar._setup_zero_rtdl_stubs()
    return ar


def _install_lib_stub():
    """`lib` pulls category_encoders -> statsmodels, which is incompatible with
    scipy >= 1.16 in some environments. Only load_json is needed here."""
    if "lib" in sys.modules:
        return
    try:
        import lib  # noqa: F401
        return
    except Exception:
        pass
    import json as _json
    lib = types.ModuleType("lib")
    lib.load_json = lambda p: _json.load(open(p))
    # Names scripts/{train,sample,utils_train}.py import at module level. Every
    # one of them sits behind a stubbed function in this test, so they are only
    # needed for the import to succeed.
    lib.round_columns = lambda *a, **k: None
    lib.read_pure_data = lambda *a, **k: (None, None, None)
    lib.change_val = lambda d: d
    lib.transform_dataset = lambda d, t, _: d
    lib.prepare_fast_dataloader = lambda *a, **k: iter(())
    lib.concat_features = lambda *a, **k: None
    lib.dump_json = lambda o, p: _json.dump(o, open(p, "w"))
    for _n in ("Transformations", "Dataset", "TaskType", "MetricsReport"):
        setattr(lib, _n, type(_n, (), {"__init__": lambda s, *a, **k: None}))
    sys.modules["lib"] = lib


class _Recorder:
    def __init__(self):
        self.calls = []

    def record(self, name, **kw):
        self.calls.append((name, kw))

    def seq(self):
        return [c[0] for c in self.calls]

    def seeds(self, name):
        return [c[1].get("seed") for c in self.calls if c[0] == name]

    def paths(self, name, key):
        return [os.path.basename(str(c[1].get(key, ""))) for c in self.calls
                if c[0] == name]


def _stub_world(rec, tmp):
    """Replace every function that trains, samples, evaluates or reads data."""
    import train as train_mod
    import sample as sample_mod

    eval_json = "results_catboost.json"

    def fake_train(**kw):
        rec.record("train", **kw)
        open(os.path.join(kw["parent_dir"], "model_ema.pt"), "w").close()

    def fake_sample(**kw):
        rec.record("sample", **kw)
        os.makedirs(kw["parent_dir"], exist_ok=True)

    def fake_eval(**kw):
        rec.record("eval", **kw)
        import json
        with open(os.path.join(kw["parent_dir"], eval_json), "w") as fh:
            json.dump({"metrics": {"test": {"score": 0.5, "accuracy": 0.5,
                                            "macro avg": {"f1-score": 0.4}}}}, fh)

    train_mod.train = fake_train
    sample_mod.sample = fake_sample

    ec = types.ModuleType("eval_catboost")
    ec.train_catboost = fake_eval
    sys.modules["eval_catboost"] = ec

    ep = types.ModuleType("eval_tabddpm_protocol")
    for n in ("compute_dcr_tabddpm_mean", "compute_dcr_baseline"):
        setattr(ep, n, lambda *a, **k: 0.1)
    ep.compute_wasserstein_mean = lambda *a, **k: 0.2
    ep.compute_membership_inference_auc = lambda *a, **k: 0.5
    ep._encode = lambda *a, **k: None
    sys.modules["eval_tabddpm_protocol"] = ep
    return fake_eval


def _run_kwargs(tmp):
    return dict(
        gnn_type="gatv2", n_layers=3, d_model=64, top_k=3, n_heads=4,
        use_attention=False,
        dataset_name="unittest_ds",
        dataset_cfg=dict(real_data_path="data/unittest_ds/",
                         num_numerical_features=3, num_classes=2,
                         is_y_cond=True, rtdl_d_layers=[8, 8],
                         num_timesteps=100, scheduler="cosine",
                         lr=1e-4, weight_decay=0.0, batch_size=16,
                         num_samples=10, sample_batch_size=10,
                         train_normalization="quantile",
                         eval_normalization=None),
        repo_root=REPO, data_root=REPO, exp_root=tmp,
        device="cpu", seed=0,
        topk_mode="adaptive_entmax15", k_min=1,
        topk_warmup_steps=0, topk_log_every=100,
    )


def test_no_split_flags_matches_ablation_runner():
    import shutil
    import tempfile

    ar = _import_runner()
    tmp_a = tempfile.mkdtemp()
    tmp_b = tempfile.mkdtemp()
    try:
        # --- reference: the original _run_one --------------------------------
        rec_a = _Recorder()
        _stub_world(rec_a, tmp_a)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None
        ar._run_one(**{**_run_kwargs(tmp_a), "skip_if_done": False})

        # --- split_runner with no split flags --------------------------------
        sys.path.insert(0, HERE)
        import split_runner as sr
        rec_b = _Recorder()
        _stub_world(rec_b, tmp_b)
        out = sr.run_split(ar, _run_kwargs(tmp_b),
                           list(range(ar.N_GEN_SEEDS)), load_checkpoint=False)
        # main() merges once every seed is in; do the same here.
        sr.merge_shards(ar, out["parent_dir"])

        assert rec_a.seq() == rec_b.seq(), (
            "call sequence differs\n"
            f"  ablation_runner: {rec_a.seq()[:8]} ... ({len(rec_a.seq())})\n"
            f"  split_runner   : {rec_b.seq()[:8]} ... ({len(rec_b.seq())})")
        assert rec_a.seeds("sample") == rec_b.seeds("sample") \
            == list(range(ar.N_GEN_SEEDS)), "generation seeds differ"
        assert rec_a.seeds("eval") == rec_b.seeds("eval"), "classifier seeds differ"
        assert rec_a.paths("sample", "parent_dir") == \
            rec_b.paths("sample", "parent_dir"), "gen_seed directory names differ"
        assert rec_a.seeds("train") == rec_b.seeds("train") == [0], \
            "training seed differs"

        # --- the consolidated outputs must match row for row ---------------
        def _find(tmp, name):
            for root, _d, files in os.walk(tmp):
                if name in files:
                    return os.path.join(root, name)
            return None

        for name in (ar.PER_SEED_FILE, "results_full_averaged.json"):
            pa, pb = _find(tmp_a, name), _find(tmp_b, name)
            assert pa and pb, f"{name} missing ({bool(pa)}/{bool(pb)})"
            assert os.path.relpath(pa, tmp_a) == os.path.relpath(pb, tmp_b), \
                f"{name} written to a different path"

        import csv as _csv
        rows_a = sorted(map(tuple, _csv.reader(open(_find(tmp_a, ar.PER_SEED_FILE)))))
        rows_b = sorted(map(tuple, _csv.reader(open(_find(tmp_b, ar.PER_SEED_FILE)))))
        assert rows_a == rows_b, (
            f"{ar.PER_SEED_FILE} content differs\n"
            f"  only in ablation_runner: {sorted(set(rows_a) - set(rows_b))[:4]}\n"
            f"  only in split_runner   : {sorted(set(rows_b) - set(rows_a))[:4]}")

        import json as _json
        ja = _json.load(open(_find(tmp_a, "results_full_averaged.json")))
        jb = _json.load(open(_find(tmp_b, "results_full_averaged.json")))
        assert set(ja) == set(jb), (
            f"metric keys differ: only_a={sorted(set(ja)-set(jb))} "
            f"only_b={sorted(set(jb)-set(ja))}")
        diff = {k: (ja[k], jb[k]) for k in ja if ja[k] != jb[k]}
        assert not diff, f"aggregated values differ: {list(diff)[:5]}"
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


def test_split_matches_monolithic():
    """Point 5 of the spec, with the heavy functions stubbed.

    Monolithic: ablation_runner._run_one, all 5 generation seeds in one process.
    Split:      split_runner train+seed 0, then seeds 1..4 in SEPARATE calls
                with load_checkpoint=True, then a merge.

    The per-seed rows and the aggregated metrics must be identical. What this
    proves is the orchestration: the seed each sample() gets, the directory each
    seed writes to, that a loaded checkpoint is reused instead of retraining,
    and that the merge is order-independent. It cannot prove GPU bit-exactness
    -- hpc_transfer/validate_split.sh does that on the cluster.
    """
    import csv as _csv
    import json as _json
    import shutil
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr

    tmp_m = tempfile.mkdtemp()
    tmp_s = tempfile.mkdtemp()
    try:
        # --- monolithic ------------------------------------------------------
        rec_m = _Recorder()
        _stub_world(rec_m, tmp_m)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None
        ar._run_one(**{**_run_kwargs(tmp_m), "skip_if_done": False})

        # --- split: one call per generation seed ------------------------------
        rec_s = _Recorder()
        _stub_world(rec_s, tmp_s)
        out = sr.run_split(ar, _run_kwargs(tmp_s), [0], load_checkpoint=False)
        for k in range(1, ar.N_GEN_SEEDS):
            out = sr.run_split(ar, _run_kwargs(tmp_s), [k], load_checkpoint=True)
        sr.merge_shards(ar, out["parent_dir"])

        # trained exactly once, across all five calls
        n_train = rec_s.seq().count("train")
        assert n_train == 1, (
            f"split retrained {n_train} times; seeds 1..4 must reuse the "
            f"checkpoint (load_checkpoint=True)")

        # each seed sampled once, with its own integer, in its own directory
        assert rec_s.seeds("sample") == list(range(ar.N_GEN_SEEDS)), \
            f"split generation seeds: {rec_s.seeds('sample')}"
        assert rec_s.paths("sample", "parent_dir") == \
            rec_m.paths("sample", "parent_dir"), "gen_seed directories differ"

        def _find(tmp, name):
            for root, _d, files in os.walk(tmp):
                if name in files:
                    return os.path.join(root, name)
            return None

        rows_m = sorted(map(tuple, _csv.reader(open(_find(tmp_m, ar.PER_SEED_FILE)))))
        rows_s = sorted(map(tuple, _csv.reader(open(_find(tmp_s, ar.PER_SEED_FILE)))))
        assert rows_m == rows_s, (
            "per-seed rows differ between monolithic and split\n"
            f"  only monolithic: {sorted(set(rows_m) - set(rows_s))[:4]}\n"
            f"  only split     : {sorted(set(rows_s) - set(rows_m))[:4]}")

        jm = _json.load(open(_find(tmp_m, "results_full_averaged.json")))
        js = _json.load(open(_find(tmp_s, "results_full_averaged.json")))
        assert jm == js, (
            "aggregated metrics differ: "
            f"{[k for k in jm if jm.get(k) != js.get(k)][:5]}")
    finally:
        shutil.rmtree(tmp_m, ignore_errors=True)
        shutil.rmtree(tmp_s, ignore_errors=True)


def test_load_checkpoint_without_checkpoint_fails_loudly():
    """Point 4: a run whose training never finished has no checkpoint."""
    import shutil
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr
    tmp = tempfile.mkdtemp()
    try:
        rec = _Recorder()
        _stub_world(rec, tmp)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None
        try:
            sr.run_split(ar, _run_kwargs(tmp), [1], load_checkpoint=True)
        except SystemExit as exc:
            assert "no checkpoint" in str(exc), f"unclear message: {exc}"
        else:
            raise AssertionError(
                "--load_checkpoint with no model_ema.pt must abort, not retrain")
        assert rec.seq().count("train") == 0, "it trained anyway"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_partial_seed_rows_are_dropped_before_rerun():
    """A seed cut off mid-way must not leave duplicate classifier rows behind.

    The exposure is a PARTIAL resume: seeds still outstanding, so no merge runs
    yet. merge_shards rebuilds per_seed_scores.csv from shards and would mask
    the problem, so the check has to happen before it. Between the resume and
    the merge, per_seed_scores.csv is what hpc_density/density_curve.py reads,
    and a seed carrying 3 stale rows plus 10 fresh ones would be averaged wrong.
    """
    import csv as _csv
    import shutil
    import tempfile
    from collections import Counter

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr

    tmp = tempfile.mkdtemp()
    try:
        rec = _Recorder()
        _stub_world(rec, tmp)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None
        rk = _run_kwargs(tmp)
        parent = sr.parent_dir_for(ar, rk)

        # A finished run, then simulate a walltime kill: seeds 0-1 complete,
        # seed 2 with only 3 classifier rows, seeds 3-4 never started.
        sr.run_split(ar, rk, list(range(ar.N_GEN_SEEDS)), load_checkpoint=False)
        sr.merge_shards(ar, parent)
        legacy = os.path.join(parent, ar.PER_SEED_FILE)
        rows = sr.read_legacy_rows(legacy)
        keep = [(g, c, m, v) for g, c, m, v in rows
                if g in (0, 1) or (g == 2 and (c == "" or int(c) < 3))]
        with open(legacy, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["gen_seed", "clf_seed", "metric", "value"])
            w.writerows(keep)
        for k in range(ar.N_GEN_SEEDS):
            for f in (sr.SEED_METRICS_FMT.format(k), sr.SHARD_FMT.format(k)):
                fp = os.path.join(parent, f)
                if os.path.exists(fp):
                    os.remove(fp)

        state = sr.scan_state(ar, parent)
        assert state["done"] == [0, 1], f"scan done={state['done']}"
        assert state.get("partial", {}).get(2) == 3, \
            f"seed 2 should be partial with 3 clf seeds: {state.get('partial')}"
        assert state["missing"] == [2, 3, 4], f"scan missing={state['missing']}"

        # --- resume ONLY seed 2, so no merge runs -------------------------
        sr.run_split(ar, rk, [2], load_checkpoint=True)

        rows = sr.read_legacy_rows(legacy)
        seed2 = [r for r in rows if r[0] == 2 and r[1] != ""]
        assert not seed2, (
            f"seed 2 still has {len(seed2)} stale classifier rows in "
            f"{ar.PER_SEED_FILE} after being re-run; drop_seed_rows did not "
            f"prune them, and until the merge runs anything reading this file "
            f"sees a seed with the wrong number of rows")
        # seeds 0 and 1 must be untouched by the pruning
        for k in (0, 1):
            n = len({r[1] for r in rows if r[0] == k and r[1] != ""})
            assert n == ar.N_CLF_SEEDS, \
                f"pruning damaged seed {k}: {n}/{ar.N_CLF_SEEDS} clf seeds"

        # --- finish and merge: every seed exactly N_CLF_SEEDS rows ---------
        sr.run_split(ar, rk, [3, 4], load_checkpoint=True)
        sr.merge_shards(ar, parent)
        rows = sr.read_legacy_rows(legacy)
        counts = Counter((g, m) for g, c, m, _v in rows if c != "")
        bad = {k: n for k, n in counts.items() if n != ar.N_CLF_SEEDS}
        assert not bad, (
            f"duplicate or missing classifier rows after merge: "
            f"{sorted(bad.items())[:6]} (expected {ar.N_CLF_SEEDS} each)")
        assert sorted({g for g, _c, _m, _v in rows}) == list(range(ar.N_GEN_SEEDS)), \
            "the legacy-imported seeds were lost by the merge"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuses_to_train_over_an_existing_checkpoint():
    """Training over model_ema.pt would mix seeds from two different models."""
    import shutil
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr
    tmp = tempfile.mkdtemp()
    try:
        rec = _Recorder()
        _stub_world(rec, tmp)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None
        rk = _run_kwargs(tmp)

        # a finished seed 0 from an earlier model
        sr.run_split(ar, rk, [0], load_checkpoint=False)
        parent = sr.parent_dir_for(ar, rk)
        assert os.path.exists(os.path.join(parent, "model_ema.pt"))
        n_before = rec.seq().count("train")

        # asking to train again must abort, not silently overwrite
        try:
            sr.run_split(ar, rk, [1], load_checkpoint=False)
        except SystemExit as exc:
            msg = str(exc)
            for want in ("refusing to train", "already exists",
                         "--load_checkpoint", "--force_retrain"):
                assert want in msg, f"error message lacks {want!r}:\n{msg}"
        else:
            raise AssertionError("it trained over the existing checkpoint")
        assert rec.seq().count("train") == n_before, "it trained anyway"

        # --force_retrain is the deliberate escape hatch
        sr.run_split(ar, rk, [1], load_checkpoint=False, force_retrain=True)
        assert rec.seq().count("train") == n_before + 1, \
            "--force_retrain did not train"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_checkpoint_refuses_a_mismatched_cat_input():
    """A "log" and a "prob" checkpoint have the same tensor shapes; loading the
    wrong one must abort. The record is written after training, a legacy
    checkpoint (no record) counts as "log", and "prob" runs live in *_catfix."""
    import json
    import shutil
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr
    tmp = tempfile.mkdtemp()
    try:
        rec = _Recorder()
        _stub_world(rec, tmp)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None

        rk_prob = {**_run_kwargs(tmp), "cat_input": "prob"}
        parent = sr.parent_dir_for(ar, rk_prob)
        assert os.path.basename(os.path.dirname(parent)) == "ablation_adaptive_entmax15_catfix"
        sr.run_split(ar, rk_prob, [0], load_checkpoint=False)
        with open(os.path.join(parent, ar.DENOISER_CONFIG_FILE)) as fh:
            assert json.load(fh)["cat_input"] == "prob", "training did not record cat_input"
        train_kw = [kw for name, kw in rec.calls if name == "train"][-1]
        assert train_kw["graph_params"]["cat_input"] == "prob", "cat_input not passed to train"
        sr.run_split(ar, rk_prob, [1], load_checkpoint=True)   # matching: loads fine

        # the same checkpoint requested as "log" (e.g. copied by hand into the
        # legacy tree) must be refused
        rk_log = {**_run_kwargs(tmp), "cat_input": "log"}
        legacy = sr.parent_dir_for(ar, rk_log)
        shutil.copytree(parent, legacy)
        try:
            sr.run_split(ar, rk_log, [2], load_checkpoint=True)
        except SystemExit as exc:
            assert "refusing to load" in str(exc) and "'prob'" in str(exc), f"unclear message: {exc}"
        else:
            raise AssertionError("a 'prob' checkpoint was loaded as 'log'")

        # a legacy checkpoint (no record) is "log": loading it as "prob" is refused
        os.remove(os.path.join(legacy, ar.DENOISER_CONFIG_FILE))
        shutil.rmtree(parent)
        shutil.copytree(legacy, parent)
        try:
            sr.run_split(ar, rk_prob, [2], load_checkpoint=True)
        except SystemExit as exc:
            assert "refusing to load" in str(exc), f"unclear message: {exc}"
        else:
            raise AssertionError("a legacy ('log') checkpoint was loaded as 'prob'")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_checkpoint_refuses_a_mismatched_topk_mode_and_drops_partial_rows():
    """The record must also match the tree's topk_mode; a matching load first
    drops the seed's partial rows (e.g. DCR/MIA rows of a job that died in
    its first CatBoost evaluation) before re-running it."""
    import csv
    import json
    import shutil
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr
    tmp = tempfile.mkdtemp()
    try:
        rec = _Recorder()
        _stub_world(rec, tmp)
        ar._load_raw_real = lambda p: (None, None, [], None)
        ar._load_raw_syn = lambda *a, **k: None
        rk = {**_run_kwargs(tmp), "cat_input": "prob"}          # adaptive_entmax15
        parent = sr.parent_dir_for(ar, rk)
        os.makedirs(parent)
        open(os.path.join(parent, "model_ema.pt"), "w").close()
        per_seed = os.path.join(parent, ar.PER_SEED_FILE)
        with open(per_seed, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gen_seed", "clf_seed", "metric", "value"])
            for m in ("dcr_tabddpm_mean", "dcr_baseline", "wasserstein_mean", "mia_auc"):
                w.writerow([0, "", m, 0.1])                      # gen-level only, no clf rows

        def write_record(mode):
            with open(os.path.join(parent, ar.DENOISER_CONFIG_FILE), "w") as fh:
                json.dump({"cat_input": "prob", "topk_mode": mode}, fh)

        write_record("fixed")                                    # wrong rule for this tree
        try:
            sr.run_split(ar, rk, [0], load_checkpoint=True)
        except SystemExit as exc:
            assert "refusing to load" in str(exc) and "topk_mode" in str(exc), f"unclear message: {exc}"
        else:
            raise AssertionError("a fixed-mode checkpoint was loaded for an entmax run")
        assert "sample" not in rec.seq(), "it sampled before refusing"

        write_record("adaptive_entmax15")
        sr.run_split(ar, rk, [0], load_checkpoint=True)
        assert "train" not in rec.seq(), "it retrained instead of loading"
        left = [r for r in csv.reader(open(per_seed)) if r and r[0] == "0"]
        assert not left, f"partial gen-seed-0 rows were not dropped: {left}"
        shard = os.path.join(parent, sr.SHARD_FMT.format(0))
        rows = list(csv.DictReader(open(shard)))
        assert sum(r["metric"] == "mia_auc" for r in rows) == 1, "gen-level rows duplicated"
        assert any(r["clf_seed"] not in ("", None) for r in rows), "no classifier rows for seed 0"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_submit_split_catfix_job_loads_checkpoint_from_catfix_tree_from_any_cwd():
    """A jobs_catfix/ script: *_catfix tree, --cat_input prob on every call,
    one --load_checkpoint job per missing seed (0-4) with no training job and no
    inter-seed dependency, merge after all, every --wrap cd's to the repo root
    -- and submit_split.sh itself is called from an unrelated directory. A
    legacy job keeps its exact command lines (no --cat_input, legacy tree)."""
    import json as _json
    import shutil
    import subprocess
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr

    tmp = tempfile.mkdtemp()
    try:
        rk = dict(_run_kwargs(tmp), dataset_name="default", exp_root=os.path.join(tmp, "transfer_runs"),
                  gnn_type="gatv2", n_layers=3, d_model=128, top_k=5, n_heads=8,
                  topk_mode="adaptive_entmax15", cat_input="prob")
        parent = sr.parent_dir_for(ar, rk)
        assert parent.endswith(os.path.join("ablation_adaptive_entmax15_catfix",
                                            "gnn_gatv2_layers3_dim128_topk5_heads8_noattn")), parent
        os.makedirs(parent)
        open(os.path.join(parent, "model_ema.pt"), "w").close()
        with open(os.path.join(parent, ar.DENOISER_CONFIG_FILE), "w") as fh:
            _json.dump({"cat_input": "prob", "topk_mode": "adaptive_entmax15"}, fh)

        env = dict(os.environ, HPC_ROOT=tmp, REPO_ROOT=REPO, PYTHON_BIN=sys.executable,
                   TRANSFER_ROOT=os.path.join(tmp, "transfer_runs"), JOBS_OUT=os.path.join(tmp, "jobs_out"))
        for k in ("GRES", "TIME", "TRAIN_ONLY", "MEM"):
            env.pop(k, None)
        elsewhere = os.path.join(tmp, "elsewhere")
        os.makedirs(elsewhere)

        def run(job):
            r = subprocess.run(["bash", os.path.join(HERE, "submit_split.sh"), job],
                               capture_output=True, text=True, env=env, cwd=elsewhere)
            out = r.stdout + r.stderr
            assert r.returncode == 0, f"{job}: exited {r.returncode}\n{out}"
            return out, [l for l in out.splitlines() if l.startswith("sbatch")]

        out, lines = run("jobs_catfix/default_entmax_catfix.sh")
        assert f"results    : {parent}" in out, "scan did not resolve the *_catfix tree"
        seeds = [l for l in lines if "--gen_seed_ids" in l]
        ids = sorted(re.search(r"--gen_seed_ids\\?\s+(\d+)", l).group(1) for l in seeds)
        assert ids == ["0", "1", "2", "3", "4"], f"wrong seeds: {ids}"
        assert not [l for l in lines if "--train_only" in l], "a train-only job was submitted"
        for l in seeds:
            assert "--load_checkpoint" in l, f"seed job would retrain:\n{l[:300]}"
            assert "--dependency" not in l, f"seed jobs must be independent:\n{l[:300]}"
        merge = [l for l in lines if "--merge_only" in l]
        assert len(merge) == 1 and merge[0].count("DRYRUN") == 5, f"merge must wait for all 5:\n{merge}"
        q = REPO.replace(" ", "\\ ")
        for l in lines:
            assert "--cat_input\\ prob" in l, f"call without --cat_input prob:\n{l[:300]}"
            assert f"--wrap cd\\ {q}\\ \\&\\&\\ " in l, f"--wrap does not cd to the repo root:\n{l[:300]}"
            assert "--mem=24G" in l or "--merge_only" in l, l[:200]
            assert "--gres=gpu:a100:1" in l and ("--time=24:00:00" in l or "--merge_only" in l), l[:200]

        # a legacy job: unchanged command lines, legacy tree
        out, lines = run("default_entmax.sh")
        assert "ablation_adaptive_entmax15/gnn_" in out and "_catfix" not in out.split("results")[1].split("\n")[0]
        assert lines and all("--cat_input" not in l for l in lines), "legacy job gained --cat_input"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_submit_split_single_job_runs_missing_seeds_then_merges_in_one_job():
    """--single_job: exactly one sbatch, from the checkpoint, every missing seed
    in sequence then the merge in the same --wrap; seed-job resources; cd to
    the repo root; --cat_input from the job script. Seeds already done are
    left out; without a checkpoint it refuses instead of training."""
    import json as _json
    import shutil
    import subprocess
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr

    tmp = tempfile.mkdtemp()
    try:
        env = dict(os.environ, HPC_ROOT=tmp, REPO_ROOT=REPO, PYTHON_BIN=sys.executable,
                   TRANSFER_ROOT=os.path.join(tmp, "transfer_runs"), JOBS_OUT=os.path.join(tmp, "jobs_out"))
        for k in ("GRES", "TIME", "TRAIN_ONLY", "MEM"):
            env.pop(k, None)
        elsewhere = os.path.join(tmp, "elsewhere")
        os.makedirs(elsewhere)

        def run(*argv):
            r = subprocess.run(["bash", os.path.join(HERE, "submit_split.sh"), *argv],
                               capture_output=True, text=True, env=env, cwd=elsewhere)
            out = r.stdout + r.stderr
            return r.returncode, out, [l for l in out.splitlines() if l.startswith("sbatch")]

        job = "jobs_catfix/insurance_fixed_catfix.sh"
        rk = dict(_run_kwargs(tmp), dataset_name="insurance", exp_root=os.path.join(tmp, "transfer_runs"),
                  gnn_type="gatv2", n_layers=2, d_model=64, top_k=5, n_heads=4,
                  topk_mode="fixed", cat_input="prob")
        parent = sr.parent_dir_for(ar, rk)
        assert parent.endswith(os.path.join("ablation_catfix", "gnn_gatv2_layers2_dim64_topk5_heads4_noattn"))

        # no checkpoint: refuse, submit nothing
        code, out, lines = run(job, "--single_job")
        assert code != 0 and "needs an existing checkpoint" in out and not lines, out

        os.makedirs(parent)
        open(os.path.join(parent, "model_ema.pt"), "w").close()
        with open(os.path.join(parent, ar.DENOISER_CONFIG_FILE), "w") as fh:
            _json.dump({"cat_input": "prob", "topk_mode": "fixed"}, fh)

        code, out, lines = run(job, "--single_job")
        assert code == 0, out
        assert len(lines) == 1, f"expected exactly one sbatch, got {len(lines)}:\n" + "\n".join(lines)
        l = lines[0]
        q = REPO.replace(" ", "\\ ")
        assert f"--wrap cd\\ {q}\\ \\&\\&\\ " in l, f"--wrap does not cd to the repo root:\n{l}"
        # the dry run prints %q-quoted args (backslash before spaces, commas,
        # &); the payload sbatch receives is the unescaped string
        payload = l.split("--wrap ", 1)[1].replace("\\", "")
        cd_part, seed_part, merge_part = payload.split("&&")
        assert cd_part.strip() == f"cd {REPO}", cd_part
        assert re.search(r"--gen_seed_ids 0,1,2,3,4 ", seed_part), f"not all missing seeds:\n{seed_part}"
        assert "--load_checkpoint" in seed_part and "--merge_only" not in seed_part, seed_part
        assert "--merge_only" in merge_part, "the merge must follow the seeds in the same job"
        assert payload.count("--cat_input prob") == 2, "both calls need --cat_input prob"
        assert "--train_only" not in l and "--dependency" not in l
        for want in ("--mem=24G", "--gres=gpu:a100:1", "--time=24:00:00", "--partition=gpu",
                     "--job-name=insurance_fixed_all"):
            assert want in l, f"missing {want}:\n{l}"

        # seed 0 already done -> only 1..4 in the job
        with open(os.path.join(parent, sr.SEED_METRICS_FMT.format(0)), "w") as fh:
            _json.dump({"gen_seed": 0, "gen_metrics": {"mia_auc": [0.5]},
                        "test_metrics": {"score": [0.5] * ar.N_CLF_SEEDS}}, fh)
        code, out, lines = run(job, "--single_job")
        assert code == 0 and len(lines) == 1, out
        assert "--gen_seed_ids 1,2,3,4 " in lines[0].replace("\\", ""), lines[0]

        # the default (no flag) is still the per-seed chain
        code, out, lines = run(job)
        assert code == 0 and len([x for x in lines if "--gen_seed_ids" in x]) == 4, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_submit_split(job_script: str, tmp: str, dataset: str = None,
                      rk_over: dict = None):
    """Drive submit_split.sh against a throwaway tree; return its output."""
    import subprocess
    env = dict(os.environ, HPC_ROOT=tmp, REPO_ROOT=REPO,
               PYTHON_BIN=sys.executable,
               TRANSFER_ROOT=os.path.join(tmp, "transfer_runs"))
    env.pop("GRES", None)
    env.pop("TIME", None)
    env.pop("TRAIN_ONLY", None)
    os.makedirs(os.path.join(tmp, "jobs_out"), exist_ok=True)
    r = subprocess.run(["bash", os.path.join(HERE, "submit_split.sh"), job_script],
                       capture_output=True, text=True, env=env, cwd=REPO)
    return r, r.stdout + r.stderr


def test_fb_comments_uses_a_train_only_chain_at_48h():
    """fb-comments, no checkpoint: 1 train-only job + 5 seed jobs at 48 h.

    One fb-comments generation seed is 14.4 - 35.3 h on its own, so train plus
    a seed in the same job could overrun even the 48 h maximum. The train-only
    chain puts training in its own job and every seed in its own, all after it
    and independent of each other.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        r, out = _run_submit_split("fb-comments_entmax.sh", tmp)
        assert r.returncode == 0, f"exited {r.returncode}\n{out}"

        train = [l for l in out.splitlines() if "--train_only" in l and l.startswith("sbatch")]
        seeds = [l for l in out.splitlines() if "--gen_seed_ids" in l]
        merge = [l for l in out.splitlines() if "--merge_only" in l]

        assert len(train) == 1, f"expected 1 train-only job, got {len(train)}"
        assert "--gen_seed_ids" not in train[0], \
            "the train-only job must run no generation seed"
        assert "--dependency" not in train[0], "the train job depends on nothing"

        assert len(seeds) == 5, f"expected 5 seed jobs, got {len(seeds)}"
        ids = sorted(re.search(r"--gen_seed_ids\\?\s+(\d+)", l).group(1) for l in seeds)
        assert ids == ["0", "1", "2", "3", "4"], f"seeds submitted: {ids}"
        for l in seeds:
            assert "--load_checkpoint" in l, f"seed job without --load_checkpoint:\n{l[:200]}"
            assert "--dependency=afterok:" in l, \
                "every seed job must wait for the train job"

        for l in train + seeds:
            assert "--time=48:00:00" in l, (
                "fb-comments jobs must get the 48 h maximum:\n" + l[:200])
        assert len(merge) == 1 and "--time=00:30:00" in merge[0], \
            "the merge job stays short"
        assert merge[0].count("DRYRUN") == 5, \
            "merge must depend on all five seed jobs"
        assert "TRAIN-ONLY chain" in out, "it did not report the train-only chain"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_other_datasets_stay_at_24h_and_train_with_seed0():
    """Only fb-comments changes shape; miniboone keeps the 24 h chain."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        r, out = _run_submit_split("miniboone_entmax.sh", tmp)
        assert r.returncode == 0, f"exited {r.returncode}\n{out}"
        assert not [l for l in out.splitlines() if "--train_only" in l], \
            "miniboone must not use the train-only chain by default"
        seeds = [l for l in out.splitlines() if "--gen_seed_ids" in l]
        assert len(seeds) == 5, f"expected 5 seed jobs, got {len(seeds)}"
        first = [l for l in seeds if re.search(r"--gen_seed_ids\\?\s+0\b", l)]
        assert len(first) == 1 and "--load_checkpoint" not in first[0], \
            "the first job should train and run seed 0"
        for l in seeds:
            assert "--time=24:00:00" in l, "miniboone stays at 24 h:\n" + l[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_submit_split_skips_training_when_checkpoint_exists():
    """submit_split.sh: checkpoint present + seed 0 done -> exactly 4 jobs,
    all --load_checkpoint, none depending on another, and no training job."""
    import json as _json
    import shutil
    import tempfile

    ar = _import_runner()
    sys.path.insert(0, HERE)
    import split_runner as sr

    tmp = tempfile.mkdtemp()
    try:
        rk = dict(_run_kwargs(tmp))
        rk["dataset_name"] = "miniboone"
        rk["exp_root"] = os.path.join(tmp, "transfer_runs")
        rk.update(gnn_type="gatv2", n_layers=3, d_model=128, top_k=5, n_heads=8)
        parent = sr.parent_dir_for(ar, rk)
        os.makedirs(parent, exist_ok=True)
        os.makedirs(os.path.join(tmp, "jobs_out"), exist_ok=True)
        open(os.path.join(parent, "model_ema.pt"), "w").close()
        with open(os.path.join(parent, sr.SEED_METRICS_FMT.format(0)), "w") as fh:
            _json.dump({"gen_seed": 0, "gen_metrics": {"mia_auc": [0.5]},
                        "test_metrics": {"score": [0.5] * ar.N_CLF_SEEDS}}, fh)

        r, out = _run_submit_split("miniboone_entmax.sh", tmp)
        assert r.returncode == 0, f"submit_split.sh exited {r.returncode}\n{out}"

        seed_jobs = [l for l in out.splitlines() if "--gen_seed_ids" in l]
        assert len(seed_jobs) == 4, (
            f"expected 4 seed jobs (1..4), got {len(seed_jobs)}:\n"
            + "\n".join(l[:120] for l in seed_jobs))
        for l in seed_jobs:
            assert "--load_checkpoint" in l, f"job without --load_checkpoint:\n{l[:200]}"
            assert "--dependency" not in l, (
                f"seed jobs must be independent when the checkpoint exists:\n{l[:200]}")
        # the --wrap payload is %q-quoted, so the value reads "\ 1"
        ids = sorted(re.search(r"--gen_seed_ids\\?\s+(\d+)", l).group(1)
                     for l in seed_jobs)
        assert ids == ["1", "2", "3", "4"], f"wrong seeds submitted: {ids}"
        assert "checkpoint present: no training job" in out, \
            "it did not report skipping training"
        merge = [l for l in out.splitlines() if "--merge_only" in l]
        assert len(merge) == 1 and merge[0].count("DRYRUN") == 4, \
            f"merge must depend on all four seed jobs:\n{merge}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc}")
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
