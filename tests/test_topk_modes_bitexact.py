"""
Bit-exactness of the topk_mode paths against the code that produced results.

  fixed               vs  jenks-ref^   (tab_ddpm before adaptive top-k existed)
  fixed / sparsemax / entmax15  vs  density-ref   (unchanged by the plain-Jenks
                                   routing: these three must stay bit-identical)
  adaptive_jenks      vs  density-ref  INTENTIONALLY DIFFERENT: adaptive_jenks is
                                   now plain Jenks (every 2-class split accepted),
                                   where density-ref applied an acceptance test on
                                   the strong-class mean. The test reports how many
                                   comparisons differ instead of asserting equality.

The reference ``tab_ddpm`` package (plus ``adaptive_adjacency.py`` where it
exists) is read straight from git at that revision and imported under its own
package name, so nothing is vendored and nothing is patched. A reference model
and a current model are built from the same state_dict and trained side by side
on identical inputs. No tolerances anywhere: every forward output, every
gradient and every parameter after every Adam step must be ``torch.equal``, and
for Jenks the two ``topk_stats.csv`` logs must be byte-identical.

Two levels:
  * DynamicAdjacency alone, across N / top_k / warm-up (incl. out-of-range k).
  * The full GraphAwareDenoiser, with the three dataset shapes of this round
    (wilt, abalone, gesture) and the GNN backbones the runs use.

Plus behavioural checks for adaptive_sparsemax / adaptive_entmax15 (these have
no reference -- they are new): the support is exactly entmax's, the weights
are exactly sigmoid(phi) on it, gradients reach only the kept edges, warm-up is
the Jenks warm-up bit for bit, and logging keeps the same columns.

Run:  python tests/test_topk_modes_bitexact.py        (prints a pass/fail table)
      pytest tests/test_topk_modes_bitexact.py -v
"""
from __future__ import annotations

import contextlib
import csv
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tab_ddpm import topk_logging as new_logging  # noqa: E402
from tab_ddpm.graph_builder import DynamicAdjacency as NewDA  # noqa: E402
from tab_ddpm.graph_denoiser import GraphAwareDenoiser as NewModel  # noqa: E402

FIXED_REV = "jenks-ref^"
DENSITY_REV = "density-ref"
T = 1000  # ablation_runner.ABLATION_NUM_TIMESTEPS

# (name, d_num, cat_sizes, num_classes, is_y_cond) -- as GraphAwareDenoiser sees
# each dataset: abalone is regression, so y rides along as an extra numeric.
SHAPES = [
    ("wilt",    5,  [],  2, True),
    ("abalone", 8,  [3], 0, False),
    ("gesture", 32, [],  5, True),
]
GNN_TYPES = ["gcn", "gat", "gatv2", "gin"]


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def _git_show(rev: str, path: str):
    proc = subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"],
                          capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def _git_ls(rev: str, path: str):
    out = subprocess.run(["git", "-C", REPO, "ls-tree", "--name-only", f"{rev}:{path}"],
                         capture_output=True, text=True, check=True).stdout
    return [name for name in out.split() if name.endswith(".py")]


_REF_CACHE: dict = {}


def load_reference(rev: str, pkg_name: str):
    """Import tab_ddpm at ``rev`` as package ``pkg_name``.

    The reference graph_builder does ``from adaptive_adjacency import ...``
    (absolute), so the reference adaptive_adjacency is swapped into
    sys.modules for the duration of the import and the current one restored
    afterwards -- each package keeps its own function objects.
    """
    if pkg_name in _REF_CACHE:
        return _REF_CACHE[pkg_name]
    base = tempfile.mkdtemp(prefix=f"{pkg_name}_")
    pkg_dir = os.path.join(base, pkg_name)
    os.makedirs(pkg_dir)
    for name in _git_ls(rev, "tab_ddpm"):
        with open(os.path.join(pkg_dir, name), "wb") as fh:
            fh.write(_git_show(rev, f"tab_ddpm/{name}"))

    saved_aa = sys.modules.pop("adaptive_adjacency", None)
    aa_src = _git_show(rev, "adaptive_adjacency.py")
    try:
        if aa_src is not None:
            aa_path = os.path.join(base, "adaptive_adjacency.py")
            with open(aa_path, "wb") as fh:
                fh.write(aa_src)
            spec = importlib.util.spec_from_file_location("adaptive_adjacency", aa_path)
            ref_aa = importlib.util.module_from_spec(spec)
            sys.modules["adaptive_adjacency"] = ref_aa
            spec.loader.exec_module(ref_aa)
        sys.path.insert(0, base)
        pkg = importlib.import_module(pkg_name)
    finally:
        if base in sys.path:
            sys.path.remove(base)
        if saved_aa is not None:
            sys.modules["adaptive_adjacency"] = saved_aa
        else:
            sys.modules.pop("adaptive_adjacency", None)
    _REF_CACHE[pkg_name] = pkg
    return pkg


def ref_fixed():
    return load_reference(FIXED_REV, "ref_fixed_tab_ddpm")


def ref_density():
    return load_reference(DENSITY_REV, "ref_density_tab_ddpm")


REFS = {"pre": ref_fixed, "density": ref_density}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def float32_default():
    # test_gnn_correctness.py sets float64 globally at import; training is fp32.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


@contextlib.contextmanager
def logging_to(module, directory):
    module.set_context(output_dir=directory, log_every=1, tag="")
    try:
        yield
    finally:
        module._state["output_dir"] = None


def _assert_same_params(a, b, what):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys(), f"{what}: parameter names differ"
    for name in pa:
        assert torch.equal(pa[name], pb[name]), f"{what}: param {name} differs"


def _assert_same_grads(a, b, what):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    for name in pa:
        ga, gb = pa[name].grad, pb[name].grad
        assert (ga is None) == (gb is None), f"{what}: grad presence differs for {name}"
        if ga is not None:
            assert torch.equal(ga, gb), f"{what}: grad of {name} differs"


def _read(path):
    if not os.path.exists(path):
        return b""
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Level 1: DynamicAdjacency alone
# ---------------------------------------------------------------------------

def _da_pair(ref_key, mode, n, d, top_k, warmup, seed):
    torch.manual_seed(seed)
    pkg = REFS[ref_key]()
    if ref_key == "pre":        # pre-adaptive DynamicAdjacency: (n, d, top_k) only
        ref = pkg.graph_builder.DynamicAdjacency(n, d, top_k)
    else:
        ref = pkg.graph_builder.DynamicAdjacency(
            n, d, top_k, topk_mode=mode, k_min=1, topk_warmup_steps=warmup, num_timesteps=T)
    new = (NewDA(n, d, top_k, topk_mode="fixed") if mode == "fixed" else
           NewDA(n, d, top_k, topk_mode=mode, k_min=1, topk_warmup_steps=warmup, num_timesteps=T))
    new.load_state_dict(ref.state_dict())
    return ref, new


def _call(mod, x, t, legacy):
    return mod(x) if legacy else mod(x, timesteps=t, block_idx=0)


def run_da_diff_case(ref_key, mode, n, top_k, warmup, seed=0, d=16, B=32, steps=6):
    """Forward-only: count the comparisons whose A differs (the two rules take
    different budgets, so stepping optimizers in lockstep would be meaningless)."""
    with float32_default():
        ref, new = _da_pair(ref_key, mode, n, d, top_k, warmup, seed)
        g = torch.Generator().manual_seed(1000 + seed)
        differing = rows = 0
        for _ in range(steps):
            x = torch.randn(B, n, d, generator=g)
            t = torch.randint(0, T, (B,), generator=g)
            a_r, a_n = ref(x, timesteps=t, block_idx=0), new(x, timesteps=t, block_idx=0)
            if not torch.equal(a_r, a_n):
                differing += 1
                rows += int((a_r != a_n).any(dim=-1).sum())
        return differing, rows, steps


def run_da_case(ref_key, mode, n, top_k, warmup, seed=0, d=16, B=32, steps=8):
    legacy = ref_key == "pre"
    with float32_default():
        ref, new = _da_pair(ref_key, mode, n, d, top_k, warmup, seed)
        opt_r = torch.optim.Adam(ref.parameters(), lr=1e-2)
        opt_n = torch.optim.Adam(new.parameters(), lr=1e-2)
        g = torch.Generator().manual_seed(1000 + seed)
        what = f"DA {mode} N={n} k={top_k} warm={warmup}"
        for step in range(steps):
            x = torch.randn(B, n, d, generator=g)
            t = torch.randint(0, T, (B,), generator=g)
            w = torch.randn(B, n, n, generator=g)
            a_r, a_n = _call(ref, x, t, legacy), _call(new, x, t, legacy)
            assert torch.equal(a_r, a_n), f"{what}: A differs at train step {step}"
            opt_r.zero_grad()
            opt_n.zero_grad()
            (a_r * w).sum().backward()
            (a_n * w).sum().backward()
            _assert_same_grads(ref, new, f"{what} step {step}")
            opt_r.step()
            opt_n.step()
            _assert_same_params(ref, new, f"{what} after step {step}")
        ref.eval()
        new.eval()
        with torch.no_grad():
            for step in range(3):
                x = torch.randn(B, n, d, generator=g)
                t = torch.full((B,), int(torch.randint(0, T, (1,), generator=g)))
                assert torch.equal(_call(ref, x, t, legacy), _call(new, x, t, legacy)), \
                    f"{what}: A differs at eval step {step}"


def da_cases():
    """(ref_key, mode, n, top_k, warmup) that must be bit-identical."""
    cases = []
    for n in (5, 9, 32):
        for top_k in (0, 3, 5, n - 1, n + 2):
            cases.append(("pre", "fixed", n, top_k, 0))
            cases.append(("density", "fixed", n, top_k, 0))
            for mode in ENTMAX_MODES:
                for warmup in (0, 4):
                    cases.append(("density", mode, n, top_k, warmup))
    return cases


def da_diff_cases():
    """adaptive_jenks vs density-ref: intentionally different (plain Jenks)."""
    return [("density", "adaptive_jenks", n, top_k, 0)
            for n in (5, 9, 32) for top_k in (0, 3, 5, n - 1, n + 2)]


# ---------------------------------------------------------------------------
# Level 2: full GraphAwareDenoiser
# ---------------------------------------------------------------------------

def _model_kwargs(shape, gnn_type, top_k):
    _, d_num, cat_sizes, num_classes, is_y_cond = shape
    return dict(d_in=d_num + sum(cat_sizes), num_classes=num_classes,
                is_y_cond=is_y_cond, d_num=d_num, cat_sizes=cat_sizes,
                d_model=16, n_layers=2, n_heads=4, graph_mode="dynamic",
                top_k=top_k, gnn_type=gnn_type, dropout=0.0, use_attention=False)


def _batch(shape, g, B, eval_t=None):
    _, d_num, cat_sizes, num_classes, _ = shape
    parts = [torch.randn(B, d_num, generator=g)]
    for K in cat_sizes:
        idx = torch.randint(0, K, (B,), generator=g)
        parts.append(torch.nn.functional.one_hot(idx, K).float())
    x = torch.cat(parts, dim=1)
    t = (torch.randint(0, T, (B,), generator=g) if eval_t is None
         else torch.full((B,), eval_t))
    y = (torch.randint(0, num_classes, (B,), generator=g) if num_classes > 0
         else torch.randn(B, generator=g))
    return x, t, y


def run_model_diff_case(ref_key, mode, shape, gnn_type, top_k, warmup, seed=0, B=64, steps=4):
    """Forward-only: how many outputs differ between the new rule and the reference."""
    kw = _model_kwargs(shape, gnn_type, top_k)
    with float32_default():
        torch.manual_seed(seed)
        extra = dict(topk_mode=mode, k_min=1, topk_warmup_steps=warmup, num_timesteps=T)
        ref = REFS[ref_key]().GraphAwareDenoiser(**kw, **extra)
        new = NewModel(**kw, **extra)
        new.load_state_dict(ref.state_dict())
        g = torch.Generator().manual_seed(2000 + seed)
        differing = 0
        with torch.no_grad():
            for _ in range(steps):
                x, t, y = _batch(shape, g, B)
                if not torch.equal(ref(x, t, y), new(x, t, y)):
                    differing += 1
        return differing, steps


def run_model_case(ref_key, mode, shape, gnn_type, top_k, warmup, seed=0, B=64, steps=8):
    kw = _model_kwargs(shape, gnn_type, top_k)
    what = f"model {ref_key}/{mode} {shape[0]} {gnn_type} k={top_k} warm={warmup}"
    log_dir = tempfile.mkdtemp(prefix="topk_log_")
    try:
        with float32_default():
            torch.manual_seed(seed)
            ref_pkg = REFS[ref_key]()
            if ref_key == "pre":
                ref = ref_pkg.GraphAwareDenoiser(**kw)
                new = NewModel(**kw, topk_mode="fixed")
            else:
                extra = dict(topk_mode=mode, k_min=1, topk_warmup_steps=warmup,
                             num_timesteps=T)
                ref = ref_pkg.GraphAwareDenoiser(**kw, **extra)
                new = NewModel(**kw, **extra)
            new.load_state_dict(ref.state_dict())

            ref_log, new_log = os.path.join(log_dir, "ref"), os.path.join(log_dir, "new")
            os.makedirs(ref_log)
            os.makedirs(new_log)
            opt_r = torch.optim.Adam(ref.parameters(), lr=1e-3)
            opt_n = torch.optim.Adam(new.parameters(), lr=1e-3)
            g = torch.Generator().manual_seed(2000 + seed)
            with logging_to(ref_pkg.topk_logging, ref_log) if ref_key != "pre" \
                    else contextlib.nullcontext(), logging_to(new_logging, new_log):
                for step in range(steps):
                    x, t, y = _batch(shape, g, B)
                    out_r, out_n = ref(x, t, y), new(x, t, y)
                    assert torch.equal(out_r, out_n), f"{what}: output differs at step {step}"
                    opt_r.zero_grad()
                    opt_n.zero_grad()
                    (out_r ** 2).mean().backward()
                    (out_n ** 2).mean().backward()
                    _assert_same_grads(ref, new, f"{what} step {step}")
                    opt_r.step()
                    opt_n.step()
                    _assert_same_params(ref, new, f"{what} after step {step}")
                ref.eval()
                new.eval()
                with torch.no_grad():
                    for step in range(3):
                        t_val = int(torch.randint(0, T, (1,), generator=g))
                        x, t, y = _batch(shape, g, B, eval_t=t_val)
                        assert torch.equal(ref(x, t, y), new(x, t, y)), \
                            f"{what}: output differs at eval step {step}"

            if mode != "fixed":
                ref_csv = _read(os.path.join(ref_log, "topk_stats.csv"))
                new_csv = _read(os.path.join(new_log, "topk_stats.csv"))
                assert ref_csv, f"{what}: reference wrote no topk_stats.csv"
                assert ref_csv == new_csv, f"{what}: topk_stats.csv differs"
            else:
                assert not os.path.exists(os.path.join(new_log, "topk_stats.csv")), \
                    f"{what}: fixed mode must not log"
            assert not os.path.exists(os.path.join(new_log, "jenks_gate_stats.csv")), \
                f"{what}: only the Jenks path writes jenks_gate_stats.csv"
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def model_cases():
    """(ref_key, mode, shape, gnn_type, top_k, warmup) that must be bit-identical."""
    cases = []
    for shape in SHAPES:
        n = shape[1] + len(shape[2])
        top_ks = sorted({3, 5, 0, n - 1} & set(range(0, n)) | {0})
        for gnn_type in GNN_TYPES:
            for top_k in top_ks:
                cases.append(("pre", "fixed", shape, gnn_type, top_k, 0))
                cases.append(("density", "fixed", shape, gnn_type, top_k, 0))
                for mode in ENTMAX_MODES:
                    cases.append(("density", mode, shape, gnn_type, top_k, 0))
    return cases


def model_diff_cases():
    return [("density", "adaptive_jenks", shape, gnn_type, top_k, 0)
            for shape in SHAPES for gnn_type in GNN_TYPES
            for top_k in sorted({3, 5, 0} & set(range(0, shape[1] + len(shape[2]))) | {0})]


# ---------------------------------------------------------------------------
# New modes: behavioural checks (no reference exists)
# ---------------------------------------------------------------------------

ENTMAX_MODES = ("adaptive_sparsemax", "adaptive_entmax15")


def _entmax_fn(mode):
    from entmax import entmax15, sparsemax
    return sparsemax if mode == "adaptive_sparsemax" else entmax15


def check_entmax_support_and_weights(mode):
    """mask == entmax support; A == sigmoid(phi) on it, 0 off it, 1 on diag;
    gradients reach phi only through kept edges; k_min=1 floor is a no-op."""
    from adaptive_adjacency import ENTMAX_ALPHA, build_adjacency_entmax
    fn = _entmax_fn(mode)
    with float32_default():
        g = torch.Generator().manual_seed(7)
        for n in (5, 9, 32):
            for scale in (0.05, 0.5, 2.0, 10.0):
                B = 64
                phi = (torch.randn(B, n, n, generator=g) * scale).requires_grad_(True)
                eye = torch.eye(n, dtype=torch.bool).expand(B, n, n)
                A, stats = build_adjacency_entmax(phi, alpha=ENTMAX_ALPHA[mode], k_min=1)
                support = fn(phi.detach().masked_fill(eye, float("-inf")), dim=-1) > 0
                kept = A.detach().ne(0) & ~eye
                assert torch.equal(kept, support), f"{mode}: kept edges != entmax support"
                sig = torch.sigmoid(phi.detach())
                assert torch.equal(A.detach()[support], sig[support]), f"{mode}: weights != sigmoid(phi)"
                assert torch.all(A.detach()[eye] == 1.0), f"{mode}: diagonal != 1"
                assert torch.all(A.detach()[~support & ~eye] == 0.0), f"{mode}: dropped edge != 0"
                k = support.sum(-1)
                assert int(k.min()) >= 1, f"{mode}: a node has no edge"
                assert torch.equal(stats["k_per_sample"], k.float().mean(-1)), f"{mode}: k_per_sample"
                w = torch.randn(B, n, n, generator=g)
                (A * w).sum().backward()
                has_grad = phi.grad.ne(0)
                assert not torch.any(has_grad & ~support), f"{mode}: gradient leaked off the support"
                # d sigmoid = s(1-s), which is exactly 0 in fp32 once s saturates.
                assert torch.equal(has_grad, support & (w * sig * (1 - sig)).ne(0)), \
                    f"{mode}: kept edge without gradient"


def check_entmax_k_min(mode):
    from adaptive_adjacency import ENTMAX_ALPHA, entmax_support
    with float32_default():
        g = torch.Generator().manual_seed(8)
        phi = torch.randn(64, 9, 9, generator=g) * 20.0   # large scale -> tiny supports
        mask1, k1 = entmax_support(phi, ENTMAX_ALPHA[mode], k_min=1)
        mask3, k3 = entmax_support(phi, ENTMAX_ALPHA[mode], k_min=3)
        assert int(k3.min()) >= 3, f"{mode}: k_min=3 not enforced"
        assert torch.all(mask3 | ~mask1), f"{mode}: k_min must only add edges"
        assert int(k1.min()) >= 1


def check_entmax_warmup_is_jenks_warmup(mode):
    """During warm-up every adaptive mode must be the Jenks warm-up bit for bit;
    after it, the entmax rule takes over (in train and in eval)."""
    from adaptive_adjacency import ENTMAX_ALPHA, build_adjacency_entmax
    with float32_default():
        for top_k in (0, 3, 8):     # 0 and 8 (= N-1) take the dense fallback
            torch.manual_seed(3)
            jenks = NewDA(9, 16, top_k, topk_mode="adaptive_jenks", topk_warmup_steps=4, num_timesteps=T)
            ent = NewDA(9, 16, top_k, topk_mode=mode, topk_warmup_steps=4, num_timesteps=T)
            ent.load_state_dict(jenks.state_dict())
            g = torch.Generator().manual_seed(4)
            for step in range(7):
                x = torch.randn(32, 9, 16, generator=g)
                t = torch.randint(0, T, (32,), generator=g)
                a_j, a_e = jenks(x, timesteps=t), ent(x, timesteps=t)
                if step < 4:
                    assert torch.equal(a_j, a_e), f"{mode} k={top_k}: warm-up differs from Jenks at step {step}"
                else:
                    with torch.no_grad():
                        phi = torch.bmm(ent.q_proj(x), ent.k_proj(x).transpose(1, 2)) / ent.scale
                        expect, _ = build_adjacency_entmax(phi, ENTMAX_ALPHA[mode], k_min=1)
                    assert torch.equal(a_e.detach(), expect), f"{mode}: post-warm-up is not the entmax rule"
            ent.eval()
            with torch.no_grad():
                x = torch.randn(32, 9, 16, generator=g)
                phi = torch.bmm(ent.q_proj(x), ent.k_proj(x).transpose(1, 2)) / ent.scale
                expect, _ = build_adjacency_entmax(phi, ENTMAX_ALPHA[mode], k_min=1)
                assert torch.equal(ent(x, timesteps=torch.full((32,), 10)), expect), \
                    f"{mode}: eval does not use the entmax rule"


def check_entmax_logging(mode):
    """Same CSV columns; one row per occupied t-bucket in training, one row per
    call when sampling; k_mean = mean support size."""
    log_dir = tempfile.mkdtemp(prefix="topk_log_")
    try:
        with float32_default(), logging_to(new_logging, log_dir):
            torch.manual_seed(5)
            da = NewDA(9, 16, 3, topk_mode=mode, num_timesteps=T)
            g = torch.Generator().manual_seed(6)
            x = torch.randn(32, 9, 16, generator=g)
            t = torch.randint(0, T, (32,), generator=g)
            A = da(x, timesteps=t, block_idx=0)
            da.eval()
            with torch.no_grad():
                A_eval = da(x, timesteps=torch.full((32,), 950), block_idx=0)
        with open(os.path.join(log_dir, "topk_stats.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
            assert list(rows[0].keys()) == new_logging.CSV_COLUMNS, f"{mode}: CSV columns changed"
        train = [r for r in rows if r["phase"] == "train"]
        sample = [r for r in rows if r["phase"] == "sample"]
        buckets = torch.clamp((t.float() / T * 10).long(), 0, 9)
        assert len(train) == len(buckets.unique()), f"{mode}: expected one train row per t-bucket"
        assert sum(int(r["n_samples"]) for r in train) == 32
        assert len(sample) == 1 and sample[0]["t_bucket"] == "9"
        k_eval = ((A_eval != 0).sum(-1) - 1).float().mean()
        assert abs(float(sample[0]["k_mean"]) - float(k_eval)) < 1e-6, f"{mode}: k_mean != mean support"
        assert A.shape == (32, 9, 9)
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def check_plain_vs_gated_routing():
    """build_adjacency(use_gate_rule=False) is plain Jenks, unit level.

    Identical to the gated variant wherever the acceptance test would pass
    (strong-class mean > 0), different wherever it would fire AND Jenks' own k
    is not already k_min (a flat row gives k = 1 either way). Both populations
    must be non-empty, or the comparison proves nothing.
    """
    from adaptive_adjacency import build_adjacency, jenks_budget
    with float32_default():
        g = torch.Generator().manual_seed(11)
        B, F, NEG = 128, 9, -1e9
        seen_same = seen_diff = 0
        for offset in (0.0, -8.0, +8.0):
            phi = torch.randn(B, F, F, generator=g) + offset
            eye = torch.eye(F, dtype=torch.bool).expand(B, F, F)
            s_sorted, _ = phi.masked_fill(eye, NEG).sort(dim=-1, descending=True)
            k_plain, stats = jenks_budget(s_sorted[..., : F - 1], k_min=1,
                                          use_gate_rule=False, delta=None)
            k_gated, _ = jenks_budget(s_sorted[..., : F - 1], k_min=1,
                                      use_gate_rule=True, delta=None)
            fires = stats["mu_strong"] <= 0
            assert torch.equal(k_plain[~fires], k_gated[~fires]), \
                "plain and gated differ on a row the acceptance test would have passed"
            assert torch.all(k_gated[fires] == 1), "gated did not fall back to k_min where it fired"
            A_p, _ = build_adjacency(phi, k_fixed=None, k_min=1, use_gate_rule=False)
            A_g, _ = build_adjacency(phi, k_fixed=None, k_min=1, use_gate_rule=True)
            differs = fires & (k_plain != 1)
            row_diff = (A_p != A_g).any(dim=-1)
            assert torch.equal(row_diff, differs), \
                "adjacency rows differ exactly where the gate fires and Jenks k != k_min"
            seen_same += int((~fires).sum())
            seen_diff += int(differs.sum())
        assert seen_same > 0 and seen_diff > 0, \
            f"both populations must be exercised (same={seen_same}, differ={seen_diff})"


def check_jenks_gate_log():
    """adaptive_jenks writes jenks_gate_stats.csv; the other modes do not, and
    topk_stats.csv keeps its columns."""
    for mode, expect_gate in (("adaptive_jenks", True), ("adaptive_sparsemax", False),
                              ("fixed", False)):
        log_dir = tempfile.mkdtemp(prefix="gate_log_")
        try:
            with float32_default(), logging_to(new_logging, log_dir):
                torch.manual_seed(12)
                da = NewDA(9, 16, 3, topk_mode=mode, num_timesteps=T)
                g = torch.Generator().manual_seed(13)
                x = torch.randn(32, 9, 16, generator=g)
                da(x, timesteps=torch.randint(0, T, (32,), generator=g), block_idx=0)
                da.eval()
                with torch.no_grad():
                    da(x, timesteps=torch.full((32,), 950), block_idx=0)
            gate_path = os.path.join(log_dir, new_logging.GATE_CSV_NAME)
            assert os.path.exists(gate_path) == expect_gate, \
                f"{mode}: jenks_gate_stats.csv presence should be {expect_gate}"
            if expect_gate:
                with open(gate_path, newline="") as fh:
                    rows = list(csv.DictReader(fh))
                assert list(rows[0].keys()) == new_logging.GATE_CSV_COLUMNS
                assert all(0.0 <= float(r["frac_gate_off"]) <= 1.0 for r in rows)
                assert any(r["phase"] == "sample" for r in rows)
                with open(os.path.join(log_dir, "topk_stats.csv"), newline="") as fh:
                    assert list(next(csv.DictReader(fh)).keys()) == new_logging.CSV_COLUMNS, \
                        "topk_stats.csv columns must not change"
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)


def check_invalid_mode_raises():
    for bad in ("adaptive_Jenks", "sparsemax", "entmax15", ""):
        try:
            NewDA(5, 16, 3, topk_mode=bad)
        except ValueError:
            continue
        raise AssertionError(f"topk_mode={bad!r} was accepted")


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

def test_unchanged_modes_dynamic_adjacency_bitexact():
    for case in da_cases():
        run_da_case(*case)


def test_unchanged_modes_full_model_bitexact():
    for case in model_cases():
        run_model_case(*case)


def test_plain_jenks_differs_from_density_ref():
    """The intended change: adaptive_jenks is no longer the gated variant."""
    total = sum(run_da_diff_case(*case)[0] for case in da_diff_cases())
    assert total > 0, "adaptive_jenks is still identical to density-ref: routing did not take effect"
    total_model = sum(run_model_diff_case(*case)[0] for case in model_diff_cases())
    assert total_model > 0


def test_plain_jenks_routing_unit():
    check_plain_vs_gated_routing()
    check_jenks_gate_log()


def test_entmax_modes():
    for mode in ENTMAX_MODES:
        check_entmax_support_and_weights(mode)
        check_entmax_k_min(mode)
        check_entmax_warmup_is_jenks_warmup(mode)
        check_entmax_logging(mode)
    check_invalid_mode_raises()


# ---------------------------------------------------------------------------
# Script entry point: pass/fail table
# ---------------------------------------------------------------------------

def _run(label, fn, *args):
    try:
        fn(*args)
        return True, ""
    except AssertionError as exc:
        return False, str(exc)


def main() -> int:
    import entmax
    print(f"torch {torch.__version__}   entmax {getattr(entmax, '__version__', '?')}   "
          f"device cpu   default dtype float32 (forced)")
    for label, rev in (("pre-adaptive", FIXED_REV), ("density", DENSITY_REV)):
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", f"{rev}^{{commit}}"],
                                capture_output=True, text=True).stdout.strip()
        print(f"{label} reference : {rev} ({commit})")
    print()

    groups = {}
    for case in da_cases():
        groups.setdefault(("DynamicAdjacency", case[0], case[1]), []).append(_run("", run_da_case, *case))
    for case in model_cases():
        groups.setdefault(("GraphAwareDenoiser", case[0], case[1]), []).append(_run("", run_model_case, *case))

    all_ok = True
    print("  must be BIT-IDENTICAL to the reference")
    print(f"  {'level':20s} {'reference':10s} {'mode':19s} {'cases':>6s} {'passed':>7s}  result")
    for (level, ref_key, mode), results in groups.items():
        n_ok = sum(ok for ok, _ in results)
        ok = n_ok == len(results)
        all_ok &= ok
        print(f"  {level:20s} {ref_key:10s} {mode:19s} {len(results):6d} {n_ok:7d}  {'PASS' if ok else 'FAIL'}")
        for passed, msg in results:
            if not passed:
                print(f"      {msg}")

    print()
    print("  INTENDED DIFFERENCE: adaptive_jenks (plain Jenks) vs density-ref (gated)")
    da_diff = [run_da_diff_case(*case) for case in da_diff_cases()]
    md_diff = [run_model_diff_case(*case) for case in model_diff_cases()]
    da_cmp, da_rows = sum(d for d, _, _ in da_diff), sum(r for _, r, _ in da_diff)
    da_tot, md_cmp, md_tot = sum(t for _, _, t in da_diff), sum(d for d, _ in md_diff), sum(t for _, t in md_diff)
    print(f"  {'DynamicAdjacency':20s} {'density':10s} {'adaptive_jenks':19s} {len(da_diff):6d} cases: "
          f"{da_cmp}/{da_tot} forward comparisons differ, {da_rows} adjacency rows")
    print(f"  {'GraphAwareDenoiser':20s} {'density':10s} {'adaptive_jenks':19s} {len(md_diff):6d} cases: "
          f"{md_cmp}/{md_tot} forward comparisons differ")
    differs_ok = da_cmp > 0 and md_cmp > 0
    all_ok &= differs_ok
    print(f"  {'':20s} {'':10s} {'':19s} {'':6s} {'PASS (the change took effect)' if differs_ok else 'FAIL: still identical'}")

    print()
    checks = []
    for mode in ENTMAX_MODES:
        checks += [(f"{mode}: support + weights + grads", check_entmax_support_and_weights, mode),
                   (f"{mode}: k_min floor", check_entmax_k_min, mode),
                   (f"{mode}: warm-up == Jenks warm-up", check_entmax_warmup_is_jenks_warmup, mode),
                   (f"{mode}: topk_stats.csv", check_entmax_logging, mode)]
    checks.append(("plain vs gated routing (build_adjacency unit)", check_plain_vs_gated_routing))
    checks.append(("jenks_gate_stats.csv written by Jenks only", check_jenks_gate_log))
    checks.append(("invalid topk_mode raises", check_invalid_mode_raises))
    for label, fn, *args in checks:
        ok, msg = _run(label, fn, *args)
        all_ok &= ok
        print(f"{label:52s} {'PASS' if ok else 'FAIL'}" + (f"  {msg}" if msg else ""))

    print()
    print("ALL PASS" if all_ok else "FAILURES ABOVE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
