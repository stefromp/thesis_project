"""
GraphAwareDenoiser(cat_input=...): the categorical-input fix.

"log"  -- the default; what every run before the fix was trained with. Must be
          bit-identical to the code before the option existed (PRE_CAT_REV), so
          existing checkpoints and split jobs that load them are unaffected.
"prob" -- categorical nodes see exp(log one-hot) = the one-hot probabilities.

Fast tests (always run):
  * "log" (explicit and default) vs PRE_CAT_REV: forward outputs, gradients and
    parameters after Adam steps are torch.equal, for dataset shapes with
    categorical columns and the backbones / top-k modes the runs use.
  * "prob" differs from "log" only through the categorical slice: identical to
    feeding a "log" model the exp'd slice, and identical to "log" when there
    are no categorical columns.
  * an unknown cat_input is rejected; results_subdir / checkpoint_cat_input.

Slow regression (RUN_SLOW=1): the toy problem that exposed the bug -- one
numerical column and a 95/5 binary column that depends on it, the pipeline's
GaussianMultinomialDiffusion (T=1000, cosine) and a small GATv2 denoiser.
"prob" must bring the final multinomial loss below the always-uniform baseline
in >= 4 of 5 seeds; "log" must still reproduce the failure (above the uniform
baseline) in >= 3 of 5 seeds, i.e. the default really is the old behaviour.

    python3 -m pytest -q tests/test_cat_input.py
    RUN_SLOW=1 python3 -m pytest -q tests/test_cat_input.py -k toy
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE, os.path.join(REPO, "ablation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tab_ddpm.graph_denoiser import GraphAwareDenoiser as NewModel  # noqa: E402
from tab_ddpm.utils import index_to_log_onehot  # noqa: E402
from test_topk_modes_bitexact import float32_default, load_reference  # noqa: E402

PRE_CAT_REV = "3b8b8b3"   # last commit before cat_input existed


def _have_revision(rev: str) -> bool:
    """The bit-exactness reference is read from git; a checkout without the
    history (e.g. the cluster copy) skips that test instead of failing it."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"{rev}^{{commit}}"],
                           capture_output=True)
        return r.returncode == 0
    except OSError:
        return False
T = 1000

# (name, d_num, cat_sizes, num_classes, is_y_cond) as the denoiser sees them
SHAPES = [
    ("cardio",  5, [2, 3, 3, 2, 2, 2], 2, True),
    ("abalone", 8, [3],                0, False),
    ("king",   18, [2, 5, 5],          0, False),
]


def _kwargs(shape, gnn_type, topk_mode):
    _, d_num, cat_sizes, num_classes, is_y_cond = shape
    return dict(d_in=d_num + sum(cat_sizes), num_classes=num_classes, is_y_cond=is_y_cond,
                d_num=d_num, cat_sizes=cat_sizes, d_model=16, n_layers=2, n_heads=4,
                graph_mode="dynamic", top_k=2, gnn_type=gnn_type, dropout=0.0,
                use_attention=False, topk_mode=topk_mode, k_min=1, topk_warmup_steps=0,
                num_timesteps=T)


def _batch(shape, g, B=48):
    """What the diffusion actually feeds the denoiser: [x_num_t | log one-hot]."""
    _, d_num, cat_sizes, num_classes, _ = shape
    parts = [torch.randn(B, d_num, generator=g)]
    if cat_sizes:
        idx = torch.stack([torch.randint(0, K, (B,), generator=g) for K in cat_sizes], 1)
        parts.append(index_to_log_onehot(idx, np.array(cat_sizes)))
    t = torch.randint(0, T, (B,), generator=g)
    y = (torch.randint(0, num_classes, (B,), generator=g) if num_classes > 0 else torch.randn(B, generator=g))
    return torch.cat(parts, 1), t, y


CASES = [(s, g, m) for s in SHAPES for g in ("gcn", "gatv2") for m in ("fixed", "adaptive_jenks")]


@pytest.mark.parametrize("shape,gnn_type,topk_mode", CASES, ids=lambda v: v[0] if isinstance(v, tuple) else v)
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit-log"])
@pytest.mark.skipif(not _have_revision(PRE_CAT_REV),
                    reason=f"git revision {PRE_CAT_REV} not available (no git history here)")
def test_log_is_bit_identical_to_pre_cat_input(shape, gnn_type, topk_mode, explicit, tmp_path):
    ref_pkg = load_reference(PRE_CAT_REV, "ref_precat_tab_ddpm")
    from tab_ddpm import topk_logging as new_log
    ref_log = sys.modules["ref_precat_tab_ddpm.topk_logging"]
    kw = _kwargs(shape, gnn_type, topk_mode)
    with float32_default():
        torch.manual_seed(0)
        ref = ref_pkg.graph_denoiser.GraphAwareDenoiser(**kw)
        new = NewModel(**kw, **({"cat_input": "log"} if explicit else {}))
        new.load_state_dict(ref.state_dict())
        opt_r = torch.optim.Adam(ref.parameters(), lr=1e-3)
        opt_n = torch.optim.Adam(new.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(1)
        new_log.set_context(output_dir=str(tmp_path / "new"), log_every=1, tag="")
        ref_log.set_context(output_dir=str(tmp_path / "ref"), log_every=1, tag="")
        for step in range(4):
            x, t, y = _batch(shape, g)
            out_r, out_n = ref(x, t, y), new(x, t, y)
            assert torch.equal(out_r, out_n), f"forward differs at step {step}"
            opt_r.zero_grad(); opt_n.zero_grad()
            out_r.square().mean().backward(); out_n.square().mean().backward()
            for (name, pr), (_, pn) in zip(ref.named_parameters(), new.named_parameters()):
                assert (pr.grad is None and pn.grad is None) or torch.equal(pr.grad, pn.grad), f"grad {name}"
            opt_r.step(); opt_n.step()
        for (name, pr), (_, pn) in zip(ref.named_parameters(), new.named_parameters()):
            assert torch.equal(pr, pn), f"param {name} after Adam"


@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
def test_prob_equals_log_model_fed_probabilities(shape):
    kw = _kwargs(shape, "gatv2", "fixed")
    with float32_default():
        torch.manual_seed(0)
        log_m = NewModel(**kw, cat_input="log")
        prob_m = NewModel(**kw, cat_input="prob")
        prob_m.load_state_dict(log_m.state_dict())
        x, t, y = _batch(shape, torch.Generator().manual_seed(2))
        d_num = shape[1]
        x_prob = torch.cat([x[:, :d_num], x[:, d_num:].exp()], 1)
        assert torch.equal(prob_m(x, t, y), log_m(x_prob, t, y))
        assert not torch.equal(prob_m(x, t, y), log_m(x, t, y)), "prob must change the model when n_cat > 0"


def test_prob_is_a_no_op_without_categoricals():
    shape = ("wilt", 5, [], 2, True)
    kw = _kwargs(shape, "gatv2", "fixed")
    with float32_default():
        torch.manual_seed(0)
        a, b = NewModel(**kw, cat_input="log"), NewModel(**kw, cat_input="prob")
        b.load_state_dict(a.state_dict())
        x, t, y = _batch(shape, torch.Generator().manual_seed(3))
        assert torch.equal(a(x, t, y), b(x, t, y))


def test_unknown_cat_input_is_rejected():
    with pytest.raises(ValueError, match="cat_input"):
        NewModel(**_kwargs(SHAPES[0], "gcn", "fixed"), cat_input="onehot")


def test_results_subdir_and_checkpoint_record(tmp_path):
    import json
    import ablation_runner as ar
    assert ar.results_subdir("fixed", "log") == "ablation"
    assert ar.results_subdir("adaptive_jenks", "log") == "ablation_adaptive_jenks"
    assert ar.results_subdir("fixed", "prob") == "ablation_catfix"
    assert ar.results_subdir("adaptive_entmax15", "prob") == "ablation_adaptive_entmax15_catfix"
    with pytest.raises(ValueError):
        ar.results_subdir("fixed", "bogus")
    # no record = a checkpoint from before the option = "log"
    assert ar.checkpoint_cat_input(str(tmp_path)) == "log"
    (tmp_path / ar.DENOISER_CONFIG_FILE).write_text(json.dumps({"cat_input": "prob"}))
    assert ar.checkpoint_cat_input(str(tmp_path)) == "prob"


# ---------------------------------------------------------------------------
# Slow toy regression
# ---------------------------------------------------------------------------
TOY_ROWS, TOY_STEPS, TOY_BATCH, TOY_LR, TOY_SEEDS = 4000, 2000, 256, 1e-3, (0, 1, 2, 3, 4)


def _toy_data(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=TOY_ROWS)
    z = x + 0.5 * rng.normal(size=TOY_ROWS)
    c = (z > np.quantile(z, 0.95)).astype(int)          # exactly 5% ones, tied to x
    return torch.tensor(np.c_[x, c], dtype=torch.float32)


def _diffusion(denoiser):
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    return GaussianMultinomialDiffusion(num_classes=np.array([2]), num_numerical_features=1, denoise_fn=denoiser,
                                        num_timesteps=T, gaussian_loss_type="mse", scheduler="cosine",
                                        device=torch.device("cpu"))


def _uniform_baseline():
    class Uniform(torch.nn.Module):
        def forward(self, x, t, **kw):
            return torch.zeros(x.shape[0], 3)
    torch.manual_seed(0)
    d, X = _diffusion(Uniform()), _toy_data(0)
    return float(np.mean([d.mixed_loss(X[torch.randint(0, len(X), (1024,))], {})[0].item() for _ in range(30)]))


def _toy_final_mloss(cat_input, seed):
    with float32_default():
        torch.manual_seed(seed)
        X = _toy_data(seed)
        model = NewModel(d_in=3, num_classes=0, is_y_cond=False, d_num=1, cat_sizes=[2], d_model=64, n_layers=2,
                         n_heads=4, graph_mode="dynamic", top_k=1, gnn_type="gatv2", use_attention=False,
                         num_timesteps=T, cat_input=cat_input)
        diff = _diffusion(model)
        opt = torch.optim.AdamW(diff.parameters(), lr=TOY_LR, weight_decay=0.0)
        losses = []
        for _ in range(TOY_STEPS):
            lm, lg = diff.mixed_loss(X[torch.randint(0, len(X), (TOY_BATCH,))], {})
            opt.zero_grad(); (lm + lg).backward(); opt.step()
            losses.append(lm.item())
        return float(np.mean(losses[-200:]))


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="slow (~5 min on CPU); set RUN_SLOW=1")
def test_toy_prob_learns_and_log_reproduces_the_failure():
    uniform = _uniform_baseline()
    prob = [_toy_final_mloss("prob", s) for s in TOY_SEEDS]
    log = [_toy_final_mloss("log", s) for s in TOY_SEEDS]
    msg = f"uniform baseline {uniform:.3f}; prob {np.round(prob, 3).tolist()}; log {np.round(log, 3).tolist()}"
    print(msg)
    assert sum(m < uniform for m in prob) >= 4, "prob should learn the 95/5 column: " + msg
    assert sum(m > uniform for m in log) >= 3, "log should still reproduce the old failure: " + msg
