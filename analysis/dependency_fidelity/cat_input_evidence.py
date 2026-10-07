"""Evidence for the categorical-input bug, recomputed from the data each run.

Used by dependency_fidelity.py to write the "Categorical-input bug" section of
REPORT.md. Read-only: loss.csv and sample files of the ORIGINAL (cat_input=log)
runs, the real data, and the repo's own diffusion / denoiser classes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "topk_comparison"))
import compare_topk as ct  # noqa: E402

REPO = ct.REPO
RULE_TREES_LEGACY = {"fixed": "ablation", "jenks": ct.RULE_DIR["jenks"], "entmax15": ct.RULE_DIR["entmax15"]}


def _torch():
    for p in (str(REPO), str(REPO / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    return torch


def _codes(ds):
    Xc = np.load(REPO / "data" / ds / "X_cat_train.npy", allow_pickle=True).astype(str)
    codes = np.stack([np.searchsorted(np.array(sorted(set(Xc[:, j]))), Xc[:, j]) for j in range(Xc.shape[1])], 1)
    K = np.array([len(set(Xc[:, j])) for j in range(Xc.shape[1])])
    return codes, K


def loss_baselines(ds) -> dict:
    """Multinomial loss of an always-uniform and a real-marginal x0 predictor,
    under the pipeline's GaussianMultinomialDiffusion (T=1000, cosine)."""
    torch = _torch()
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    codes, K = _codes(ds)
    marg = np.concatenate([np.bincount(codes[:, j], minlength=K[j]) / len(codes) for j in range(len(K))])

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x, t, **kw):
            return self.v.repeat(x.shape[0], 1)

    out = {}
    for name, v in (("uniform", torch.zeros(int(K.sum()))), ("marginal", torch.log(torch.tensor(marg).float() + 1e-12))):
        torch.manual_seed(0)
        d = GaussianMultinomialDiffusion(num_classes=K, num_numerical_features=0, denoise_fn=Const(v),
                                         num_timesteps=1000, scheduler="cosine", device=torch.device("cpu"))
        rng = np.random.default_rng(0)
        n = min(4096, len(codes))
        out[name] = float(np.mean([d.mixed_loss(torch.tensor(codes[rng.choice(len(codes), n, replace=False)]).float(),
                                                {})[0].item() for _ in range(20)]))
    return out


def final_mloss(ds) -> dict:
    out = {}
    for r, t in RULE_TREES_LEGACY.items():
        f = sorted((ct.RESULTS_ROOT / ds / t).glob("gnn_*/loss.csv"))
        if f:
            out[r] = float(pd.read_csv(f[0]).mloss.tail(10).mean())
    return out


def rowwise_agreement(ds) -> dict:
    """Share of rows whose sampled category is identical between two rules'
    gen seed 0 (same seed -> same noise), per categorical column."""
    got = {}
    for r, t in RULE_TREES_LEGACY.items():
        f = sorted((ct.RESULTS_ROOT / ds / t).glob("gnn_*/gen_seed_0/X_cat_train.npy"))
        if f:
            got[r] = np.load(f[0], allow_pickle=True).astype(str)
    if len(got) < 2:
        return {}
    a, b = list(got)[:2]
    return {"pair": f"{a} vs {b}",
            "agreement": [float((got[a][:, j] == got[b][:, j]).mean()) for j in range(got[a].shape[1])]}


def marginal_tvd(ds) -> list:
    real = np.load(REPO / "data" / ds / "X_cat_train.npy", allow_pickle=True).astype(str)
    f = sorted((ct.RESULTS_ROOT / ds / RULE_TREES_LEGACY["jenks"]).glob("gnn_*/gen_seed_0/X_cat_train.npy"))
    if not f:
        return []
    syn = np.load(f[0], allow_pickle=True).astype(str)
    out = []
    for j in range(real.shape[1]):
        vals = sorted(set(real[:, j]))
        pr = np.array([(real[:, j] == v).mean() for v in vals])
        ps = np.array([(syn[:, j] == v).mean() for v in vals])
        out.append(float(0.5 * np.abs(pr - ps).sum()))
    return out


def embedding_norms() -> dict:
    """Fresh cardio-shaped denoiser: mean node-embedding norms at init."""
    torch = _torch()
    from tab_ddpm.graph_denoiser import GraphAwareDenoiser, timestep_embedding
    from tab_ddpm.utils import index_to_log_onehot
    torch.manual_seed(0)
    K, d_num, B = [2, 3, 3, 2, 2, 2], 5, 512
    m = GraphAwareDenoiser(d_in=d_num + sum(K), num_classes=2, is_y_cond=True, d_num=d_num, cat_sizes=K,
                           d_model=128, n_layers=3, n_heads=8, graph_mode="dynamic", top_k=5, gnn_type="gatv2",
                           use_attention=False)
    x_cat = torch.stack([torch.randint(0, k, (B,)) for k in K], 1)
    logoh = index_to_log_onehot(x_cat, np.array(K))
    num = (torch.randn(B, d_num).unsqueeze(-1) * m.num_in_weight.unsqueeze(0) + m.num_in_bias.unsqueeze(0))
    off, cl, cp = 0, [], []
    with torch.no_grad():
        for i, k in enumerate(K):
            s = logoh[:, off:off + k]
            cl.append(m.cat_in_projs[i](s).norm(dim=-1).mean().item())
            cp.append(m.cat_in_projs[i](s.exp()).norm(dim=-1).mean().item())
            off += k
        temb = m.time_embed(timestep_embedding(torch.randint(0, 1000, (B,)), m.dim_t)).norm(dim=-1).mean().item()
        e1 = m.time_embed(timestep_embedding(torch.full((1,), 1), m.dim_t))
        e2 = m.time_embed(timestep_embedding(torch.full((1,), 999), m.dim_t))
        dt = (e2 - e1).norm().item()
    return {"numerical": float(num.norm(dim=-1).mean()), "categorical_log": float(np.mean(cl)),
            "categorical_prob": float(np.mean(cp)), "timestep": temb,
            "t_range_over_cat_log": dt / float(np.mean(cl)), "t_range_over_num": dt / float(num.norm(dim=-1).mean())}


def section(cov: pd.DataFrame) -> list[str]:
    L = ["## Categorical-input bug\n"]
    L.append("**Cause.** `GraphAwareDenoiser` embedded each categorical column from the diffusion's *log* one-hot "
             "(`tab_ddpm/utils.py:133`, `log(one_hot.clamp(min=1e-30))`, so empty classes are ≈ −69) through its "
             "own `Linear(K, d_model)` (`tab_ddpm/graph_denoiser.py`, the categorical input projection), with no "
             "normalisation. With K = 2–5 the initial weights are large, and the categorical node embedding "
             "dwarfs the timestep embedding added to it: those nodes barely see t. Some categorical columns then "
             "learn to copy their noisy input x_t instead of predicting x_0; the reverse chain stays near its "
             "uniform starting draw. The original TabDDPM MLP gets the same input through one shared "
             "`Linear(d_in, 128)` (`tab_ddpm/modules.py:444,459`) and did not show the failure in the toy test.")
    L.append("\n**Fix.** `GraphAwareDenoiser(cat_input=\"prob\")` feeds `exp()` of that slice, the one-hot "
             "probabilities. `cat_input=\"log\"` (default) is bit-identical to the code before the option "
             "(tests/test_cat_input.py, against 3b8b8b3), so existing checkpoints and split jobs are unaffected. "
             "Fixed runs go to `*_catfix` trees; `split_runner --load_checkpoint` refuses a checkpoint trained with "
             "the other setting (`denoiser_config.json`, absent = \"log\").\n")

    L.append("### Evidence\n")
    L.append("**1. Multinomial loss vs trivial predictors** (final MLoss = mean of the last 10 logged values of "
             "`loss.csv`; baselines computed with the pipeline's diffusion, T = 1000, cosine). Below the marginal "
             "baseline = the model uses dependencies; at or above the uniform baseline = it has learned nothing.\n")
    L.append("| dataset | always-uniform U | real marginal M | fixed | Jenks | entmax-1.5 |")
    L.append("|---|---|---|---|---|---|")
    cat_ds = sorted(d for d in cov.dataset.unique()
                    if (REPO / "data" / d / "X_cat_train.npy").exists()
                    and any(list((ct.RESULTS_ROOT / d / t).glob("gnn_*/loss.csv")) for t in RULE_TREES_LEGACY.values()))
    for ds in cat_ds:
        b, f = loss_baselines(ds), final_mloss(ds)
        def code(v):
            # ≥U: no better than always-uniform; <M: clearly beats the marginal
            # (uses dependencies); ≈M otherwise.
            if v >= b["uniform"] - 0.02:
                return "≥U"
            return "<M" if v < b["marginal"] - 0.05 else "≈M"
        vals = [f.get(r) for r in ("fixed", "jenks", "entmax15")]
        L.append(f"| {ds} | {b['uniform']:.3f} | {b['marginal']:.3f} | "
                 + " | ".join(f"{v:.3f} {code(v)}" if v is not None else "—" for v in vals) + " |")

    L.append("\n≥U = no better than always predicting uniform (within 0.02); <M = more than 0.05 below the "
             "real-marginal predictor, i.e. the model uses dependencies; ≈M = in between. Where U ≈ M (abalone, "
             "insurance, churn2: near-balanced marginals) the loss alone cannot separate the two.")
    e = embedding_norms()
    L.append(f"\n**2. Embedding scale at initialisation** (fresh cardio-shaped GATv2 denoiser, d_model 128): mean "
             f"node-embedding norm — categorical from log one-hot **{e['categorical_log']:.0f}**, categorical from "
             f"one-hot probabilities {e['categorical_prob']:.1f}, numerical {e['numerical']:.2f}, timestep "
             f"{e['timestep']:.2f}. The whole t = 1 → 999 change of the timestep embedding is "
             f"{e['t_range_over_cat_log']:.1%} of a categorical node's embedding, against "
             f"{e['t_range_over_num']:.0%} of a numerical node's.")

    L.append("\n**3. The broken columns ignore the model.** Every rule samples with the same gen seed, i.e. the "
             "same noise. Share of rows with the identical category in two independently trained rules "
             "(gen seed 0), next to the column's TVD to the real marginal (Jenks, gen seed 0):\n")
    L.append("| dataset | pair | per column: agreement (TVD to real) |")
    L.append("|---|---|---|")
    for ds in ("cardio", "churn2", "default", "king"):
        if ds not in cat_ds:
            continue
        ag, tv = rowwise_agreement(ds), marginal_tvd(ds)
        if not ag:
            continue
        cols = ", ".join(f"cat_{j} {a:.3f} ({tv[j]:.2f})" for j, a in enumerate(ag["agreement"]))
        L.append(f"| {ds} | {ag['pair']} | {cols} |")
    L.append("\nColumns with near-total agreement across models *and* a large TVD (e.g. cardio cat_4, cat_5; "
             "churn2 cat_1, cat_2) are set by the shared noise, not by the model. The output is not decoded "
             "wrongly: `X_cat_unnorm` codes map one-to-one to the decoded labels.")

    L.append("\n**4. Toy test** (`tests/test_cat_input.py`, `RUN_SLOW=1`; one numerical column and a 95/5 binary "
             "column tied to it, the pipeline's diffusion, GATv2, 2 000 steps, 5 seeds; uniform-baseline MLoss "
             "0.724, marginal 0.210). `log`: 4/5 seeds stuck at a sampled P(c=1) ≈ 0.50 with MLoss 1.20–1.68 "
             "(worse than uniform). `prob`: 5/5 seeds learn (MLoss 0.106–0.144, sampled P 0.04–0.09 in 4/5, 0.28 "
             "in one; corr(x, c) +0.13 to +0.49 vs +0.42 real). The TabDDPM MLP: 5/5 learn (MLoss 0.11–0.20).")

    chk = HERE / "cat_input_local_check.json"
    if chk.exists():
        r = json.loads(chk.read_text())
        st = r["settings"]
        L.append(f"\n**5. Local real-data check** (`cat_input_local_check.py`; CPU, small GATv2 d{st['d_model']} "
                 f"L{st['n_layers']}, {st['steps']} steps, one seed, {st['n_samples']:,} samples — a direction "
                 "check, not the cluster protocol):\n")
        L.append("| dataset | cat_input | final MLoss | uniform | marginal | TVD per categorical column | max |")
        L.append("|---|---|---|---|---|---|---|")
        for run in r["runs"]:
            b = r["baselines"][run["dataset"]]
            L.append(f"| {run['dataset']} | {run['cat_input']} | {run['final_mloss']:.3f} | {b['uniform']:.3f} | "
                     f"{b['marginal']:.3f} | {', '.join(f'{v:.3f}' for v in run['tvd_per_column'])} | "
                     f"{run['tvd_max']:.3f} |")

    L.append("\n### Affected runs\n")
    aff = cov[cov.cat_input_affected].dataset.unique().tolist() if "cat_input_affected" in cov else []
    L.append("Every GNN run trained before the fix on a dataset **with categorical columns**, for every rule: the "
             "full ablation grids (`transfer_runs/exp/<ds>/ablation/<ds>_ablation.json`), the transfer and "
             "adaptive runs, and fb-comments, still running with `cat_input=log`. Datasets: "
             + ", ".join(f"{d} ({ct.n_categorical(d)} cat)" for d in sorted(aff)) + ". Numerical-only datasets "
             "(california, diabetes, gesture, higgs-small, house, miniboone, wilt) are unaffected: with no "
             "categorical column the option changes nothing. Within one dataset all rules share the training "
             "seed and hence the same initial categorical weights, so the stuck columns tend to be the same "
             "across rules; comparisons between rules are less affected than absolute numbers, but the "
             "oversized categorical nodes also enter the attention scores the top-k rules act on, so an effect "
             "on rule comparisons cannot be ruled out. Reruns: `hpc_transfer/jobs_catfix/` "
             "(`hpc_transfer/submit_catfix.sh`, priority 1: cardio, default, king; 2: abalone, adult, buddy, "
             "churn2, insurance).\n")
    return L
