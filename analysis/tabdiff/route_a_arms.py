"""
Route A arms (selection rules) and their run directories, resolved by the rule
token in the arm directory name -- never by an architecture string (diabetes is
v2_gcn_layers2_..._heads4, the others v2_gatv2_layers3_..._heads8, magic layers2).

    fixed     the main run: token topk<k>, k = the source configuration's top_k
    jenks     token jenks      (--topk_mode adaptive_jenks)
    entmax15  token entmax15   (--topk_mode adaptive_entmax15)
    dense     ablation topk_dense: token topk<N-1> (beijing, magic only)

Every directory found by token is checked against tabdiff_gnn/config.run_dir
(the layout the runner wrote), so the two can never disagree silently.
Standard library only (+ tabdiff_gnn/config.py): the cluster analysis imports it.
"""
from __future__ import annotations

import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "tabdiff_gnn") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "tabdiff_gnn"))
import config as C  # noqa: E402  (tabdiff_gnn/config.py)

ARMS = ("fixed", "jenks", "entmax15", "dense")
# arm -> (--topk_mode, --ablation)
ARM_MODE = {"fixed": ("fixed", None), "jenks": ("adaptive_jenks", None),
            "entmax15": ("adaptive_entmax15", None), "dense": ("fixed", "topk_dense")}
ARM_LABEL = {"fixed": "fixed top-k", "jenks": "Jenks", "entmax15": "entmax-1.5", "dense": "dense (k = N−1)"}


def has_arm(ds: str, arm: str) -> bool:
    """dense exists only where the topk_dense ablation is defined."""
    return arm != "dense" or ds in C.ABLATIONS["topk_dense"]["datasets"]


def rule_token(ds: str, arm: str, repo: str = REPO) -> str:
    mode, abl = ARM_MODE[arm]
    if mode != "fixed":
        return C.RULE_TAG[mode]
    return f"topk{C.gnn_config(ds, repo, ablation=abl)[0]['top_k']}"


def arm_dir(exp_root: str, ds: str, arm: str, seed: int = 0, repo: str = REPO) -> str | None:
    """<exp_root>/exp_tabdiff_gnn/tabdiff_<ds>/attngnn/<arm dir>/seed<seed>, or None if
    there is no such run on disk. Raises if the token matches several directories or the
    match differs from config.run_dir."""
    if not has_arm(ds, arm):
        return None
    tok = rule_token(ds, arm, repo)
    base = os.path.join(C.results_root(exp_root), C.target_name(ds), "attngnn")
    hits = [d for d in sorted(glob.glob(os.path.join(base, "*"))) if tok in os.path.basename(d).split("_")]
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"{ds}/{arm}: token {tok!r} matches {len(hits)} directories: {hits}")
    path = os.path.join(hits[0], f"seed{seed}")
    mode, abl = ARM_MODE[arm]
    want = C.run_dir(exp_root, C.target_name(ds), "attngnn", seed, repo_root=repo, topk_mode=mode, ablation=abl)
    if os.path.realpath(path) != os.path.realpath(want):
        raise RuntimeError(f"{ds}/{arm}: found {path} by token {tok!r}, config.run_dir says {want}")
    return path if os.path.isdir(path) else None
