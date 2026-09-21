"""
Adaptive sparsification of the per-block dynamic adjacency (Section 5.4.4).

Replaces the fixed top-k with a per-node, per-sample, per-block, per-timestep
budget obtained by a 2-class Jenks natural-breaks split of the raw scores phi.

Drop-in: call build_adjacency(phi, ...) where top-k is currently applied,
i.e. after Eq. (5.4.7) and before Eq. (5.4.8).
"""

import torch

NEG = -1e9  # "large negative value" used for masking, as in Section 5.4.4


@torch.no_grad()
def jenks_budget(s, k_min=1, use_gate_rule=True, delta=None):
    """
    2-class Jenks split of already-sorted (descending) scores.

    s : [..., n]  off-diagonal scores per node, sorted descending
    returns
        k    : [...]  size of the strong class (clamped to >= k_min)
        stats: dict with mean of strong class and the gap, for logging
    """
    n = s.size(-1)
    c1, c2 = s.cumsum(-1), (s ** 2).cumsum(-1)
    m = torch.arange(1, n, device=s.device, dtype=s.dtype)   # strong-class size
    a1, a2 = c1[..., :-1], c2[..., :-1]                      # strong class
    b1, b2 = c1[..., -1:] - a1, c2[..., -1:] - a2            # weak class

    # sum of squared deviations about class means, for every candidate split
    sdcm = (a2 - a1 ** 2 / m) + (b2 - b1 ** 2 / (n - m))
    best = sdcm.argmin(-1, keepdim=True)                     # [..., 1]

    mu_s = (a1 / m).gather(-1, best).squeeze(-1)             # mean of strong class
    mu_w = (b1 / (n - m)).gather(-1, best).squeeze(-1)       # mean of weak class
    k = best.squeeze(-1) + 1

    # safeguard against a forced split of an essentially flat row
    if use_gate_rule:
        accept = mu_s > 0.0          # sigma(mu_s) > 0.5: gates are actually open
    elif delta is not None:
        accept = (mu_s - mu_w) > delta
    else:
        accept = torch.ones_like(k, dtype=torch.bool)

    k = torch.where(accept, k, torch.full_like(k, k_min))
    return k.clamp_(min=k_min, max=n), {"mu_strong": mu_s, "gap": mu_s - mu_w}


def build_adjacency(phi, k_fixed=None, k_min=1, sym="none",
                    use_gate_rule=True, delta=None, share_across_batch=False):
    """
    phi : [B, F, F] raw scores from Eq. (5.4.7); row i = neighbours of node i.
    k_fixed : if given, behave exactly like the current fixed top-k (warm-up).
    sym : "none" (directed, current design) | "union" | "mutual".
    share_across_batch : use the batch median budget instead of a per-sample one.
    """
    B, F, _ = phi.shape
    eye = torch.eye(F, dtype=torch.bool, device=phi.device).expand(B, F, F)
    phi_off = phi.masked_fill(eye, NEG)                      # drop self-loops

    s, _ = phi_off.sort(dim=-1, descending=True)
    s = s[..., : F - 1]                                      # sorted off-diagonal

    if k_fixed is not None:
        k = torch.full((B, F), min(k_fixed, F - 1), device=phi.device)
        stats = {}
    else:
        k, stats = jenks_budget(s.detach(), k_min, use_gate_rule, delta)
        if share_across_batch:
            k = k.median(dim=0, keepdim=True).values.expand(B, F)

    # keep every score >= the k-th largest one of its row
    thresh = s.gather(-1, (k - 1).unsqueeze(-1))             # [B, F, 1]
    mask = phi_off >= thresh

    if sym == "union":
        mask = mask | mask.transpose(-1, -2)
    elif sym == "mutual":
        mask = mask & mask.transpose(-1, -2)

    A = torch.sigmoid(phi.masked_fill(~mask, NEG))           # Eq. (5.4.8)
    A = A.masked_fill(eye, 1.0)                              # enforced self-loops
    k_float = k.float()
    stats["k_mean"] = k_float.mean()
    stats["k_per_sample"] = k_float.mean(dim=-1)              # [B], for per-t-bucket logging
    return A, stats
