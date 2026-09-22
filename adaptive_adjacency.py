"""
Adaptive sparsification of the per-block dynamic adjacency (Section 5.4.4).

Replaces the fixed top-k with a per-node, per-sample, per-block, per-timestep
budget obtained by a 2-class Jenks natural-breaks split of the raw scores phi.

Drop-in: call build_adjacency(phi, ...) where top-k is currently applied,
i.e. after Eq. (5.4.7) and before Eq. (5.4.8).

build_adjacency_entmax(phi, alpha, ...) is a second rule for the same slot:
the kept edges are the support of alpha-entmax (sparsemax for alpha=2,
entmax-1.5 for alpha=1.5) over each row. Mask-only -- the weights on the kept
edges are the same sigmoid gates in both.
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


# ---------------------------------------------------------------------------
# alpha-entmax support (second adaptive rule, same role as the Jenks budget)
# ---------------------------------------------------------------------------

# alpha per topk_mode. Fixed, not learned: no new trainable parameters.
ENTMAX_ALPHA = {"adaptive_sparsemax": 2.0, "adaptive_entmax15": 1.5}


@torch.no_grad()
def entmax_support(phi, alpha, k_min=1):
    """
    Support of alpha-entmax over each row's off-diagonal scores.

    phi   : [B, F, F] raw scores; row i = neighbours of node i
    alpha : 2.0 (sparsemax) or 1.5 (entmax-1.5); both exact, sort-based
    returns
        mask : [B, F, F] bool, True = kept edge; diagonal always False
        k    : [B, F]    support size per node (>= k_min)
    """
    # Lazy: only these two modes need the package (Peters, Niculae & Martins).
    from entmax import entmax15, sparsemax

    B, F, _ = phi.shape
    eye = torch.eye(F, dtype=torch.bool, device=phi.device).expand(B, F, F)
    phi_off = phi.detach().masked_fill(eye, float("-inf"))   # drop self-loops

    if alpha == 2.0:
        p = sparsemax(phi_off, dim=-1)
    elif alpha == 1.5:
        p = entmax15(phi_off, dim=-1)
    else:
        raise ValueError(f"alpha must be 2.0 (sparsemax) or 1.5 (entmax15), got {alpha}")
    mask = p > 0                                             # the support

    # Floor of k_min kept edges per node. Both maps always keep the row max,
    # so for k_min=1 this is a no-op; it is explicit so the guarantee does not
    # rest on that property.
    top = phi_off.topk(min(k_min, F - 1), dim=-1).indices
    mask = mask.scatter(-1, top, True)
    return mask, mask.sum(dim=-1)


def build_adjacency_entmax(phi, alpha, k_min=1):
    """
    Same contract as build_adjacency, but the kept edges are the support of
    alpha-entmax over each row instead of a top-k / Jenks budget.

    Mask-only: entmax decides WHICH edges exist. It never supplies the edge
    weights -- those stay the independent sigmoid gates of Eq. (5.4.8), built
    by the same two lines as build_adjacency. The support is computed under
    no_grad, so gradients reach phi only through sigmoid(phi) on kept edges.
    """
    B, F, _ = phi.shape
    eye = torch.eye(F, dtype=torch.bool, device=phi.device).expand(B, F, F)
    mask, k = entmax_support(phi, alpha, k_min)

    A = torch.sigmoid(phi.masked_fill(~mask, NEG))           # Eq. (5.4.8)
    A = A.masked_fill(eye, 1.0)                              # enforced self-loops
    k_float = k.float()
    stats = {"k_mean": k_float.mean(),
             "k_per_sample": k_float.mean(dim=-1)}           # [B], for per-t-bucket logging
    return A, stats
