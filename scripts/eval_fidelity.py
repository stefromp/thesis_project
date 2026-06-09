"""
Fidelity metrics for synthetic tabular data evaluation.

Column-wise  : KST (numerical), TVD (categorical), ColumnWise_score
Pair-wise    : DPCM (num x num), DCSM (cat x cat), Mixed (num x cat), PairWise_score
Joint        : Coverage, beta_Recall
"""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

MAX_SAMPLES = 50_000


def _subsample(arr, n, rng):
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), n, replace=False)]


def _encode_combined(X_real_num, X_syn_num, X_real_cat, X_syn_cat):
    """Min-max scale numericals + OHE categoricals into a single feature matrix."""
    parts_r, parts_s = [], []

    if X_real_num is not None and X_real_num.shape[1] > 0:
        mm = MinMaxScaler().fit(X_real_num)
        parts_r.append(mm.transform(X_real_num))
        parts_s.append(mm.transform(X_syn_num))

    if X_real_cat is not None and X_real_cat.shape[1] > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_real_cat)
        parts_r.append(ohe.transform(X_real_cat) / np.sqrt(2))
        parts_s.append(ohe.transform(X_syn_cat) / np.sqrt(2))

    return np.concatenate(parts_r, axis=1), np.concatenate(parts_s, axis=1)


# ── column-wise ───────────────────────────────────────────────────────────────

def _kst_per_column(X_real_num, X_syn_num):
    """KS statistic for each numerical column."""
    return [
        ks_2samp(X_real_num[:, i], X_syn_num[:, i]).statistic
        for i in range(X_real_num.shape[1])
    ]


def _tvd_per_column(X_real_cat, X_syn_cat, cat_sizes):
    """TVD for each categorical column.

    TVD_j = 0.5 * sum_omega |R(omega) - S(omega)|
    """
    tvds = []
    for j, K in enumerate(cat_sizes):
        cats = np.arange(K)
        r = np.array([np.mean(X_real_cat[:, j] == c) for c in cats])
        s = np.array([np.mean(X_syn_cat[:,  j] == c) for c in cats])
        tvds.append(0.5 * float(np.sum(np.abs(r - s))))
    return tvds


def compute_kst(X_real_num, X_syn_num):
    """Mean KS statistic across all numerical columns (lower = better)."""
    return float(np.mean(_kst_per_column(X_real_num, X_syn_num)))


def compute_tvd(X_real_cat, X_syn_cat, cat_sizes):
    """Mean TVD across all categorical columns (lower = better)."""
    return float(np.mean(_tvd_per_column(X_real_cat, X_syn_cat, cat_sizes)))


def compute_column_wise_score(X_real_num, X_syn_num, X_real_cat, X_syn_cat, cat_sizes):
    """
    Aggregated column-wise fidelity score in [0, 100] (higher = better, real data = 100).

    For each column j:
        score_j = 1 - KST_j   if column j is numerical
        score_j = 1 - TVD_j   if column j is categorical

    ColumnWise = mean_j(score_j) * 100
    """
    scores = []

    if X_real_num is not None and X_real_num.shape[1] > 0:
        scores.extend(1.0 - v for v in _kst_per_column(X_real_num, X_syn_num))

    if X_real_cat is not None and X_real_cat.shape[1] > 0:
        scores.extend(1.0 - v for v in _tvd_per_column(X_real_cat, X_syn_cat, cat_sizes))

    return float(np.mean(scores) * 100)


# ── pair-wise ─────────────────────────────────────────────────────────────────

def _joint_prob(X_cat, ci, cj, Ki, Kj):
    C = np.zeros((Ki, Kj))
    np.add.at(C, (X_cat[:, ci], X_cat[:, cj]), 1)
    total = C.sum()
    return C / total if total > 0 else C


def _conditional_kst(X_real_num, X_syn_num, X_real_cat, X_syn_cat, ni, cj, K, min_samples):
    """Frequency-weighted mean KST of num_i conditional on each value of cat_j."""
    cat_ksts, weights = [], []
    for c in range(K):
        real_vals = X_real_num[X_real_cat[:, cj] == c, ni]
        syn_vals  = X_syn_num[X_syn_cat[:,  cj] == c, ni]
        if len(real_vals) >= min_samples and len(syn_vals) >= min_samples:
            cat_ksts.append(ks_2samp(real_vals, syn_vals).statistic)
            weights.append(float(len(real_vals)))
    if not cat_ksts:
        return None
    w = np.array(weights)
    w /= w.sum()
    return float(np.dot(w, cat_ksts))


def compute_dpcm(X_real_num, X_syn_num):
    """
    Mean absolute difference of Pearson correlation coefficients across all
    unique num x num column pairs (i < j). Lower = better.

    DPCM = mean_{i<j} |rho_real(i,j) - rho_syn(i,j)|
    """
    C_r = np.corrcoef(X_real_num.T)
    C_s = np.corrcoef(X_syn_num.T)
    d = C_r.shape[0]
    diffs = [abs(C_r[i, j] - C_s[i, j]) for i in range(d) for j in range(i + 1, d)]
    return float(np.mean(diffs)) if diffs else 0.0


def compute_dcsm(X_real_cat, X_syn_cat, cat_sizes):
    """
    Mean TVD over joint distributions for all unique cat x cat column pairs (i < j).
    Lower = better.

    For each pair (i, j):
        TVD_pair = 0.5 * sum_{a,b} |P_real(a,b) - P_syn(a,b)|

    DCSM = mean_{i<j} TVD_pair(i,j)
    """
    n_cat = X_real_cat.shape[1]
    if n_cat < 2:
        return 0.0

    pair_tvds = []
    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            Ki, Kj = cat_sizes[i], cat_sizes[j]
            Pr = _joint_prob(X_real_cat, i, j, Ki, Kj)
            Ps = _joint_prob(X_syn_cat,  i, j, Ki, Kj)
            pair_tvds.append(0.5 * float(np.sum(np.abs(Pr - Ps))))

    return float(np.mean(pair_tvds))


def compute_mixed(X_real_num, X_syn_num, X_real_cat, X_syn_cat, cat_sizes, min_samples=30):
    """
    Mean KS statistic of conditional numerical distributions for all num x cat pairs.
    Lower = better.

    For each (num_i, cat_j) pair:
        Weighted-mean KST of {num_i | cat_j=c} in real vs synthetic across all c
    Mixed = mean over all (num_i, cat_j) pairs
    """
    d_num = X_real_num.shape[1]
    n_cat = X_real_cat.shape[1]
    if n_cat == 0:
        return 0.0

    pair_scores = []
    for ni in range(d_num):
        for cj in range(n_cat):
            v = _conditional_kst(
                X_real_num, X_syn_num, X_real_cat, X_syn_cat,
                ni, cj, cat_sizes[cj], min_samples,
            )
            if v is not None:
                pair_scores.append(v)

    return float(np.mean(pair_scores)) if pair_scores else 0.0


def compute_pair_wise_score(
    X_real_num, X_syn_num, X_real_cat, X_syn_cat, cat_sizes, min_samples=30
):
    """
    Unified pair-wise fidelity score in [0, 100] (higher = better, real data ≈ 100).

    For every unique feature pair (i < j) across all column types, compute:

        sim_ij = 1 - distance_ij

    where:
        num x num  →  distance_ij = |rho_real(i,j) - rho_syn(i,j)|       ∈ [0, 2]
        cat x cat  →  distance_ij = TVD of joint P(a,b)                   ∈ [0, 1]
        num x cat  →  distance_ij = freq-weighted mean KST per category    ∈ [0, 1]

    PairWise_score = 100 × mean_{all pairs} sim_ij
    """
    sims = []

    d_num = X_real_num.shape[1] if X_real_num is not None else 0
    has_cat = X_real_cat is not None and X_real_cat.shape[1] > 0
    n_cat = X_real_cat.shape[1] if has_cat else 0

    # num x num
    if d_num >= 2:
        C_r = np.corrcoef(X_real_num.T)
        C_s = np.corrcoef(X_syn_num.T)
        for i in range(d_num):
            for j in range(i + 1, d_num):
                sims.append(1.0 - abs(C_r[i, j] - C_s[i, j]))

    # cat x cat
    if has_cat and n_cat >= 2:
        for i in range(n_cat):
            for j in range(i + 1, n_cat):
                Ki, Kj = cat_sizes[i], cat_sizes[j]
                Pr = _joint_prob(X_real_cat, i, j, Ki, Kj)
                Ps = _joint_prob(X_syn_cat,  i, j, Ki, Kj)
                tvd = 0.5 * float(np.sum(np.abs(Pr - Ps)))
                sims.append(1.0 - tvd)

    # num x cat
    if d_num > 0 and has_cat:
        for ni in range(d_num):
            for cj in range(n_cat):
                v = _conditional_kst(
                    X_real_num, X_syn_num, X_real_cat, X_syn_cat,
                    ni, cj, cat_sizes[cj], min_samples,
                )
                if v is not None:
                    sims.append(1.0 - v)

    return float(np.mean(sims) * 100) if sims else 0.0


# ── joint distribution ─────────────────────────────────────────────────────────

def _knn_radii(X, k):
    """Distance from each point to its k-th nearest neighbour (self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X)
    dists, _ = nn.kneighbors(X)
    return dists[:, k]


def compute_coverage(X_real, X_syn, k=5):
    """
    Fraction of real points whose k-NN ball (radius = k-th NN distance in real data)
    contains at least one synthetic point.
    Higher = better mode coverage.  (Naeem et al., 2020)
    """
    radii = _knn_radii(X_real, k)
    nn_syn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(X_syn)
    dists, _ = nn_syn.kneighbors(X_real)
    return float((dists[:, 0] <= radii).mean())


def compute_beta_recall(X_real, X_syn, k=5):
    """
    Fraction of real points that fall within at least one k-NN ball of synthetic data.
    A real point r is recalled if dist(r, s_j) <= radius_syn(s_j) for some j in its k-NN.
    Higher = better.  Proxy for beta-Recall (Kynkäänniemi et al., 2019).
    """
    radii_syn = _knn_radii(X_syn, k)
    nn_syn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(X_syn)
    dists, idx = nn_syn.kneighbors(X_real)           # (n_real, k)
    recalled = np.any(dists <= radii_syn[idx], axis=1).mean()
    return float(recalled)


# ── unified entry point ────────────────────────────────────────────────────────

def compute_fidelity_metrics(
    X_real_num,
    X_syn_num,
    X_real_cat,
    X_syn_cat,
    cat_sizes,
    k=5,
    seed=0,
):
    """
    Compute all fidelity metrics and return them as a dict.

    Parameters
    ----------
    X_real_num, X_syn_num : ndarray (n, d_num)          raw numerical features
    X_real_cat, X_syn_cat : ndarray (n, n_cat) or None  ordinal-encoded categoricals
    cat_sizes             : list[int]  cardinality of each categorical column
    k                     : int        neighbour count for Coverage / beta_Recall
    seed                  : int        for subsampling reproducibility
    """
    rng = np.random.default_rng(seed)
    out = {}

    has_cat = X_real_cat is not None and X_real_cat.shape[1] > 0
    n_cat   = X_real_cat.shape[1] if has_cat else 0

    # ── column-wise ───────────────────────────────────────────────────────────
    out["KST"]             = compute_kst(X_real_num, X_syn_num)
    out["TVD"]             = compute_tvd(X_real_cat, X_syn_cat, cat_sizes) if has_cat else None
    out["ColumnWise_score"] = compute_column_wise_score(
        X_real_num, X_syn_num,
        X_real_cat if has_cat else None,
        X_syn_cat  if has_cat else None,
        cat_sizes,
    )

    # ── pair-wise ─────────────────────────────────────────────────────────────
    out["DPCM"]  = compute_dpcm(X_real_num, X_syn_num)
    out["DCSM"]  = compute_dcsm(X_real_cat, X_syn_cat, cat_sizes) if n_cat >= 2 else None
    out["mixed"] = compute_mixed(
        X_real_num, X_syn_num, X_real_cat, X_syn_cat, cat_sizes
    ) if has_cat else None
    out["PairWise_score"] = compute_pair_wise_score(
        X_real_num, X_syn_num,
        X_real_cat if has_cat else None,
        X_syn_cat  if has_cat else None,
        cat_sizes,
    )

    # ── joint distribution (subsampled for scalability) ───────────────────────
    X_r_enc, X_s_enc = _encode_combined(X_real_num, X_syn_num, X_real_cat, X_syn_cat)
    X_r_sub = _subsample(X_r_enc, MAX_SAMPLES, rng)
    X_s_sub = _subsample(X_s_enc, MAX_SAMPLES, rng)

    out["Coverage"]    = compute_coverage(X_r_sub, X_s_sub, k=k)
    out["beta_Recall"] = compute_beta_recall(X_r_sub, X_s_sub, k=k)

    return out
