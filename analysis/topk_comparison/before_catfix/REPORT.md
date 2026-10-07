# Top-k rule comparison: fixed vs Jenks vs entmax-1.5

_Generated 2026-10-04 by `compare_topk.py` from `transfer_runs/exp`. Every number below is computed by the script; rerun it to refresh._

## How to read the two subsets

- **Held-out evaluation — transferred datasets (complete: 4).** The configuration was chosen by transfer from other datasets, without seeing any of this dataset's results. **The primary claims rest on this subset.**
- **Development set — ablated datasets (complete: 9).** The configuration is the argmax of the fixed-mode score on the **test split** over the dataset's own grid (caveats 1–2). Results are in-sample with respect to that selection and therefore optimistic: most of all for fixed, the rule the selection was made with, and also relative to the TabDDPM paper, which tuned on the validation split.
- **Pooled** numbers mix the two and inherit the development-set bias; they are reported for completeness only.

## Coverage

- Datasets known: 14. Complete for all three rules: **13** (abalone, adult, buddy, california, cardio, churn2, default, diabetes, gesture, house, insurance, king, wilt).
- Not complete: fb-comments (fixed partial, jenks partial, entmax15 missing). fb-comments has no results (decision pending) and is reported as missing, not analysed.
- Excluded entirely (jobs still running): higgs-small, miniboone.
- **Fixed on the development set is summary-only** for 5 datasets (abalone, adult, buddy, churn2, insurance): the grid JSON stores mean/std only. Their fixed utility has no per-generation-seed values, so no utility CI or win/tie/loss exists for Jenks/entmax vs fixed on them; fidelity/privacy metrics get an exact Welch test from the summary. The per-seed rerun in `hpc_fixed_perseed/` (same configuration, same training seed 0) is picked up automatically once its results exist. It only adds per-seed values: it removes neither the seed winner's curse nor the test-split selection bias (caveats 2 and 5).
- Configuration check: no mismatches (gnn type, layers, d_model, top_k, heads, attention identical across the three rules).

- **Categorical-input fix:** datasets on the fixed (`*_catfix`) runs: none yet. Datasets with categorical columns still on the original runs, marked § in the tables: abalone, adult, buddy, cardio, churn2, default, fb-comments, insurance, king. A dataset switches only when all three rules are complete in `*_catfix`; the two are never mixed. See caveat 10.

## Key findings

- **Held-out evaluation (transferred, n=4) — primary:** average rank fixed 2.00, Jenks 1.50, entmax-1.5 2.50; Friedman p=0.368. W/T/L: Jenks vs fixed 2/2/0 (median Δ +2.26%); entmax-1.5 vs fixed 1/1/2 (median Δ -0.44%); entmax-1.5 vs Jenks 0/1/3 (median Δ -0.38%). With n=4 no across-dataset test can reach significance (smallest exact two-sided Wilcoxon p is 0.125).
- **Development set (ablated, n=9) — in-sample, optimistic for fixed:** average rank fixed 1.56, Jenks 2.00, entmax-1.5 2.44; Friedman p=0.1690. entmax-1.5 vs Jenks W/T/L 1/2/6 (median Δ -1.02%).
- **Pooled (n=13; inherits the development-set bias, completeness only):** average rank fixed 1.69, Jenks 1.85, entmax-1.5 2.46; Friedman p=0.1160.
- **Large utility changes vs fixed (|Δ| > 5%):** house entmax-1.5 +5.4% [held-out, win]; king Jenks +9.4% [held-out, win]; king entmax-1.5 -9.3% [held-out, tie]; abalone Jenks -23.5% [development, no per-seed fixed]; insurance entmax-1.5 -48.5% [development, no per-seed fixed].
- **Privacy vs fixed:** 19 resolved regressions across 8 datasets (10 on held-out: default, house, king); the largest relative DCR drops are default Jenks -15.2%; buddy Jenks -13.8%; adult Jenks -12.9%. See 'Privacy as a constraint'.
- **Mechanism:** median k_eff/k_fixed is Jenks 0.87, entmax-1.5 0.56; both rules keep fewer neighbours than fixed on some datasets and more on others (table below), so neither is uniformly sparser.
- **Meta-features (exploratory):** strongest association is Jenks Δ vs n_features (ρ=+0.66, p=0.014, Holm p=0.111, n=13). assoc_density_0.3: Jenks ρ=-0.13 (p=0.661), entmax-1.5 ρ=+0.40 (p=0.174).

## Primary utility across datasets (score = accuracy / −RMSE)

| subset | role | n | avg rank fixed | avg rank Jenks | avg rank entmax | Friedman p |
|---|---|---|---|---|---|---|
| held-out evaluation (transferred) | **primary** | 4 | 2.00 | 1.50 | 2.50 | 0.3679 |
| development set (ablated) | in-sample, optimistic | 9 | 1.56 | 2.00 | 2.44 | 0.1690 |
| pooled (held-out + development) | completeness only | 13 | 1.69 | 1.85 | 2.46 | 0.1160 |

Pairwise Wilcoxon signed-rank on the relative Δ (% of the reference mean), Holm-corrected within each subset; W/T/L from per-dataset 95% CIs:

| subset | pair | median Δ% | Wilcoxon p | Holm p | W/T/L | not testable |
|---|---|---|---|---|---|---|
| held-out evaluation (transferred) | Jenks vs fixed | +2.26 | 0.375 | 1.000 | 2/2/0 | 0 |
| held-out evaluation (transferred) | entmax-1.5 vs fixed | -0.44 | 0.625 | 1.000 | 1/1/2 | 0 |
| held-out evaluation (transferred) | entmax-1.5 vs Jenks | -0.38 | 0.625 | 1.000 | 0/1/3 | 0 |
| development set (ablated) | Jenks vs fixed | -0.29 | 0.164 | 0.328 | 1/2/1 | 5 |
| development set (ablated) | entmax-1.5 vs fixed | -0.53 | 0.098 | 0.293 | 0/3/1 | 5 |
| development set (ablated) | entmax-1.5 vs Jenks | -1.02 | 0.426 | 0.426 | 1/2/6 | 0 |
| pooled (held-out + development) | Jenks vs fixed | -0.13 | 0.685 | 0.685 | 3/4/1 | 5 |
| pooled (held-out + development) | entmax-1.5 vs fixed | -0.53 | 0.110 | 0.330 | 1/4/3 | 5 |
| pooled (held-out + development) | entmax-1.5 vs Jenks | -0.84 | 0.191 | 0.382 | 1/3/9 | 0 |

Only the **held-out** rows support claims. On the development set the configuration was selected on the test split by fixed-mode score, which favours fixed.

## Per-dataset primary utility (Δ vs fixed, positive = better)

| dataset | subset | Jenks Δ% | outcome | entmax Δ% | outcome | entmax vs Jenks |
|---|---|---|---|---|---|---|
| cardio | held-out | +0.00 | tie | -0.22 | loss | -0.23 (loss) |
| default | held-out | -0.13 | tie | -0.67 | loss | -0.54 (loss) |
| house | held-out | +4.52 | win | +5.39 | win | +0.91 (tie) |
| king | held-out | +9.42 | win | -9.27 | tie | -20.63 (loss) |
| abalone | development | -23.45 | n/a (fixed has no per-seed utility) | +0.75 | n/a (fixed has no per-seed utility) | +19.61 (win) |
| adult | development | -0.29 | n/a (fixed has no per-seed utility) | -1.13 | n/a (fixed has no per-seed utility) | -0.84 (loss) |
| buddy | development | +0.50 | n/a (fixed has no per-seed utility) | -0.53 | n/a (fixed has no per-seed utility) | -1.02 (loss) |
| california | development | -0.94 | loss | -3.53 | loss | -2.56 (loss) |
| churn2 | development | +0.06 | n/a (fixed has no per-seed utility) | -1.24 | n/a (fixed has no per-seed utility) | -1.30 (loss) |
| diabetes | development | -1.65 | tie | +0.36 | tie | +2.04 (tie) |
| gesture | development | +0.73 | win | -0.33 | tie | -1.05 (loss) |
| insurance | development | -1.74 | n/a (fixed has no per-seed utility) | -48.53 | n/a (fixed has no per-seed utility) | -46.00 (loss) |
| wilt | development | -0.17 | tie | -0.14 | tie | +0.03 (tie) |

## Privacy as a constraint

Cases where the 95% CI shows a privacy regression (DCR lower, or MIA AUC further from 0.5). "Beyond seed noise" is statistical, not practical: with n=5 and very small seed sd, changes well under 1% of the reference are resolved, so read the relative size too.

- abalone: entmax-1.5 vs fixed, DCR (mean L2): 0.0515 → 0.0486 (Δ=-0.002975, -5.8% of reference, -4.4 sd of reference; Welch t from summary stats)
- abalone: entmax-1.5 vs Jenks, DCR (mean L2): 0.0703 → 0.0486 (Δ=-0.02175, -30.9% of reference, -94.2 sd of reference; bootstrap (gen seeds, independent resampling))
- abalone: entmax-1.5 vs Jenks, DCR (median+label): 0.0677 → 0.0488 (Δ=-0.01894, -28.0% of reference, -51.0 sd of reference; bootstrap (gen seeds, independent resampling))
- adult: Jenks vs fixed, DCR (median+label): 0.2073 → 0.1806 (Δ=-0.02664, -12.9% of reference, -69.0 sd of reference; Welch t from summary stats)
- buddy: Jenks vs fixed, DCR (mean L2): 0.2595 → 0.2475 (Δ=-0.01201, -4.6% of reference, -15.1 sd of reference; Welch t from summary stats)
- buddy: Jenks vs fixed, DCR (median+label): 0.2045 → 0.1763 (Δ=-0.02822, -13.8% of reference, -32.6 sd of reference; Welch t from summary stats)
- buddy: entmax-1.5 vs fixed, |MIA AUC - 0.5|: 0.0026 → 0.0033 (Δ=-0.0007141, -0.3 sd of reference; Welch t from summary stats (raw AUC))
- default: Jenks vs fixed, DCR (mean L2): 0.2685 → 0.2277 (Δ=-0.04081, -15.2% of reference, -5.7 sd of reference; bootstrap (gen seeds, independent resampling))
- default: Jenks vs fixed, DCR (median+label): 0.1004 → 0.0889 (Δ=-0.01148, -11.4% of reference, -30.3 sd of reference; bootstrap (gen seeds, independent resampling))
- default: Jenks vs fixed, |MIA AUC - 0.5|: 0.0044 → 0.0081 (Δ=-0.003689, -1.8 sd of reference; bootstrap (gen seeds, independent resampling))
- default: entmax-1.5 vs fixed, |MIA AUC - 0.5|: 0.0044 → 0.0072 (Δ=-0.002833, -1.4 sd of reference; bootstrap (gen seeds, independent resampling))
- diabetes: entmax-1.5 vs Jenks, DCR (mean L2): 0.2036 → 0.1935 (Δ=-0.01012, -5.0% of reference, -1.7 sd of reference; bootstrap (gen seeds, independent resampling))
- gesture: Jenks vs fixed, DCR (mean L2): 0.1104 → 0.1100 (Δ=-0.0004607, -0.4% of reference, -1.6 sd of reference; bootstrap (gen seeds, independent resampling))
- gesture: Jenks vs fixed, DCR (median+label): 0.0862 → 0.0851 (Δ=-0.001142, -1.3% of reference, -2.5 sd of reference; bootstrap (gen seeds, independent resampling))
- house: Jenks vs fixed, DCR (mean L2): 0.1035 → 0.1030 (Δ=-0.0004785, -0.5% of reference, -3.7 sd of reference; bootstrap (gen seeds, independent resampling))
- house: Jenks vs fixed, DCR (median+label): 0.0953 → 0.0951 (Δ=-0.0001529, -0.2% of reference, -12.4 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs fixed, DCR (mean L2): 0.1035 → 0.1023 (Δ=-0.001214, -1.2% of reference, -9.5 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs fixed, DCR (median+label): 0.0953 → 0.0945 (Δ=-0.000811, -0.9% of reference, -65.6 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs Jenks, DCR (mean L2): 0.1030 → 0.1023 (Δ=-0.0007355, -0.7% of reference, -4.0 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs Jenks, DCR (median+label): 0.0951 → 0.0945 (Δ=-0.000658, -0.7% of reference, -8.4 sd of reference; bootstrap (gen seeds, independent resampling))
- king: Jenks vs fixed, DCR (mean L2): 0.2223 → 0.2203 (Δ=-0.001987, -0.9% of reference, -2.4 sd of reference; bootstrap (gen seeds, independent resampling))
- king: Jenks vs fixed, DCR (median+label): 0.2011 → 0.1950 (Δ=-0.006047, -3.0% of reference, -10.2 sd of reference; bootstrap (gen seeds, independent resampling))
- wilt: Jenks vs fixed, DCR (mean L2): 0.0217 → 0.0214 (Δ=-0.0003426, -1.6% of reference, -1.7 sd of reference; bootstrap (gen seeds, independent resampling))
- wilt: entmax-1.5 vs fixed, DCR (mean L2): 0.0217 → 0.0213 (Δ=-0.0004414, -2.0% of reference, -2.2 sd of reference; bootstrap (gen seeds, independent resampling))

Fidelity (Wasserstein) W/T/L vs fixed: Jenks 3/5/5; entmax-1.5 5/3/5.

## Mechanism: effective neighbours per node (generation phase)

| dataset | N nodes | fixed k | Jenks k_eff | entmax k_eff |
|---|---|---|---|---|
| abalone | 9 | 3 | 1.97 | 4.75 |
| adult | 14 | 5 | 7.20 | 1.29 |
| buddy | 9 | 5 | 3.32 | 1.05 |
| california | 9 | 5 | 2.58 | 2.82 |
| cardio | 11 | 5 | 4.37 | 1.70 |
| churn2 | 11 | 5 | 5.58 | 2.39 |
| default | 23 | 5 | 8.93 | 1.13 |
| diabetes | 8 | 5 | 2.68 | 6.02 |
| gesture | 32 | 5 | 11.79 | 7.96 |
| house | 17 | 5 | 6.58 | 12.65 |
| insurance | 7 | 5 | 3.19 | 1.75 |
| king | 21 | 5 | 8.98 | 6.80 |
| wilt | 5 | 3 | 1.99 | 4.00 |

- Jenks: fewer neighbours than fixed on 7/13 datasets; median k_eff/k_fixed = 0.87.

- entmax-1.5: fewer neighbours than fixed on 7/13 datasets; median k_eff/k_fixed = 0.56.

k_eff = mean over generation-time calls, blocks and timestep buckets of the off-diagonal neighbours kept per node, weighted by sample count. Fixed k = min(top_k, N−1).

### Jenks acceptance test (counterfactual only)

What the code does:

- `adaptive_adjacency.py:38` — the 2-class split minimises the within-class sum of squared deviations of the sorted raw scores
- `adaptive_adjacency.py:43` — k = size of the strong class
- `adaptive_adjacency.py:47` — the acceptance test ("gate"): accept only if the strong-class mean raw score is > 0, i.e. its sigmoid edge weight is > 0.5
- `adaptive_adjacency.py:53` — on rejection the row falls back to **k_min** (not to the fixed top_k)
- `tab_ddpm/graph_builder.py:305` — the model's call site switches the test off: every split is accepted and k_min never binds
- `tab_ddpm/graph_builder.py:333` — `jenks_gate_stats.csv` logs the share of rows where the test WOULD have fired

**These runs used plain Jenks: no split was ever rejected.** The acceptance test was switched off in commit 3b8b8b3 (2026-09-23 21:40 UTC), which also introduced `jenks_gate_stats.csv`. All 14 local Jenks runs started after it (earliest 2026-09-24T05:15:41Z), and every one has that file, which only the post-switch code writes. So `jenks_gate_stats.csv` does not record accept/reject decisions. It records how often the dropped test *would* have rejected the split. The k of those rows is not logged separately, so "k when accepted vs rejected" cannot be computed from these files.

| dataset | subset | fixed k | Jenks k_eff | would-reject share, generation | would-reject share, last 20% of training | Jenks vs fixed (score) |
|---|---|---|---|---|---|---|
| cardio | held-out | 5 | 4.37 | 11.0% | 8.4% | +0.00% (tie) |
| default | held-out | 5 | 8.93 | 2.5% | 2.5% | -0.13% (tie) |
| house | held-out | 5 | 6.58 | 84.3% | 84.0% | +4.52% (win) |
| king | held-out | 5 | 8.98 | 19.5% | 19.6% | +9.42% (win) |
| abalone | development | 3 | 1.97 | 9.6% | 9.9% | -23.45% (no per-seed fixed) |
| adult | development | 5 | 7.20 | 20.6% | 20.7% | -0.29% (no per-seed fixed) |
| buddy | development | 5 | 3.32 | 8.0% | 8.0% | +0.50% (no per-seed fixed) |
| california | development | 5 | 2.58 | 51.1% | 51.3% | -0.94% (loss) |
| churn2 | development | 5 | 5.58 | 32.4% | 32.3% | +0.06% (no per-seed fixed) |
| diabetes | development | 5 | 2.68 | 91.0% | 92.7% | -1.65% (tie) |
| gesture | development | 5 | 11.79 | 99.7% | 99.7% | +0.73% (win) |
| insurance | development | 5 | 3.19 | 35.3% | 35.4% | -1.74% (no per-seed fixed) |
| wilt | development | 3 | 1.99 | 56.7% | 57.3% | -0.17% (tie) |

Held-out, descriptive: would-reject share at generation by Jenks-vs-fixed outcome — tie: cardio 11.0%, default 2.5%; win: house 84.3%, king 19.5%.
Across all 13 Jenks runs, the would-reject share vs k_eff/k_fixed: Spearman ρ=-0.08 (p=0.789), descriptive only.

The question "do Jenks ties coincide with the gate rejecting?" has no gate to test here. What can be said is whether the rows that a gate would have rejected (weak strong class) are common where Jenks ties or gains, and the table above gives that. The share is a property of the learned scores under plain Jenks. It does not predict what a gated run would have done, because gating changes the adjacency and therefore training.

## Δ utility vs dependency-graph meta-features (exploratory)

| rule | predictor | n | Spearman ρ | p | Holm p (8 tests) |
|---|---|---|---|---|---|
| Jenks | assoc_density_0.3 | 13 | -0.13 | 0.661 | 1.000 |
| Jenks | assoc_assoc_mean | 13 | -0.20 | 0.505 | 1.000 |
| Jenks | n_features | 13 | +0.66 | 0.014 | 0.111 |
| Jenks | num_cat_ratio | 13 | +0.13 | 0.667 | 1.000 |
| entmax-1.5 | assoc_density_0.3 | 13 | +0.40 | 0.174 | 1.000 |
| entmax-1.5 | assoc_assoc_mean | 13 | +0.42 | 0.156 | 1.000 |
| entmax-1.5 | n_features | 13 | -0.06 | 0.836 | 1.000 |
| entmax-1.5 | num_cat_ratio | 13 | +0.35 | 0.234 | 1.000 |

Exploratory only: n is the number of datasets (small), no causal reading; development-set datasets enter with a fixed mean that is in-sample with respect to selection (optimistic), and held-out alone is too small (n=4) for a correlation. `assoc_*` (rank-based specification matrix: |Spearman|, Cramér's V, correlation ratio) is the predictor. `mg_*` (the repo's static `build_static_adjacency` graph: |Pearson| / Cramér's V / normalised MI, pre-thresholded at 0.1) is reported only descriptively below: it is a second encoding of the same data dependencies, so testing it too would double the number of exploratory tests without adding an independent variable. Note that it is *not* produced by the top-k rule — the runs use `graph_mode="dynamic"`, whose learned adjacency is what the rules sparsify, and neither meta-feature reads it.

| dataset | assoc_density_0.3 | mg_density_0.3 | mg_n_nodes |
|---|---|---|---|
| abalone | 1.000 | 1.000 | 9 |
| adult | 0.132 | 0.231 | 14 |
| buddy | 0.222 | 0.417 | 9 |
| california | 0.083 | 0.361 | 9 |
| cardio | 0.109 | 0.109 | 11 |
| churn2 | 0.055 | 0.055 | 11 |
| default | 0.478 | 0.617 | 23 |
| diabetes | 0.143 | 0.643 | 8 |
| fb-comments | 0.321 | 0.443 | 52 |
| gesture | 0.081 | 0.190 | 32 |
| house | 0.272 | 0.434 | 17 |
| insurance | 0.143 | 0.190 | 7 |
| king | 0.243 | 0.448 | 21 |
| wilt | 0.300 | 0.600 | 5 |

## Secondary utility (TabDDPM convention: macro-F1 / R²)

- held-out evaluation (transferred) (n=4): avg ranks fixed 2.25, Jenks 1.75, entmax-1.5 2.00; Friedman p=0.7788; Jenks vs fixed W/T/L 3/1/0; entmax-1.5 vs fixed W/T/L 2/1/1; entmax-1.5 vs Jenks W/T/L 1/1/2.
- development set (ablated) (n=9): avg ranks fixed 1.33, Jenks 2.22, entmax-1.5 2.44; Friedman p=0.0446; Jenks vs fixed W/T/L 1/2/1; entmax-1.5 vs fixed W/T/L 0/3/1; entmax-1.5 vs Jenks W/T/L 1/3/5.
- pooled (held-out + development) (n=13): avg ranks fixed 1.62, Jenks 2.08, entmax-1.5 2.31; Friedman p=0.1988; Jenks vs fixed W/T/L 4/3/1; entmax-1.5 vs fixed W/T/L 2/4/2; entmax-1.5 vs Jenks W/T/L 2/4/7.

Class imbalance (majority/minority ratio) is shown per dataset in `tables/utility_secondary.md`; accuracy is least informative where it is large (wilt 17.5, buddy 6.1, churn2 3.9, default 3.5, adult 3.2).

## Caveats for the thesis text

1. **Actual configuration-selection rule.** The ablated datasets' Jenks/entmax (and hence fixed) configuration is the argmax of `score.mean` over the fixed grid, via `best_manifest.py`:

   > "Best" here is the repository's own definition --
   > ``ablation/existing_results.load_best()``, which ranks by ``metrics.score.mean``

   It differs from the `config_id` in `hpc_transfer/best_per_dataset.csv` / `predictions.md` (composite rank, ties broken on parameter count) in **7/9** ablated datasets (abalone, buddy, california, churn2, diabetes, insurance, wilt). Describe the argmax-score rule in the text.
2. **The selection score is the test-split score.** `load_best` ranks on `metrics.score.mean`, and that value is the mean of the CatBoost metrics read from the *test* part of each evaluation report:

   - `hpc_transfer/best_manifest.py:58` — the manifest takes `load_best`'s winner
   - `ablation/existing_results.py:29` — `load_best` reads the fixed-grid JSONs
   - `ablation/existing_results.py:113` — the ranked quantity is `metrics.score`
   - `ablation/existing_results.py:120` — argmax of `score.mean`
   - `ablation/ablation_runner.py:764` — every CatBoost metric, `score` included, is read from the **test** block
   - `ablation/ablation_runner.py:791` — those test values are averaged into `results_full_averaged.json`
   - `ablation/ablation_runner.py:878` — the grid JSON entry copies that file verbatim

   None of the 9 fixed-grid JSONs stores a validation metric. So the development-set configuration was chosen on the same split its results are reported on: those results are in-sample and optimistic (most for fixed), and are not comparable with the TabDDPM paper, which tuned on the validation split. A validation-based re-selection needs per-configuration validation metrics (e.g. per-run `results_catboost.json` on the cluster), if they still exist.
3. **Transferred datasets ran the scaled alternative, not the LOO-validated primary.** Configurations present in the results: `gnn_gatv2_layers3_dim128_topk5_heads8_noattn`. The LOO-validated global best is `gnn_gatv2_layers3_dim64_topk3_heads4_noattn`. `predictions.md` gives the reason for the scaled job:

   > `d64` is therefore a ceiling of the shared abalone/diabetes/insurance sub-grid, not a measured optimum — and all seven targets are larger than every donor. Each target therefore gets **two jobs**: the LOO-validated configuration, and one size-scaled alternative.

   > | scaled alternative | `gnn_gatv2_layers3_dim128_topk5_heads8_noattn` | measured best on churn2 and inside california's 1σ tie set — the two largest donors; the `d_model`-ceiling result says the targets, all larger still, should not be run at `d64` alone |

   `predictions.md` plans *both* jobs per target; only the scaled one exists in the results tree, so no transferred result speaks for the LOO-validated configuration.
4. **n = 5 generation seeds of one trained model per cell.** Training-seed variance is not in any CI; bootstrap CIs with n=5 are narrow and the Welch tests have low power. Treat per-dataset W/T/L as descriptive.
5. **Seed winner's curse on the development set.** The fixed number is the maximum over the dataset's grid (up to 24 configurations) at training seed 0, so fixed is favoured. The per-seed rerun uses the same configuration and the same training seed 0: it adds per-generation-seed values (CIs, win/tie/loss) but removes neither this bias nor the test-split selection bias (caveat 2). The development set stays in-sample regardless.
6. **Missing datasets:** fb-comments — excluded from every across-dataset test.
7. **Units.** Across-dataset Wilcoxon and the meta-feature correlations use Δ as % of the reference mean, because regression RMSE (e.g. insurance ≈ 4.7k) and accuracy are not on one scale. MIA is scored as |mean AUC − 0.5|.
8. **Two DCR variants.** `dcr_tabddpm_mean` (mean L2, no label) is the only one recorded for every run; `dcr_baseline` (median, label included) is missing from some seed-0 fixed grids. Only `dcr_baseline` uses the same recipe as the TabDDPM baseline `privacy.json`, so only it may be compared with those numbers.
9. **The hpc_density README's Jenks densities describe the GATED variant.** Its "Jenks ~1–2.4 neighbours" line was written in de700e6 (2026-09-22), when the call site still passed `use_gate_rule=True` (also in 3394bd8, the code shipped for the earlier ARIS Jenks runs). Under the gate, every rejected row falls back to k_min = 1, which pulls the mean budget towards 1. All Jenks runs here are plain Jenks (see 'Jenks acceptance test'), so their k_eff (2.0–11.8) is not comparable with the README figure; do not quote the README for these runs.
10. **Categorical-input bug (all runs before the fix).** The GNN denoiser embedded each categorical column from the diffusion's log one-hot (entries ≈ −69) through its own `Linear(K, d_model)`, which makes categorical node embeddings ~200× the timestep embedding at initialization; some categorical columns then learn to copy their noisy input instead of denoising it (multinomial loss at or above an always-uniform predictor on cardio, churn2, abalone). It affects every rule the same way on a given dataset, but all absolute numbers on datasets with categorical columns, marked §. The fix (`cat_input="prob"`) is rerun into `*_catfix` trees; evidence and the list of affected runs are in `analysis/dependency_fidelity/REPORT.md` ('Categorical-input bug').
