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
- **Wasserstein and MIA AUC of 24 cells are recomputed post hoc** (`analysis/integrity/posthoc_gen_metrics.csv`): `hpc_transfer/split_runner.py` called both functions without `dataset_name`; the TypeError was caught and only logged, so the `*_catfix` shards hold utility and DCR only. The functions are deterministic and the recomputation reproduces stored values of non-split runs to 1e-10 (`backfill_gen_metrics.py`).
- Configuration check: no mismatches (gnn type, layers, d_model, top_k, heads, attention identical across the three rules).

- **Categorical-input fix:** datasets on the fixed (`*_catfix`) runs: abalone, adult, buddy, cardio, churn2, default, insurance, king. Datasets with categorical columns still on the original runs, marked § in the tables: fb-comments. A dataset switches only when all three rules are complete in `*_catfix`; the two are never mixed. See caveat 10.

## Key findings

- **Held-out evaluation (transferred, n=4) — primary:** average rank fixed 2.50, Jenks 2.25, entmax-1.5 1.25; Friedman p=0.174. W/T/L: Jenks vs fixed 1/3/0 (median Δ +0.02%); entmax-1.5 vs fixed 2/2/0 (median Δ +2.73%); entmax-1.5 vs Jenks 2/2/0 (median Δ +0.50%). With n=4 no across-dataset test can reach significance (smallest exact two-sided Wilcoxon p is 0.125).
- **Development set (ablated, n=9) — in-sample, optimistic for fixed:** average rank fixed 1.44, Jenks 2.11, entmax-1.5 2.44; Friedman p=0.0970. entmax-1.5 vs Jenks W/T/L 0/6/3 (median Δ -0.04%).
- **Pooled (n=13; inherits the development-set bias, completeness only):** average rank fixed 1.77, Jenks 2.15, entmax-1.5 2.08; Friedman p=0.5836.
- **Large utility changes vs fixed (|Δ| > 5%):** house entmax-1.5 +5.4% [held-out, win]; king entmax-1.5 +6.5% [held-out, tie].
- **Privacy vs fixed:** 17 resolved regressions across 7 datasets (9 on held-out: default, house, king); the largest relative DCR drops are default entmax-1.5 -5.5%; default Jenks -4.0%; king entmax-1.5 -2.4%. See 'Privacy as a constraint'.
- **Mechanism:** median k_eff/k_fixed is Jenks 0.75, entmax-1.5 1.02; both rules keep fewer neighbours than fixed on some datasets and more on others (table below), so neither is uniformly sparser.
- **Meta-features (exploratory):** strongest association is entmax-1.5 Δ vs assoc_density_0.3 (ρ=+0.33, p=0.271, Holm p=1.000, n=13). assoc_density_0.3: Jenks ρ=+0.19 (p=0.535), entmax-1.5 ρ=+0.33 (p=0.271).

## Primary utility across datasets (score = accuracy / −RMSE)

| subset | role | n | avg rank fixed | avg rank Jenks | avg rank entmax | Friedman p |
|---|---|---|---|---|---|---|
| held-out evaluation (transferred) | **primary** | 4 | 2.50 | 2.25 | 1.25 | 0.1738 |
| development set (ablated) | in-sample, optimistic | 9 | 1.44 | 2.11 | 2.44 | 0.0970 |
| pooled (held-out + development) | completeness only | 13 | 1.77 | 2.15 | 2.08 | 0.5836 |

Pairwise Wilcoxon signed-rank on the relative Δ (% of the reference mean), Holm-corrected within each subset; W/T/L from per-dataset 95% CIs:

| subset | pair | median Δ% | Wilcoxon p | Holm p | W/T/L | not testable |
|---|---|---|---|---|---|---|
| held-out evaluation (transferred) | Jenks vs fixed | +0.02 | 0.875 | 0.875 | 1/3/0 | 0 |
| held-out evaluation (transferred) | entmax-1.5 vs fixed | +2.73 | 0.125 | 0.375 | 2/2/0 | 0 |
| held-out evaluation (transferred) | entmax-1.5 vs Jenks | +0.50 | 0.250 | 0.500 | 2/2/0 | 0 |
| development set (ablated) | Jenks vs fixed | -0.12 | 0.652 | 0.852 | 2/5/2 | 0 |
| development set (ablated) | entmax-1.5 vs fixed | -0.16 | 0.098 | 0.293 | 0/7/2 | 0 |
| development set (ablated) | entmax-1.5 vs Jenks | -0.04 | 0.426 | 0.852 | 0/6/3 | 0 |
| pooled (held-out + development) | Jenks vs fixed | -0.06 | 0.685 | 1.000 | 3/8/2 | 0 |
| pooled (held-out + development) | entmax-1.5 vs fixed | -0.03 | 0.735 | 1.000 | 2/9/2 | 0 |
| pooled (held-out + development) | entmax-1.5 vs Jenks | +0.03 | 0.946 | 1.000 | 2/8/3 | 0 |

Only the **held-out** rows support claims. On the development set the configuration was selected on the test split by fixed-mode score, which favours fixed.

## Per-dataset primary utility (Δ vs fixed, positive = better)

| dataset | subset | Jenks Δ% | outcome | entmax Δ% | outcome | entmax vs Jenks |
|---|---|---|---|---|---|---|
| cardio | held-out | -0.01 | tie | +0.07 | win | +0.09 (win) |
| default | held-out | +0.06 | tie | +0.00 | tie | -0.06 (tie) |
| house | held-out | +4.52 | win | +5.39 | win | +0.91 (tie) |
| king | held-out | -1.81 | tie | +6.45 | tie | +8.11 (win) |
| abalone | development | +0.43 | tie | -0.17 | tie | -0.60 (tie) |
| adult | development | -0.12 | loss | -0.16 | loss | -0.04 (tie) |
| buddy | development | -0.06 | tie | -0.03 | tie | +0.04 (tie) |
| california | development | -0.94 | loss | -3.53 | loss | -2.56 (loss) |
| churn2 | development | -0.19 | tie | -0.03 | tie | +0.16 (tie) |
| diabetes | development | -1.65 | tie | +0.36 | tie | +2.04 (tie) |
| gesture | development | +0.73 | win | -0.33 | tie | -1.05 (loss) |
| insurance | development | +0.59 | win | -0.21 | tie | -0.80 (loss) |
| wilt | development | -0.17 | tie | -0.14 | tie | +0.03 (tie) |

## Privacy as a constraint

Cases where the 95% CI shows a privacy regression (DCR lower, or MIA AUC further from 0.5). "Beyond seed noise" is statistical, not practical: with n=5 and very small seed sd, changes well under 1% of the reference are resolved, so read the relative size too.

- abalone: Jenks vs fixed, DCR (median+label): 0.0499 → 0.0496 (Δ=-0.0003244, -0.7% of reference, -2.5 sd of reference; bootstrap (gen seeds, independent resampling))
- abalone: entmax-1.5 vs fixed, DCR (median+label): 0.0499 → 0.0491 (Δ=-0.0008074, -1.6% of reference, -6.2 sd of reference; bootstrap (gen seeds, independent resampling))
- abalone: entmax-1.5 vs Jenks, DCR (median+label): 0.0496 → 0.0491 (Δ=-0.000483, -1.0% of reference, -2.7 sd of reference; bootstrap (gen seeds, independent resampling))
- cardio: entmax-1.5 vs Jenks, DCR (mean L2): 0.0164 → 0.0162 (Δ=-0.0002223, -1.4% of reference, -5.4 sd of reference; bootstrap (gen seeds, independent resampling))
- cardio: entmax-1.5 vs Jenks, DCR (median+label): 0.0116 → 0.0116 (Δ=-3.008e-05, -0.3% of reference, -1.4 sd of reference; bootstrap (gen seeds, independent resampling))
- default: Jenks vs fixed, DCR (median+label): 0.0849 → 0.0815 (Δ=-0.00342, -4.0% of reference, -17.0 sd of reference; bootstrap (gen seeds, independent resampling))
- default: entmax-1.5 vs fixed, DCR (median+label): 0.0849 → 0.0802 (Δ=-0.004703, -5.5% of reference, -23.4 sd of reference; bootstrap (gen seeds, independent resampling))
- default: entmax-1.5 vs Jenks, DCR (mean L2): 0.2813 → 0.2160 (Δ=-0.06526, -23.2% of reference, -0.8 sd of reference; bootstrap (gen seeds, independent resampling))
- default: entmax-1.5 vs Jenks, DCR (median+label): 0.0815 → 0.0802 (Δ=-0.001283, -1.6% of reference, -4.9 sd of reference; bootstrap (gen seeds, independent resampling))
- diabetes: entmax-1.5 vs Jenks, DCR (mean L2): 0.2036 → 0.1935 (Δ=-0.01012, -5.0% of reference, -1.7 sd of reference; bootstrap (gen seeds, independent resampling))
- gesture: Jenks vs fixed, DCR (mean L2): 0.1104 → 0.1100 (Δ=-0.0004607, -0.4% of reference, -1.6 sd of reference; bootstrap (gen seeds, independent resampling))
- gesture: Jenks vs fixed, DCR (median+label): 0.0862 → 0.0851 (Δ=-0.001142, -1.3% of reference, -2.5 sd of reference; bootstrap (gen seeds, independent resampling))
- house: Jenks vs fixed, DCR (mean L2): 0.1035 → 0.1030 (Δ=-0.0004785, -0.5% of reference, -3.7 sd of reference; bootstrap (gen seeds, independent resampling))
- house: Jenks vs fixed, DCR (median+label): 0.0953 → 0.0951 (Δ=-0.0001529, -0.2% of reference, -12.4 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs fixed, DCR (mean L2): 0.1035 → 0.1023 (Δ=-0.001214, -1.2% of reference, -9.5 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs fixed, DCR (median+label): 0.0953 → 0.0945 (Δ=-0.000811, -0.9% of reference, -65.6 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs Jenks, DCR (mean L2): 0.1030 → 0.1023 (Δ=-0.0007355, -0.7% of reference, -4.0 sd of reference; bootstrap (gen seeds, independent resampling))
- house: entmax-1.5 vs Jenks, DCR (median+label): 0.0951 → 0.0945 (Δ=-0.000658, -0.7% of reference, -8.4 sd of reference; bootstrap (gen seeds, independent resampling))
- insurance: entmax-1.5 vs fixed, DCR (mean L2): 0.0919 → 0.0905 (Δ=-0.001471, -1.6% of reference, -2.2 sd of reference; bootstrap (gen seeds, independent resampling))
- insurance: entmax-1.5 vs fixed, DCR (median+label): 0.0872 → 0.0856 (Δ=-0.001569, -1.8% of reference, -2.1 sd of reference; bootstrap (gen seeds, independent resampling))
- insurance: entmax-1.5 vs Jenks, DCR (mean L2): 0.0939 → 0.0905 (Δ=-0.003424, -3.6% of reference, -3.1 sd of reference; bootstrap (gen seeds, independent resampling))
- insurance: entmax-1.5 vs Jenks, DCR (median+label): 0.0886 → 0.0856 (Δ=-0.003037, -3.4% of reference, -2.2 sd of reference; bootstrap (gen seeds, independent resampling))
- king: Jenks vs fixed, DCR (median+label): 0.1992 → 0.1959 (Δ=-0.00322, -1.6% of reference, -5.9 sd of reference; bootstrap (gen seeds, independent resampling))
- king: entmax-1.5 vs fixed, DCR (mean L2): 0.2212 → 0.2175 (Δ=-0.00368, -1.7% of reference, -7.3 sd of reference; bootstrap (gen seeds, independent resampling))
- king: entmax-1.5 vs fixed, DCR (median+label): 0.1992 → 0.1944 (Δ=-0.004717, -2.4% of reference, -8.7 sd of reference; bootstrap (gen seeds, independent resampling))
- king: entmax-1.5 vs Jenks, DCR (mean L2): 0.2211 → 0.2175 (Δ=-0.003644, -1.6% of reference, -4.5 sd of reference; bootstrap (gen seeds, independent resampling))
- king: entmax-1.5 vs Jenks, DCR (median+label): 0.1959 → 0.1944 (Δ=-0.001497, -0.8% of reference, -3.0 sd of reference; bootstrap (gen seeds, independent resampling))
- wilt: Jenks vs fixed, DCR (mean L2): 0.0217 → 0.0214 (Δ=-0.0003426, -1.6% of reference, -1.7 sd of reference; bootstrap (gen seeds, independent resampling))
- wilt: entmax-1.5 vs fixed, DCR (mean L2): 0.0217 → 0.0213 (Δ=-0.0004414, -2.0% of reference, -2.2 sd of reference; bootstrap (gen seeds, independent resampling))

Fidelity (Wasserstein) W/T/L vs fixed: Jenks 5/5/3; entmax-1.5 6/5/2.

## Mechanism: effective neighbours per node (generation phase)

| dataset | N nodes | fixed k | Jenks k_eff | entmax k_eff |
|---|---|---|---|---|
| abalone | 9 | 3 | 2.73 | 5.87 |
| adult | 14 | 5 | 3.56 | 1.28 |
| buddy | 9 | 5 | 2.56 | 1.91 |
| california | 9 | 5 | 2.58 | 2.82 |
| cardio | 11 | 5 | 3.75 | 4.16 |
| churn2 | 11 | 5 | 4.01 | 5.10 |
| default | 23 | 5 | 5.57 | 4.46 |
| diabetes | 8 | 5 | 2.68 | 6.02 |
| gesture | 32 | 5 | 11.79 | 7.96 |
| house | 17 | 5 | 6.58 | 12.65 |
| insurance | 7 | 5 | 2.53 | 5.09 |
| king | 21 | 5 | 9.02 | 8.80 |
| wilt | 5 | 3 | 1.99 | 4.00 |

- Jenks: fewer neighbours than fixed on 9/13 datasets; median k_eff/k_fixed = 0.75.

- entmax-1.5: fewer neighbours than fixed on 5/13 datasets; median k_eff/k_fixed = 1.02.

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
| cardio | held-out | 5 | 3.75 | 81.0% | 80.9% | -0.01% (tie) |
| default | held-out | 5 | 5.57 | 95.7% | 95.6% | +0.06% (tie) |
| house | held-out | 5 | 6.58 | 84.3% | 84.0% | +4.52% (win) |
| king | held-out | 5 | 9.02 | 88.9% | 89.4% | -1.81% (tie) |
| abalone | development | 3 | 2.73 | 66.7% | 67.1% | +0.43% (tie) |
| adult | development | 5 | 3.56 | 75.8% | 76.0% | -0.12% (loss) |
| buddy | development | 5 | 2.56 | 61.6% | 61.6% | -0.06% (tie) |
| california | development | 5 | 2.58 | 51.1% | 51.3% | -0.94% (loss) |
| churn2 | development | 5 | 4.01 | 77.6% | 78.0% | -0.19% (tie) |
| diabetes | development | 5 | 2.68 | 91.0% | 92.7% | -1.65% (tie) |
| gesture | development | 5 | 11.79 | 99.7% | 99.7% | +0.73% (win) |
| insurance | development | 5 | 2.53 | 49.7% | 49.8% | +0.59% (win) |
| wilt | development | 3 | 1.99 | 56.7% | 57.3% | -0.17% (tie) |

Held-out, descriptive: would-reject share at generation by Jenks-vs-fixed outcome — tie: cardio 81.0%, default 95.7%, king 88.9%; win: house 84.3%.
Across all 13 Jenks runs, the would-reject share vs k_eff/k_fixed: Spearman ρ=+0.76 (p=0.003), descriptive only.

The question "do Jenks ties coincide with the gate rejecting?" has no gate to test here. What can be said is whether the rows that a gate would have rejected (weak strong class) are common where Jenks ties or gains, and the table above gives that. The share is a property of the learned scores under plain Jenks. It does not predict what a gated run would have done, because gating changes the adjacency and therefore training.

## Δ utility vs dependency-graph meta-features (exploratory)

| rule | predictor | n | Spearman ρ | p | Holm p (8 tests) |
|---|---|---|---|---|---|
| Jenks | assoc_density_0.3 | 13 | +0.19 | 0.535 | 1.000 |
| Jenks | assoc_assoc_mean | 13 | +0.00 | 1.000 | 1.000 |
| Jenks | n_features | 13 | +0.25 | 0.412 | 1.000 |
| Jenks | num_cat_ratio | 13 | +0.12 | 0.694 | 1.000 |
| entmax-1.5 | assoc_density_0.3 | 13 | +0.33 | 0.271 | 1.000 |
| entmax-1.5 | assoc_assoc_mean | 13 | +0.26 | 0.394 | 1.000 |
| entmax-1.5 | n_features | 13 | +0.24 | 0.439 | 1.000 |
| entmax-1.5 | num_cat_ratio | 13 | -0.10 | 0.754 | 1.000 |

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

- held-out evaluation (transferred) (n=4): avg ranks fixed 2.50, Jenks 2.25, entmax-1.5 1.25; Friedman p=0.1738; Jenks vs fixed W/T/L 2/2/0; entmax-1.5 vs fixed W/T/L 3/1/0; entmax-1.5 vs Jenks W/T/L 2/1/1.
- development set (ablated) (n=9): avg ranks fixed 1.56, Jenks 2.11, entmax-1.5 2.33; Friedman p=0.2359; Jenks vs fixed W/T/L 2/6/1; entmax-1.5 vs fixed W/T/L 0/7/2; entmax-1.5 vs Jenks W/T/L 0/7/2.
- pooled (held-out + development) (n=13): avg ranks fixed 1.85, Jenks 2.15, entmax-1.5 2.00; Friedman p=0.7351; Jenks vs fixed W/T/L 4/8/1; entmax-1.5 vs fixed W/T/L 3/8/2; entmax-1.5 vs Jenks W/T/L 2/8/3.

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
