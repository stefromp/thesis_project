**DCR (median nearest-neighbour L2 to real train) ↑**

| Method | AB | AD | BU | CA | CAR | CH | DE | DI | FB | GE | HI | HO | IN | KI | MI | WI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Holdout* | 0.0490 | 0.1059 | 0.1642 | 0.0408 | 0.0114 | 0.2146 | 0.0743 | 0.1744 | — | 0.0840 | — | 0.0948 | 0.1015 | 0.1752 | — | 0.0166 |
| TabDDPM § | 0.0499 | 0.1044 | 0.1430 | 0.0406 | 0.0122 | 0.1574 | 0.1123 ‡ | 0.2036 | 0.1115 ◇ | 0.0589 | 0.4487 ◇ | 0.0859 | 0.0405 | 0.1887 | 0.0219 ◇ | 0.0164 |
| GNN-fixed | 0.0499 ±.0001 | 0.1163 ±.0003 | 0.1635 ±.0004 | 0.0436 ±.0002 | 0.0116 ±.0000 | 0.2150 ±.0007 | 0.0849 ±.0002 | 0.1758 ±.0032 | — | 0.0862 ±.0005 | — | 0.0953 ±.0000 | 0.0872 ±.0007 | 0.1992 ±.0005 | — | 0.0165 ±.0001 |
| GNN-Jenks | 0.0496 ±.0002 | 0.1223 ±.0005 | 0.1644 ±.0006 | 0.0441 ±.0001 | 0.0116 ±.0000 | 0.2151 ±.0006 | 0.0815 ±.0003 | 0.1808 ±.0034 | — | 0.0851 ±.0004 | — | 0.0951 ±.0001 | 0.0886 ±.0014 | 0.1959 ±.0005 | — | 0.0165 ±.0001 |
| GNN-entmax | 0.0491 ±.0002 | 0.1317 ±.0004 | 0.1652 ±.0005 | 0.0445 ±.0001 | 0.0116 ±.0000 | 0.2148 ±.0006 | 0.0802 ±.0001 | 0.1799 ±.0045 | — | 0.0884 ±.0007 | — | 0.0945 ±.0001 | 0.0856 ±.0008 | 0.1944 ±.0006 | — | 0.0164 ±.0001 |

**DCR / holdout DCR**

| Method | AB | AD | BU | CA | CAR | CH | DE | DI | FB | GE | HI | HO | IN | KI | MI | WI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TabDDPM § | 1.019 | 0.985 | 0.871 | 0.994 | 1.072 | 0.733 | — | 1.167 | — | 0.701 | — | 0.906 | 0.399 | 1.077 | — | 0.987 |
| GNN-fixed | 1.018 ±.003 | 1.098 ±.003 | 0.996 ±.002 | 1.067 ±.004 | 1.018 ±.002 | 1.001 ±.003 | 1.143 ±.003 | 1.008 ±.019 | — | 1.026 ±.005 | — | 1.005 ±.000 | 0.858 ±.007 | 1.136 ±.003 | — | 0.989 ±.004 |
| GNN-Jenks | 1.012 ±.004 | 1.155 ±.005 | 1.001 ±.004 | 1.079 ±.001 | 1.020 ±.002 | 1.002 ±.003 | 1.097 ±.004 | 1.037 ±.020 | — | 1.013 ±.005 | — | 1.004 ±.001 | 0.873 ±.013 | 1.118 ±.003 | — | 0.989 ±.005 |
| GNN-entmax | 1.002 ±.004 | 1.243 ±.003 | 1.006 ±.003 | 1.091 ±.003 | 1.018 ±.002 | 1.001 ±.003 | 1.079 ±.002 | 1.032 ±.026 | — | 1.052 ±.008 | — | 0.997 ±.001 | 0.843 ±.008 | 1.110 ±.003 | — | 0.987 ±.003 |

**MIA AUC (→ 0.5)**

| Method | AB | AD | BU | CA | CAR | CH | DE | DI | FB | GE | HI | HO | IN | KI | MI | WI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GNN-fixed | 0.495 ±.005 | 0.504 ±.001 | 0.508 ±.003 | 0.504 ±.003 | 0.500 ±.002 | 0.506 ±.004 | 0.495 ±.001 | 0.540 ±.016 | — | 0.501 ±.001 | — | 0.508 ±.001 | 0.567 ±.022 | 0.503 ±.001 | — | 0.528 ±.007 |
| GNN-Jenks | 0.497 ±.004 | 0.503 ±.001 | 0.507 ±.003 | 0.505 ±.001 | 0.499 ±.003 | 0.505 ±.004 | 0.496 ±.001 | 0.540 ±.023 | — | 0.501 ±.001 | — | 0.508 ±.002 | 0.550 ±.020 | 0.504 ±.001 | — | 0.531 ±.010 |
| GNN-entmax | 0.496 ±.003 | 0.503 ±.001 | 0.504 ±.003 | 0.503 ±.004 | 0.500 ±.001 | 0.505 ±.005 | 0.497 ±.002 | 0.532 ±.020 | — | 0.501 ±.001 | — | 0.508 ±.001 | 0.564 ±.013 | 0.504 ±.001 | — | 0.530 ±.008 |

DCR: median, over synthetic rows, of the L2 distance to the closest real training row (numerical columns min-max scaled and categorical columns one-hot / √2, both fitted on real train; target included). GNN rows: mean ± sd over 5 generation seeds. § TabDDPM: the authors' exp/<ds>/ddpm_cb_best/privacy.json, the same statistic computed by the same code (scripts/resample_privacy.py `privacy_metrics`, median), on one generated sample, so no sd; that code reproduces our stored GNN DCR to ~1e-12 when run on our samples (`DCR_RECIPE_CHECK.md`). It subsamples the synthetic rows to 50,000 when more were generated (≤0.4% effect on the median, measured on our samples) and the real train rows to 50,000 when more exist: ◇ = real train set subsampled (FB, HI, MI), not comparable to an all-rows DCR. ‡ DE: computed on the authors' version of the dataset, not like-for-like. *Holdout*: the same statistic for the real test split, i.e. the distance a fresh sample from the data distribution has to the training set; DCR / holdout ≈ 1 means synthetic rows are as far from the training rows as unseen real rows, < 1 closer (not computed where ‡ or ◇). MIA AUC: distance-based membership-inference attack (train vs test rows scored by proximity to the synthetic set), 0.5 = no leakage; not available for TabDDPM. No value is highlighted: a larger DCR is not unconditionally better (it can also reflect lower fidelity).
