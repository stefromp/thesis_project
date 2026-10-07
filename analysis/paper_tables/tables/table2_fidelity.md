**Column Shapes error (%) ↓**

| Method | AB | AD | BU | CA | CAR | CH | DE | DI | FB | GE | HI | HO | IN | KI | MI | WI | Rank (n=13) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Real test* | 2.70 | 0.64 | 1.64 | 1.42 | 0.54 | 1.49 | 0.98 | 7.74 | — | 1.98 | — | 1.51 | 3.26 | 1.31 | — | 2.94 |  |
| GNN-fixed | 3.07 ±.33 | **0.81** ±.04 | 1.54 ±.27 | **1.57** ±.17 | 0.59 ±.11 | <u>1.42</u> ±.10 | 7.19 ±.15 | 7.80 ±.58 | — | 2.91 ±.34 | — | 1.68 ±.10 | 4.54 ±1.31 | 1.86 ±.12 | — | **2.53** ±.39 | 2.46 |
| GNN-Jenks | <u>3.00</u> ±.33 | <u>0.82</u> ±.07 | <u>1.52</u> ±.22 | <u>1.62</u> ±.13 | **0.56** ±.08 | **1.41** ±.13 | <u>6.76</u> ±.15 | **7.34** ±.74 | — | **2.55** ±.28 | — | <u>1.67</u> ±.06 | <u>4.20</u> ±.85 | <u>1.73</u> ±.12 | — | <u>2.55</u> ±.40 | 1.73 |
| GNN-entmax | **2.85** ±.43 | 0.95 ±.04 | **1.47** ±.33 | <u>1.62</u> ±.16 | <u>0.57</u> ±.09 | 1.46 ±.07 | **6.68** ±.18 | <u>7.42</u> ±.56 | — | <u>2.59</u> ±.21 | — | **1.51** ±.05 | **4.14** ±1.07 | **1.71** ±.13 | — | 2.56 ±.44 | 1.81 |

**Column Pair Trends error (%) ↓**

| Method | AB | AD | BU | CA | CAR | CH | DE | DI | FB | GE | HI | HO | IN | KI | MI | WI | Rank (n=13) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Real test* | 1.96 | 1.60 | 4.06 | 0.82 | 1.03 | 2.87 | 1.52 | 4.92 | — | 2.68 | — | 1.10 | 8.12 | 0.98 | — | 2.41 |  |
| GNN-fixed | **3.21** ±.43 | **2.01** ±.02 | 3.90 ±.10 | **0.75** ±.14 | 0.87 ±.34 | <u>2.55</u> ±.11 | 2.41 ±.24 | **5.34** ±.32 | — | 2.08 ±.10 | — | 0.97 ±.13 | 8.12 ±1.03 | <u>1.81</u> ±.08 | — | 2.59 ±1.34 | 2.27 |
| GNN-Jenks | 3.31 ±.61 | <u>2.16</u> ±.10 | <u>3.82</u> ±.09 | 0.82 ±.19 | **0.77** ±.13 | **2.50** ±.16 | <u>2.13</u> ±.16 | 7.25 ±1.06 | — | **1.94** ±.11 | — | <u>0.85</u> ±.12 | **7.50** ±.65 | 2.05 ±.08 | — | <u>2.21</u> ±.86 | 2.00 |
| GNN-entmax | <u>3.30</u> ±.54 | 2.48 ±.02 | **3.81** ±.21 | <u>0.77</u> ±.11 | <u>0.84</u> ±.20 | <u>2.55</u> ±.08 | **1.96** ±.25 | <u>5.88</u> ±.52 | — | <u>2.07</u> ±.10 | — | **0.79** ±.11 | <u>7.72</u> ±.39 | **1.68** ±.08 | — | **2.03** ±.62 | 1.73 |

Error = 100 × (1 − score), lower is better; mean ± sd over 5 generation seeds (sd with ddof=1). Column Shapes: SDMetrics per-column similarity (KS complement for numerical, TV complement for categorical columns) averaged over columns. Column Pair Trends: correlation / contingency similarity averaged over ALL column pairs, each pair scored on bins shared between real and synthetic data (not the SDMetrics QualityReport aggregate). Each synthetic sample (and the Real-test reference) is compared with the real train split on the same number of rows. *Real test* = the real test split scored against the real train split (sampling-noise reference). **Bold** best, <u>underline</u> second best among the three GNN variants; Rank = average over the 13 datasets with all three. **These values are computed with our own implementation and splits and are not comparable to the Shape/Trend numbers reported in the TabDiff paper.**
