| Method | MLE (n=7) | MLE w/o CTABGAN (n=12) | Shape (n=13) | Trend (n=13) | DCR (n=12) | median Δ MLE vs TabDDPM | Wilcoxon p / Holm (n=12) |
|---|---|---|---|---|---|---|---|
| CTGAN † | 7.57 (0/7) | 7.33 (0/12) | — | — | — | — | — |
| TVAE † | 7.50 (0/7) | 6.83 (0/12) | — | — | — | — | — |
| CTABGAN † | 7.14 (0/7) | — | — | — | — | — | — |
| CTABGAN+ † | 6.71 (0/7) | 6.58 (0/12) | — | — | — | — | — |
| SMOTE † | 4.29 (2/7) | 3.58 (3/12) | — | — | — | — | — |
| TabDDPM | 2.43 (3/7) | 2.62 (4/12) | — | — | 3.25 (3/12) | — | — |
| GNN-fixed | 2.43 (3/7) | 2.62 (3/12) | 2.46 (3/13) | 2.27 (4/13) | 2.25 (4/12) | -0.0003 | 0.791 / 0.848 |
| GNN-Jenks | 3.79 (0/7) | 3.21 (2/12) | 1.73 (4/13) | 2.00 (4/13) | 2.04 (3/12) | -0.0023 | 0.266 / 0.798 |
| GNN-entmax | 3.14 (2/7) | 3.21 (3/12) | 1.81 (6/13) | 1.73 (5/13) | 2.46 (4/12) | -0.0014 | 0.424 / 0.848 |

Each metric cell: average rank (1 = best) over the n datasets where every method ranked in that column has a value, and in brackets the number of those datasets on which the method ranks first (ties count for every tied method). MLE ranks are over all synthetic methods of Table 1 (DE excluded); Shape and Trend are available for our three variants only (ranks among three); DCR for TabDDPM and our three variants (ranks among four; DE and the datasets whose real train set the TabDDPM DCR subsamples excluded; TabDDPM's DCR is from one generated sample). These ranks are therefore not on the same scale as the MLE ranks. Shape/Trend: lower error is better; DCR: larger is ranked better. Last two columns: per-dataset difference in ML efficiency, GNN variant minus TabDDPM (authors' values), over the n = 12 datasets where both exist (DE excluded); two-sided Wilcoxon signed-rank test, p-values raw and Holm-corrected over the three variants. † reported values (TabDDPM paper, Table 5).
