| dataset | task | imbal. | metric | fixed | Jenks | entmax-1.5 | fixed grid summary (ref) |
|---|---|---|---|---|---|---|---|
| **Held-out evaluation (transferred): configuration chosen without seeing any of its results** | | | | | | | |
| cardio § | binclass | 1.01 | macro-F1 ↑ | **0.7355 ± 0.0008** | 0.7354 ± 0.0005 | 0.7340 ± 0.0002 | — |
| default § | binclass | 3.52 | macro-F1 ↑ | 0.6811 ± 0.0011 | 0.6837 ± 0.0005 | **0.6937 ± 0.0013** | — |
| fb-comments § | regression | — | R² ↑ | partial | partial | missing | — |
| house | regression | — | R² ↑ | 0.5971 ± 0.0068 | 0.6326 ± 0.0071 | **0.6393 ± 0.0076** | — |
| king § | regression | — | R² ↑ | 0.8296 ± 0.0327 | **0.8607 ± 0.0185** | 0.7969 ± 0.0313 | — |
| **Development set (ablated): configuration selected on the test split by fixed-mode score; in-sample and optimistic** | | | | | | | |
| abalone § | regression | — | R² ↑ | 0.5560 ± 0.0082 ‡ | 0.3234 ± 0.0064 | **0.5627 ± 0.0078** | — |
| adult § | binclass | 3.15 | macro-F1 ↑ | **0.7882 ± 0.0015** ‡ | 0.7873 ± 0.0006 | 0.7844 ± 0.0004 | — |
| buddy § | multiclass | 6.10 | macro-F1 ↑ | **0.8914 ± 0.0028** ‡ | 0.8890 ± 0.0042 | 0.8631 ± 0.0040 | — |
| california | regression | — | R² ↑ | **0.8256 ± 0.0012** | 0.8223 ± 0.0014 | 0.8131 ± 0.0011 | 0.8251 ± 0.0013 |
| churn2 § | binclass | 3.91 | macro-F1 ↑ | **0.7426 ± 0.0050** ‡ | 0.7426 ± 0.0042 | 0.7197 ± 0.0062 | — |
| diabetes | binclass | 1.87 | macro-F1 ↑ | **0.7585 ± 0.0208** | 0.7374 ± 0.0222 | 0.7561 ± 0.0197 | 0.7625 ± 0.0235 |
| gesture | multiclass | 2.96 | macro-F1 ↑ | 0.5489 ± 0.0044 | **0.5525 ± 0.0012** | 0.5494 ± 0.0052 | 0.5495 ± 0.0084 |
| insurance § | regression | — | R² ↑ | **0.8165 ± 0.0013** ‡ | 0.8101 ± 0.0029 | 0.5950 ± 0.0235 | — |
| wilt | binclass | 17.54 | macro-F1 ↑ | **0.9064 ± 0.0116** | 0.8966 ± 0.0080 | 0.8994 ± 0.0080 | 0.9069 ± 0.0094 |

Primary claims rest on the held-out (transferred) rows; development-set (ablated) rows are in-sample with respect to configuration selection, which was made on the test split by fixed-mode score, so they are optimistic, most of all for fixed. mean ± sd over generation seeds (n=5, each averaged over 10 CatBoost seeds). Bold = best mean in the row. † partial (<5 gen seeds). § trained with the categorical-input bug (cat_input=log; see caveats). ‡ fixed grid summary: sd spans all 5×10 gen×clf runs, not generation seeds. Imbal. = majority/minority class ratio (— for regression).
