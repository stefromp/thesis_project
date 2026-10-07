| dataset | task | imbal. | metric | fixed | Jenks | entmax-1.5 | fixed grid summary (ref) |
|---|---|---|---|---|---|---|---|
| **Held-out evaluation (transferred): configuration chosen without seeing any of its results** | | | | | | | |
| cardio § | binclass | 1.01 | Acc ↑ | 0.7361 ± 0.0008 | **0.7361 ± 0.0005** | 0.7344 ± 0.0002 | — |
| default § | binclass | 3.52 | Acc ↑ | **0.8173 ± 0.0005** | 0.8162 ± 0.0014 | 0.8119 ± 0.0006 | — |
| fb-comments § | regression | — | RMSE ↓ | partial | partial | missing | — |
| house | regression | — | RMSE ↓ | 34105.7 ± 289.9 | 32564.3 ± 316.7 | **32268.9 ± 338.9** | — |
| king § | regression | — | RMSE ↓ | 144539.0 ± 13357.0 | **130930.5 ± 8899.7** | 157941.8 ± 12420.2 | — |
| **Development set (ablated): configuration selected on the test split by fixed-mode score; in-sample and optimistic** | | | | | | | |
| abalone § | regression | — | RMSE ↓ | 2.0098 ± 0.0185 ‡ | 2.4811 ± 0.0116 | **1.9946 ± 0.0180** | — |
| adult § | binclass | 3.15 | Acc ↑ | **0.8580 ± 0.0009** ‡ | 0.8555 ± 0.0005 | 0.8483 ± 0.0005 | — |
| buddy § | multiclass | 6.10 | Acc ↑ | 0.8820 ± 0.0022 ‡ | **0.8864 ± 0.0019** | 0.8774 ± 0.0015 | — |
| california | regression | — | RMSE ↓ | **0.4768 ± 0.0016** | 0.4813 ± 0.0019 | 0.4936 ± 0.0015 | 0.4776 ± 0.0017 |
| churn2 § | binclass | 3.91 | Acc ↑ | 0.8583 ± 0.0028 ‡ | **0.8588 ± 0.0027** | 0.8476 ± 0.0018 | — |
| diabetes | binclass | 1.87 | Acc ↑ | 0.7881 ± 0.0176 | 0.7751 ± 0.0148 | **0.7909 ± 0.0150** | 0.7921 ± 0.0195 |
| gesture | multiclass | 2.96 | Acc ↑ | 0.6032 ± 0.0036 | **0.6076 ± 0.0016** | 0.6013 ± 0.0039 | 0.6019 ± 0.0062 |
| insurance § | regression | — | RMSE ↓ | **4718.2 ± 17.0** ‡ | 4800.2 ± 36.0 | 7008.1 ± 205.8 | — |
| wilt | binclass | 17.54 | Acc ↑ | **0.9809 ± 0.0020** | 0.9792 ± 0.0014 | 0.9796 ± 0.0016 | 0.9809 ± 0.0019 |

Primary claims rest on the held-out (transferred) rows; development-set (ablated) rows are in-sample with respect to configuration selection, which was made on the test split by fixed-mode score, so they are optimistic, most of all for fixed. mean ± sd over generation seeds (n=5, each averaged over 10 CatBoost seeds). Bold = best mean in the row. † partial (<5 gen seeds). § trained with the categorical-input bug (cat_input=log; see caveats). ‡ fixed grid summary: sd spans all 5×10 gen×clf runs, not generation seeds. Imbal. = majority/minority class ratio (— for regression).
