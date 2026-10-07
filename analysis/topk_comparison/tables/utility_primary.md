| dataset | task | imbal. | metric | fixed | Jenks | entmax-1.5 | fixed grid summary (ref) |
|---|---|---|---|---|---|---|---|
| **Held-out evaluation (transferred): configuration chosen without seeing any of its results** | | | | | | | |
| cardio | binclass | 1.01 | Acc ↑ | 0.7370 ± 0.0003 | 0.7369 ± 0.0004 | **0.7375 ± 0.0004** | — |
| default | binclass | 3.52 | Acc ↑ | 0.8170 ± 0.0010 | **0.8175 ± 0.0007** | 0.8170 ± 0.0005 | — |
| fb-comments § | regression | — | RMSE ↓ | partial | partial | missing | — |
| house | regression | — | RMSE ↓ | 34105.7 ± 289.9 | 32564.3 ± 316.7 | **32268.9 ± 338.9** | — |
| king | regression | — | RMSE ↓ | 133707.4 ± 10516.2 | 136123.9 ± 5699.8 | **125077.8 ± 7475.4** | — |
| **Development set (ablated): configuration selected on the test split by fixed-mode score; in-sample and optimistic** | | | | | | | |
| abalone | regression | — | RMSE ↓ | 2.0231 ± 0.0130 | **2.0145 ± 0.0214** | 2.0266 ± 0.0284 | — |
| adult | binclass | 3.15 | Acc ↑ | **0.8598 ± 0.0005** | 0.8588 ± 0.0007 | 0.8584 ± 0.0002 | — |
| buddy | multiclass | 6.10 | Acc ↑ | **0.8964 ± 0.0021** | 0.8958 ± 0.0016 | 0.8962 ± 0.0012 | — |
| california | regression | — | RMSE ↓ | **0.4768 ± 0.0016** | 0.4813 ± 0.0019 | 0.4936 ± 0.0015 | 0.4776 ± 0.0017 |
| churn2 | binclass | 3.91 | Acc ↑ | **0.8604 ± 0.0012** | 0.8587 ± 0.0018 | 0.8601 ± 0.0012 | — |
| diabetes | binclass | 1.87 | Acc ↑ | 0.7881 ± 0.0176 | 0.7751 ± 0.0148 | **0.7909 ± 0.0150** | 0.7921 ± 0.0195 |
| gesture | multiclass | 2.96 | Acc ↑ | 0.6032 ± 0.0036 | **0.6076 ± 0.0016** | 0.6013 ± 0.0039 | 0.6019 ± 0.0062 |
| insurance | regression | — | RMSE ↓ | 4774.5 ± 20.2 | **4746.4 ± 7.5** | 4784.3 ± 20.7 | — |
| wilt | binclass | 17.54 | Acc ↑ | **0.9809 ± 0.0020** | 0.9792 ± 0.0014 | 0.9796 ± 0.0016 | 0.9809 ± 0.0019 |

Primary claims rest on the held-out (transferred) rows; development-set (ablated) rows are in-sample with respect to configuration selection, which was made on the test split by fixed-mode score, so they are optimistic, most of all for fixed. mean ± sd over generation seeds (n=5, each averaged over 10 CatBoost seeds). Bold = best mean in the row. † partial (<5 gen seeds). § trained with the categorical-input bug (cat_input=log; see caveats). ‡ fixed grid summary: sd spans all 5×10 gen×clf runs, not generation seeds. Imbal. = majority/minority class ratio (— for regression).
