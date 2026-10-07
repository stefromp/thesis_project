"""
TabDiff numbers REPORTED by GATD -- "Geometry-Aware Tabular Diffusion"
(D. T. Zagardo, ICML 2026, arXiv:2606.02607v1), transcribed by hand from the
PDF (pdftotext -layout of https://arxiv.org/pdf/2606.02607v1).

GATD re-ran TabDiff "from the official TabDiff codebase under their published
8,000-epoch protocol with three training seeds and twenty generation seeds"
(Sec. 4.2), so these std include retraining variance -- unlike the TabDiff
paper's, which span the 20 tables of one trained model.

Table 9 (A.5) "MLP+Geom vs. TabDiff with mean ± std per dataset", TabDiff rows:
  shape, trend   error rate, %                     (lower is better)
  mle1           AUROC (classification) / R2 (beijing, news)  (higher)
  mle2           F1 (classification) / RMSE (beijing, news)
Table 16 (A.11): TabDiff parameters, training hours and seconds per table on
one NVIDIA T4 (used by tabdiff_gnn/make_jobs.py as walltime upper bounds).
Only the TabDiff benchmark datasets are kept (GATD's Bikeshare, California,
Power are not in this study).
"""

SOURCE = "GATD (Zagardo, ICML 2026, arXiv:2606.02607v1)"
TABLE_OF = {"shape": "Table 9", "trend": "Table 9", "mle1": "Table 9", "mle2": "Table 9",
            "params": "Table 16"}
PROTOCOL = "3 training seeds x 20 generation seeds, 8000 epochs (TabDiff's protocol)"

# dataset -> (mean, std)
TABDIFF = {
    "shape": {"adult": (0.665, 0.0712), "default": (1.29, 0.0777), "diabetes": (1.42, 0.194),
              "magic": (0.811, 0.108), "shoppers": (1.34, 0.102), "beijing": (1.00, 0.0572),
              "news": (2.41, 0.0448)},
    "trend": {"adult": (1.53, 0.208), "default": (4.71, 1.58), "diabetes": (3.48, 0.516),
              "magic": (0.876, 0.228), "shoppers": (1.87, 0.127), "beijing": (2.60, 0.0933),
              "news": (1.74, 0.0697)},
    # AUROC for classification, R2 for regression
    "mle1": {"adult": (0.912, 0.00211), "default": (0.758, 0.00701), "diabetes": (0.663, 0.0401),
             "magic": (0.936, 0.00322), "shoppers": (0.921, 0.00391), "beijing": (0.698, 0.0141),
             "news": (0.0151, 0.0537)},
    # F1 for classification, RMSE for regression
    "mle2": {"adult": (0.667, 0.00854), "default": (0.439, 0.0148), "diabetes": (0.525, 0.0527),
             "magic": (0.818, 0.00784), "shoppers": (0.617, 0.0171), "beijing": (0.564, 0.0131),
             "news": (0.870, 0.0237)},
}
# Table 16: TabDiff params, train hours (T4), sample seconds per table (T4)
TABDIFF_COST = {"adult": ("10.6M", 3.62, 34.6), "beijing": ("10.6M", 4.21, 33.0),
                "default": ("10.7M", 3.74, 29.5), "diabetes": ("10.8M", 13.51, 299.5),
                "magic": ("10.6M", 2.24, 14.6), "news": ("10.9M", 5.34, 51.4),
                "shoppers": ("10.6M", 2.19, 19.2)}


def cell(metric: str, dataset: str):
    return TABDIFF.get(metric, {}).get(dataset)
