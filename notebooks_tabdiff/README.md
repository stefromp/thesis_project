# NOT TabDiff's splits or protocol

The `*-kaggle-tabdiff.ipynb` notebooks here run `ablation/ablation_runner.py`
on `data/<ds>_tabdiff_split/` (a size-matched re-split, **not** TabDiff's own
split) and score with `ablation/eval_xgboost.py` / `ablation/eval_tabdiff_fidelity.py`
(re-implementations of TabDiff's evaluators, **not** TabDiff's code; e.g. their
MLE selects hyper-parameters on the real validation split, while TabDiff's
`eval/mle/mle.py` carves the validation set out of the synthetic table when a
dataset has no val.csv). Numbers from these notebooks are not directly
comparable with the TabDiff paper's tables.

For that comparison use the TabDiff-transfer study: TabDiff's own splits
(`data/tabdiff_<ds>/`) and TabDiff's own evaluation code (`third_party/TabDiff`
@ 5ecdb33, called unchanged by `hpc_transfer/tabdiff_study/tabdiff_eval.py`).
See `hpc_transfer/tabdiff_study/README.md`.
