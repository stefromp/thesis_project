# shellcheck shell=bash
#
# Best ablation configuration for: churn2   (ABLATED dataset)
# Generated 2026-09-24 by hpc_transfer/best_manifest.py.
#
# Sourced by hpc_transfer/common.sh for ONE run:
#   hpc_transfer/jobs/churn2_jenks.sh    topk_mode adaptive_jenks
#
# There is no fixed-mode job for this dataset: its fixed-top-k baseline already
# exists in tabddpm_ablation/churn2_ablation.json. Keeping this configuration's
# own top_k means the output directory name matches that baseline exactly, so
# the adaptive/fixed pair lines up by name.
#
# PROVENANCE
#   source            : the dataset's OWN ablation results -- not a transfer.
#                       ablation/existing_results.load_best(), which ranks by
#                       metrics.score.mean (accuracy for classification,
#                       negative RMSE for regression, higher always better).
#   selected from     : 10 finished configuration(s) in tabddpm_ablation/
#   winner's score    : 0.858260 +/- 0.002784
#                       (std spans 5 generation x 10 classifier seeds of a
#                       SINGLE trained model -- ablation_runner trains once per
#                       cell, then re-samples, so training-seed variance is not
#                       in it)
#   grid snr          : 3.00  (weak)
#   task              : binclass, 6400 train rows, 11 nodes
#   synthetic rows    : 26000
#
# NOTE: this is a different scope from hpc/, which replays EVERY recorded
# combination of 7 of these datasets (48 runs). Submit one study or the other,
# not both -- they write the same directories.

DATASET="churn2"

GNN_TYPE="gatv2"
N_LAYERS=2
D_MODEL=128
TOP_K=5
N_HEADS=4
USE_ATTENTION=0    # mandatory 0: see hpc_transfer/common.sh

RUN_MODES="adaptive_jenks adaptive_entmax15"

PROVENANCE="own ablation best (load_best, score=0.858260); 10 configs; snr 3.00 (weak)"
STRATEGY="own_ablation_best"
NEIGHBOURS=""
NEAREST_DISTANCE=""
CONFIDENCE="medium"
