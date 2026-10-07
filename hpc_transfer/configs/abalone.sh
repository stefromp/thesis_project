# shellcheck shell=bash
#
# Best ablation configuration for: abalone   (ABLATED dataset)
# Generated 2026-09-24 by hpc_transfer/best_manifest.py.
#
# Sourced by hpc_transfer/common.sh for ONE run:
#   hpc_transfer/jobs/abalone_jenks.sh    topk_mode adaptive_jenks
#
# There is no fixed-mode job for this dataset: its fixed-top-k baseline already
# exists in tabddpm_ablation/abalone_ablation.json. Keeping this configuration's
# own top_k means the output directory name matches that baseline exactly, so
# the adaptive/fixed pair lines up by name.
#
# PROVENANCE
#   source            : the dataset's OWN ablation results -- not a transfer.
#                       ablation/existing_results.load_best(), which ranks by
#                       metrics.score.mean (accuracy for classification,
#                       negative RMSE for regression, higher always better).
#   selected from     : 19 finished configuration(s) in tabddpm_ablation/
#   winner's score    : -2.009767 +/- 0.018535
#                       (std spans 5 generation x 10 classifier seeds of a
#                       SINGLE trained model -- ablation_runner trains once per
#                       cell, then re-samples, so training-seed variance is not
#                       in it)
#   grid snr          : 9.24  (yes)
#   task              : regression, 2672 train rows, 9 nodes
#   synthetic rows    : 20800
#
# NOTE: this is a different scope from hpc/, which replays EVERY recorded
# combination of 7 of these datasets (48 runs). Submit one study or the other,
# not both -- they write the same directories.

DATASET="abalone"

GNN_TYPE="gatv2"
N_LAYERS=3
D_MODEL=64
TOP_K=3
N_HEADS=4
USE_ATTENTION=0    # mandatory 0: see hpc_transfer/common.sh

RUN_MODES="adaptive_jenks adaptive_entmax15"

PROVENANCE="own ablation best (load_best, score=-2.009767); 19 configs; snr 9.24 (yes)"
STRATEGY="own_ablation_best"
NEIGHBOURS=""
NEAREST_DISTANCE=""
CONFIDENCE="high"
