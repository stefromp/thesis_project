# shellcheck shell=bash
#
# Best ablation configuration for: insurance   (ABLATED dataset)
# Generated 2026-09-24 by hpc_transfer/best_manifest.py.
#
# Sourced by hpc_transfer/common.sh for ONE run:
#   hpc_transfer/jobs/insurance_jenks.sh    topk_mode adaptive_jenks
#
# There is no fixed-mode job for this dataset: its fixed-top-k baseline already
# exists in tabddpm_ablation/insurance_ablation.json. Keeping this configuration's
# own top_k means the output directory name matches that baseline exactly, so
# the adaptive/fixed pair lines up by name.
#
# PROVENANCE
#   source            : the dataset's OWN ablation results -- not a transfer.
#                       ablation/existing_results.load_best(), which ranks by
#                       metrics.score.mean (accuracy for classification,
#                       negative RMSE for regression, higher always better).
#   selected from     : 16 finished configuration(s) in tabddpm_ablation/
#   winner's score    : -4718.236328 +/- 17.043612
#                       (std spans 5 generation x 10 classifier seeds of a
#                       SINGLE trained model -- ablation_runner trains once per
#                       cell, then re-samples, so training-seed variance is not
#                       in it)
#   grid snr          : 52.50  (yes)
#   task              : regression, 856 train rows, 7 nodes
#   synthetic rows    : 7200
#
# NOTE: this is a different scope from hpc/, which replays EVERY recorded
# combination of 7 of these datasets (48 runs). Submit one study or the other,
# not both -- they write the same directories.

DATASET="insurance"

GNN_TYPE="gatv2"
N_LAYERS=2
D_MODEL=64
TOP_K=5
N_HEADS=4
USE_ATTENTION=0    # mandatory 0: see hpc_transfer/common.sh

RUN_MODES="adaptive_jenks adaptive_entmax15"

PROVENANCE="own ablation best (load_best, score=-4718.236328); 16 configs; snr 52.50 (yes)"
STRATEGY="own_ablation_best"
NEIGHBOURS=""
NEAREST_DISTANCE=""
CONFIDENCE="high"
