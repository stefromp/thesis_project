# shellcheck shell=bash
#
# Best ablation configuration for: diabetes   (ABLATED dataset)
# Generated 2026-09-24 by hpc_transfer/best_manifest.py.
#
# Sourced by hpc_transfer/common.sh for ONE run:
#   hpc_transfer/jobs/diabetes_jenks.sh    topk_mode adaptive_jenks
#
# There is no fixed-mode job for this dataset: its fixed-top-k baseline already
# exists in tabddpm_ablation/diabetes_ablation.json. Keeping this configuration's
# own top_k means the output directory name matches that baseline exactly, so
# the adaptive/fixed pair lines up by name.
#
# PROVENANCE
#   source            : the dataset's OWN ablation results -- not a transfer.
#                       ablation/existing_results.load_best(), which ranks by
#                       metrics.score.mean (accuracy for classification,
#                       negative RMSE for regression, higher always better).
#   selected from     : 24 finished configuration(s) in tabddpm_ablation/
#   winner's score    : 0.792078 +/- 0.019545
#                       (std spans 5 generation x 10 classifier seeds of a
#                       SINGLE trained model -- ablation_runner trains once per
#                       cell, then re-samples, so training-seed variance is not
#                       in it)
#   grid snr          : 1.40  (no)
#   task              : binclass, 491 train rows, 8 nodes
#   synthetic rows    : 500
#
# NOTE: this is a different scope from hpc/, which replays EVERY recorded
# combination of 7 of these datasets (48 runs). Submit one study or the other,
# not both -- they write the same directories.
#
# !! THIS DATASET CANNOT DISCRIMINATE BETWEEN CONFIGURATIONS !!
#   snr = 1.40 (grid spread / median per-cell std on the primary metric).
#   Below about 2 the whole grid sits inside its own noise, so "best" here is
#   very close to an arbitrary pick among 24 equivalent configurations.
#   The adaptive-vs-fixed comparison for this dataset is still valid -- it is a
#   paired run of the SAME configuration -- but do not read the choice of
#   configuration as evidence of anything.

DATASET="diabetes"

GNN_TYPE="gin"
N_LAYERS=2
D_MODEL=32
TOP_K=5
N_HEADS=4
USE_ATTENTION=0    # mandatory 0: see hpc_transfer/common.sh

RUN_MODES="adaptive_jenks adaptive_entmax15"

PROVENANCE="own ablation best (load_best, score=0.792078); 24 configs; snr 1.40 (no)"
STRATEGY="own_ablation_best"
NEIGHBOURS=""
NEAREST_DISTANCE=""
CONFIDENCE="low"
