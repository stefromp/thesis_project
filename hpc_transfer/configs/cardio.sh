# shellcheck shell=bash
#
# Transferred configuration for: cardio
# Generated 2026-09-24 by hpc_transfer/phase3_transfer.py -- see predictions.md.
#
# Sourced by hpc_transfer/common.sh for BOTH runs of this dataset:
#   hpc_transfer/jobs/cardio_transfer.sh        topk_mode fixed
#   hpc_transfer/jobs/cardio_transfer_jenks.sh  topk_mode adaptive_jenks
# The architecture is identical across the two; only the top-k rule differs,
# which is the whole point of the comparison. The two land in sibling result
# trees (ablation/ and ablation_adaptive_jenks/) that ablation_runner derives
# from --topk_mode, so they can never be confused.
#
# PROVENANCE
#   strategy          : C (global best over the ablated datasets)
#                       Chosen by leave-one-out on the donor pool: strategy C
#                       reached mean utility regret 0.158 against 0.484 (B_k3)
#                       and 0.714 (A_k1). Neither nearest-neighbour transfer
#                       nor the per-axis vote beat the global best, so C is
#                       used for every target, as the brief requires.
#   donor pool        : abalone, california, churn2, insurance
#                       (the ablated datasets whose grid both discriminates
#                       between configs and is large enough for an argmax to
#                       mean something)
#   nearest donor     : churn2 (distance 2.207)
#   all distances     : churn2;california;insurance = 2.207;2.539;2.781
#   confidence        : medium
#   size vs donors    : 44,800 rows (3.4x the largest donor),
#                       11 nodes (1.0x the widest donor graph)
#   structural OOD    : n_cat;num_cat_ratio
#
# WHY d_model 128 AND NOT THE LOO ANSWER'S 64
#   In 8 of the 9 ablated datasets the best d_model is the largest value that
#   dataset happened to test; the one exception (wilt) is also the only dataset
#   with no discriminative signal at all. The axis was never saturated, so the
#   global best's d64 is a ceiling of the shared abalone/diabetes/insurance
#   sub-grid rather than a measured optimum -- and every target is larger than
#   every donor. d128/k5/h8 is a configuration that was actually measured: best
#   on churn2 and inside california's 1-sigma tie set, the two largest donors.
#   hpc_wide/ reached the same configuration independently, by a train-size
#   bracket argument, for five of these seven datasets.
#
# NOTE ON THE AXIS PARTNERS
#   k5 and h8 are d128's measured partners in the ablation. They are not mixed
#   with d64's k3/h4: no axis combination here is one that was never run.
#   In adaptive_jenks mode top_k is unused except during warm-up (default 0)
#   and in the directory name -- kept at 5 so both runs carry the same name.

DATASET="cardio"

GNN_TYPE="gatv2"
N_LAYERS=3
D_MODEL=128
TOP_K=5
N_HEADS=8
USE_ATTENTION=0    # mandatory 0: see hpc_transfer/common.sh

RUN_MODES="fixed adaptive_jenks adaptive_entmax15"

PROVENANCE="strategy C (global best); nearest donor churn2 d=2.207; confidence medium"
STRATEGY="C_global"
NEIGHBOURS="churn2;california;insurance"
NEAREST_DISTANCE="2.207"
CONFIDENCE="medium"
