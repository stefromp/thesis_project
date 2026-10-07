#!/usr/bin/env bash
#
# Submission loop for Route A (AttnGNN d_model 256 inside TabDiff). NOTHING HERE IS EXECUTED:
# every sbatch line is commented out on purpose; uncomment one block at a time.
# No TabDiff (transformer) runs: TabDiff's numbers are the reported ones. No DCR.
# '###' = OVER_CAP (training estimate > 48 h, TabDiff's Trainer cannot resume): blocked
# until the timing probe (README) shows it fits. Regenerate: python3 tabdiff_gnn/make_jobs.py.

cd "$(dirname "$0")/jobs_tabdiff_gnn" || exit 1

# ==== block 1: AttnGNN d256 seed 0: magic, shoppers, adult, beijing
# gnn magic: 5 h walltime, ~3.2 GPU-h
# sbatch gnn_magic_s0.sh
# gnn shoppers: 4 h walltime, ~2.6 GPU-h
# sbatch gnn_shoppers_s0.sh
# gnn adult: 6 h walltime, ~4.2 GPU-h
# sbatch gnn_adult_s0.sh
# gnn beijing: 10 h walltime, ~6.9 GPU-h
# sbatch gnn_beijing_s0.sh
# block 1: ~16.9 GPU-h

# ==== block 2: AttnGNN d256 seed 0: default, news (resumable 3-job chain), diabetes
# gnn default: 17 h walltime, ~12.5 GPU-h
# sbatch gnn_default_s0.sh
# gnn news: CHAIN of 3 (48 h + 48 h + 5 h), ~55.9 GPU-h
# j_gnn_news_s0_1=$(sbatch --parsable gnn_news_s0_p1.sh)
# j_gnn_news_s0_2=$(sbatch --parsable --dependency=afterok:${j_gnn_news_s0_1} gnn_news_s0_p2.sh)
# j_gnn_news_s0_3=$(sbatch --parsable --dependency=afterok:${j_gnn_news_s0_2} gnn_news_s0_p3.sh)
# gnn diabetes: 20 h walltime, ~15.3 GPU-h
# sbatch gnn_diabetes_s0.sh
# block 2: ~83.7 GPU-h

# ==== training seed 1 (ONLY AFTER CONFIRMATION)
# gnn magic: 5 h walltime, ~3.2 GPU-h
## sbatch gnn_magic_s1.sh
# gnn shoppers: 4 h walltime, ~2.6 GPU-h
## sbatch gnn_shoppers_s1.sh
# gnn adult: 6 h walltime, ~4.2 GPU-h
## sbatch gnn_adult_s1.sh
# gnn beijing: 10 h walltime, ~6.9 GPU-h
## sbatch gnn_beijing_s1.sh
# gnn default: 17 h walltime, ~12.5 GPU-h
## sbatch gnn_default_s1.sh
# gnn news: CHAIN of 3 (48 h + 48 h + 5 h), ~55.9 GPU-h
## j_gnn_news_s1_1=$(sbatch --parsable gnn_news_s1_p1.sh)
## j_gnn_news_s1_2=$(sbatch --parsable --dependency=afterok:${j_gnn_news_s1_1} gnn_news_s1_p2.sh)
## j_gnn_news_s1_3=$(sbatch --parsable --dependency=afterok:${j_gnn_news_s1_2} gnn_news_s1_p3.sh)
# gnn diabetes: 20 h walltime, ~15.3 GPU-h
## sbatch gnn_diabetes_s1.sh
# seed 1: ~100.6 GPU-h

# ==== training seed 2 (ONLY AFTER CONFIRMATION)
# gnn magic: 5 h walltime, ~3.2 GPU-h
## sbatch gnn_magic_s2.sh
# gnn shoppers: 4 h walltime, ~2.6 GPU-h
## sbatch gnn_shoppers_s2.sh
# gnn adult: 6 h walltime, ~4.2 GPU-h
## sbatch gnn_adult_s2.sh
# gnn beijing: 10 h walltime, ~6.9 GPU-h
## sbatch gnn_beijing_s2.sh
# gnn default: 17 h walltime, ~12.5 GPU-h
## sbatch gnn_default_s2.sh
# gnn news: CHAIN of 3 (48 h + 48 h + 5 h), ~55.9 GPU-h
## j_gnn_news_s2_1=$(sbatch --parsable gnn_news_s2_p1.sh)
## j_gnn_news_s2_2=$(sbatch --parsable --dependency=afterok:${j_gnn_news_s2_1} gnn_news_s2_p2.sh)
## j_gnn_news_s2_3=$(sbatch --parsable --dependency=afterok:${j_gnn_news_s2_2} gnn_news_s2_p3.sh)
# gnn diabetes: 20 h walltime, ~15.3 GPU-h
## sbatch gnn_diabetes_s2.sh
# seed 2: ~100.6 GPU-h

# GPU-h (estimate, before the x1.3 margin): seed 0 100.6; seeds 1-2 201.2 more

# Analysis afterwards: python3 analysis/tabdiff/compare.py --route_a
