# TabDiff-transfer study

Our GNN denoiser on the TabDiff benchmark (adult, default, shoppers, magic,
beijing, news; diabetes dropped), on **TabDiff's own splits**, scored with
**TabDiff's own evaluation code** ([MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff)
@ `5ecdb3356261aea72716cc9a779f31d7ad083bf4`, submodule `third_party/TabDiff`), so the
numbers sit directly next to the TabDiff paper's tables.

Each dataset borrows from its nearest same-task TabDDPM dataset (the "source"):

| TabDiff dataset | source | lr | steps | batch | T | GNN (fixed top-k, cat_input=prob, dynamic, no attention) |
|---|---|---|---|---|---|---|
| magic    | wilt    | 1.071e-4 | 30000 | 256  | 100  | GATv2 L2 d64 k3 h8 |
| shoppers | adult   | 2.010e-3 | 30000 | 4096 | 100  | GCN L2 d128 k5 |
| adult    | adult   | 2.010e-3 | 30000 | 4096 | 100  | GCN L2 d128 k5 |
| default  | default | 4.682e-4 | 30000 | 4096 | 1000 | GATv2 L3 d128 k5 h8 |
| beijing  | king    | 7.444e-4 | 30000 | 4096 | 100  | GATv2 L3 d128 k5 h8 |
| news     | king    | 7.444e-4 | 30000 | 4096 | 100  | GATv2 L3 d128 k5 h8 |

Training setup = the TabDDPM authors' `exp/<src>/ddpm_cb_best/config.toml` (lr, steps,
batch, T, weight decay, scheduler, sampling batch via `ablation_runner.training_setup`,
plus `[train.T] normalization`, `[model_params] is_y_cond`); GNN = `hpc_transfer/configs/<src>.sh`
(= `analysis/paper_tables` GNN configs). Every value and its file:line is printed by
`runner.py --dry_run`, written into each job script header and into `run_meta.json`.
The GNN is the only model of the study (no MLP arm).

## One job per dataset

`hpc_transfer/jobs_tabdiff/<ds>.sh` (gpu partition, `--gres=gpu:a100:1`, `--mem=24G`, cd to
repo root) runs `runner.py --dataset <ds>`, stage by stage:

```
tabdiff_<ds>      train -> sample (20 tables) -> eval_main (Shape/Trend, MLE gpu_hist, C2ST) -> eval_quality (alpha/beta)
tabdiff_<ds>_dcr  train -> sample (20 tables) -> eval_dcr                       (not for magic)
```

Each stage writes its outputs, then `<run>/stages/<stage>.done`. A resubmitted job skips
every stage whose marker is present and whose outputs still check out (model record,
20 tables, evaluation files), so a crash or timeout never repeats finished work; a stage
that does run invalidates the later markers of its target. A model trained differently
(other cat_input / setup / source) is refused, never reused. Walltime = sum of the stage
estimates × 1.3 (rounded up, cap 48 h); a dataset that would exceed 48 h is split into
`<ds>_main.sh` + `<ds>_dcr.sh` chained with `--dependency=afterok` -- currently none is
(news, the largest, is 31 h). `tabdiff_eval.py` remains usable on its own to re-run an
evaluation on existing tables.

## Protocol (TabDiff's)

* Train once per target on TabDiff's train split; sample **20 tables of n_train rows**
  (seeds 0..19); mean ± std (ddof=1) over the 20.
* Metrics = TabDiff's `tabdiff/metrics.py` (`TabMetrics`) through their
  `Trainer.report_test`: Shape / Trend (sdmetrics QualityReport, reported as error %),
  MLE (XGBoost 36-point grid, val = 1/9 of the synthetic table, AUC or RMSE on their
  test split), C2ST (sdmetrics LogisticDetection); α-precision / β-recall by their
  `eval/eval_quality.py` (synthcity, naive variants); DCR by `evaluate_dcr` /
  `report_test_dcr` on a separate model trained on the 50/50 `<ds>_dcr` split
  (all but magic, as in their Table 10). All of it inside the dataset's GPU job.
* Deviations (also in run_meta.json and the analysis): final EMA weights instead of
  "best EMA loss after epoch 4000"; the source's TabDDPM training setup instead of
  TabDiff's config; the MLE's global numpy RNG is seeded per table; our loader's
  val file is a copy of test (unused by training).

## Files

| path | role |
|---|---|
| `study.py` | targets, sources, names (`tabdiff_` prefix enforced), result tree `<exp_root>/exp_tabdiff/`, transferred values + file:line, TabDiff file digests, package pins |
| `prepare_data.py` | runs TabDiff's `download_dataset.py` (adult, default) / local raw files + their `process_dataset.py` unchanged in `data/tabdiff_raw/`; writes `data/tabdiff_<ds>[_dcr]/` + `data/tabdiff_manifest.json` |
| `export.py` | sampler output → CSV in exactly TabDiff's raw format (columns, order, labels, dtypes) |
| `runner.py` | the dataset job: stages with done-markers (train, sample+export, TabDiff evaluation; then the same for the _dcr model) |
| `tabdiff_eval.py` | thin wrapper calling TabDiff's code (main / dcr / quality); lists every adaptation; also standalone for re-evaluation |
| `selfcheck.py` | real-vs-real through the export path + TabDiff's evaluator |
| `make_jobs.py` | writes `../jobs_tabdiff/*.sh`, `walltime_estimates.csv`, `../submit_tabdiff.sh` |
| `mock/` | local mock dry run of a real dataset job (`dryrun_mock.sh`) |
| `tests/test_tabdiff_transfer.py` | tests |
| `analysis/tabdiff/compare.py` | ours vs the paper (reported numbers in `analysis/tabdiff/reported.py`) |

Results: `$TRANSFER_ROOT/exp_tabdiff/tabdiff_<ds>[_dcr]/gnn/<combo>/`
(`run_meta.json`, `train_setup.json`, `denoiser_config.json`, `model_ema.pt`, `stages/*.done`,
`samples/samples_<k>.csv`, `tabdiff_eval/{all_results,avg_std,quality,dcr}.csv`, `eval_meta.json`).

## Install (cluster, `module load python/3.10`)

Two extra venvs; the training venv (`venvs/tabddpm310`) is unchanged.

```bash
module purge; module load python/3.10 cuda/11.8
HPC_ROOT=/work/pr019/atvoulo/Rompos

# 1. eval venv: density / MLE / C2ST, and DCR on the GPU job's card (CUDA torch)
python -m venv $HPC_ROOT/venvs/tabdiff_eval310
E=$HPC_ROOT/venvs/tabdiff_eval310/bin/pip
$E install --upgrade pip
$E install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
$E install numpy==1.26.4 pandas==2.2.3 "scipy<1.16" scikit-learn==1.5.2 \
           sdmetrics==0.18.0 xgboost==2.1.4 category_encoders==2.6.4 statsmodels==0.14.4 \
           icecream prdc plotly pillow tqdm tomli

# 2. synthcity venv: alpha-precision / beta-recall (TabDiff keeps it separate too)
python -m venv $HPC_ROOT/venvs/tabdiff_synthcity310
P=$HPC_ROOT/venvs/tabdiff_synthcity310/bin/pip
$P install --upgrade pip
$P install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
$P install synthcity==0.2.12 opacus==1.5.2 numpy==1.26.4 xgboost==2.1.4 category_encoders tomli
```

Why these pins (TabDiff's `tabdiff.yaml` / `synthcity.yaml` pin none of them):
sdmetrics 0.18.0 and xgboost 2.1.4 were current when TabDiff's eval code was committed
(2025-02-16/19); sdmetrics ≥0.28 crashes `LogisticDetection` on TabDiff's integer column
labels and changes Trend; xgboost 3 removed `tree_method='gpu_hist'`; scikit-learn ≥1.6
removed `mean_squared_error(squared=False)` (their regression MLE); `category_encoders`
2.6.4 + `icecream` are imported by their `utils_train.py → src/`; `opacus` ≥1.5.3 needs
torch ≥2.4 but synthcity pins torch <2.3. Verified locally (Python 3.11, macOS) with the
same pins; on the cluster the first test below verifies them under 3.10.

## Upload list (the cluster has no git history)

Commit first, then record the commit for `run_meta.json` (the runner reads this file when
git is unavailable):

```bash
echo "$(git rev-parse HEAD)$(git status --porcelain --untracked-files=no | grep -q . && echo -dirty)" \
    > hpc_transfer/tabdiff_study/UPLOAD_COMMIT
```

```
third_party/TabDiff/                     # whole submodule working tree (no .git needed)
hpc_transfer/tabdiff_study/              # all files, incl. UPLOAD_COMMIT
hpc_transfer/jobs_tabdiff/               # <ds>.sh x6 + common_tabdiff.sh + walltime_estimates.csv
hpc_transfer/submit_tabdiff.sh
hpc_transfer/common.sh                   # cat_input_preflight (unchanged)
hpc_transfer/configs/{adult,default,king,wilt}.sh
exp/{adult,default,king,wilt}/ddpm_cb_best/config.toml
ablation/ablation_runner.py  ablation/{adult,default,king,wilt}_ablation.py
scripts/train.py  scripts/sample.py  scripts/utils_train.py
lib/  tab_ddpm/                          # current working-tree versions (cat_input, divergence check)
data/tabdiff_{magic,shoppers,shoppers_dcr,adult,adult_dcr,default,default_dcr,beijing,beijing_dcr,news,news_dcr}/
data/tabdiff_manifest.json
tests/test_tabdiff_transfer.py           # optional
```
Not needed on the cluster: `data/tabdiff_raw/` (downloads + TabDiff's work dir),
`analysis/`, `hpc_transfer/tabdiff_study/mock/`.

## Cluster tests (before any submission)

All on the gpu partition (the only usable one). If `--gres=gpu:a100:1` is rejected
("invalid GRES format", see the note in `hpc_transfer/jobs/*_jenks.sh`), find the accepted
string with `sinfo -p gpu -o "%N %G"` and pass it as `sbatch --gres=<string> <ds>.sh`
(a command-line --gres overrides the directive) -- and use the same string below.

```bash
cd $HPC_ROOT/thesis_project
PY=$HPC_ROOT/venvs/tabddpm310/bin/python
EV=$HPC_ROOT/venvs/tabdiff_eval310/bin/python
SY=$HPC_ROOT/venvs/tabdiff_synthcity310/bin/python
GPU="--partition=gpu --gres=gpu:a100:1 --cpus-per-task=26 --mem=24G"

# 1. code + data + transferred values + stage state, no GPU (login node)
for d in magic shoppers adult default beijing news; do
  $PY hpc_transfer/tabdiff_study/runner.py --dataset $d --exp_root $HPC_ROOT/transfer_runs --dry_run | grep DRY
done
$PY -m pytest -q tests/test_tabdiff_transfer.py      # if pytest is in the venv

# 2. TabDiff's evaluator on the A100, real-vs-real through our export path, no patch
#    (gpu_hist as TabDiff); expect shape/trend < 3, mle ~ .91-.92, c2st ~ 100, alpha ~ 99
srun $GPU --time=00:30:00 $EV hpc_transfer/tabdiff_study/selfcheck.py --target tabdiff_shoppers \
     --work $HPC_ROOT/tabdiff_selfcheck --n_runs 2 --quality_python $SY --device cuda:0
srun $GPU --time=00:20:00 $EV hpc_transfer/tabdiff_study/selfcheck.py --target tabdiff_shoppers_dcr \
     --work $HPC_ROOT/tabdiff_selfcheck --n_runs 2 --device cuda:0

# 3. timing + full stage chain for the least certain estimate (news: 48 graph nodes vs
#    king's 21): 500 steps, 1 table per model, through the MOCK shim, throw-away tree
srun $GPU --time=02:00:00 env MOCK_STEPS=500 MOCK_RUNS=1 $PY hpc_transfer/tabdiff_study/mock/mock_python.py \
     hpc_transfer/tabdiff_study/runner.py --dataset news --exp_root $HPC_ROOT/tabdiff_timing \
     --device cuda:0 --eval_python $EV --synthcity_python $SY
#    s/step = (train_finished_utc - train_started_utc) / 500 in each run_meta.json;
#    x 30000 x 2 models + 40 tables must stay well inside news.sh's 31 h.
```

## Submission

`hpc_transfer/submit_tabdiff.sh` -- every `sbatch` is commented out; one job per dataset,
cheapest first (magic, shoppers, adult, default, beijing, news). About 28 GPU-h (linear
node scaling) to 41 GPU-h (quadratic) in total, see `../jobs_tabdiff/walltime_estimates.csv`.
A job that dies is resubmitted unchanged and continues at the first unfinished stage.
Then `python3 analysis/tabdiff/compare.py`.
