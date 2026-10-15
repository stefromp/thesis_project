# Route A — AttnGNN as the denoiser inside TabDiff

TabDiff ([MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff) @ `5ecdb33`, `third_party/TabDiff`,
**never modified**, digests checked) runs through its own classes:

- `UnifiedCtimeDiffusion`: EDM for the numerical columns, masked diffusion for the categorical ones, learnable per-column schedules.
- `Model`/`Precond`.
- `Trainer`: training loop, EMA, best-EMA checkpoint after epoch 4000, 50-step sampler.
- `TabDiffDataset` and `TabMetrics`.

One thing is substituted: the denoiser. **One configuration is run: AttnGNN V2 with `d_model` 256 on all 7 datasets** (magic, shoppers, adult, default, beijing, news, diabetes). Its neighbour-selection rule comes in three arms: the main **fixed** top-k arm, plus **adaptive_jenks** and **adaptive_entmax15** (see "Adaptive neighbour-selection arms" below). Everything except the rule is identical across the arms.

**TabDiff itself is not re-run.** We compare against the reported TabDiff numbers:
- primary: GATD's re-run of TabDiff, 3 training seeds × 20 tables, whose std includes retraining variance;
- secondary: the TabDiff paper;
- wNMIS and C2ST (XGB): tabpc's TabDiff.

The `denoiser=transformer` code path stays, with its tests, as evidence that the framework is TabDiff's: the wrapper is bit-identical to TabDiff's `UniModMLP`, and the mirrored entry point builds the same model as `tabdiff/main.py`. No jobs are generated for it.

The V2 denoiser chain is:

1. TabDiff `Tokenizer` at `d_model` (its [CLS] token is dropped, as in `UniModMLP`).
2. One **shared** noise embedding added to every node: TabDiff's `PositionalEmbedding(dim_t=128)` with the sin/cos swap, then Linear-SiLU-Linear.
3. `tab_ddpm.graph_denoiser.AttnGNNBlock` × n_layers, **imported unchanged**: dynamic `DynamicAdjacency`, fixed top-k, `use_attention=False`.
4. TabDiff `Reconstructor` at `d_model`.

All cross-column mixing goes through the learned sparse graph. TabDiff's two transformers and its flattened MLP are replaced.

The GNN hyperparameters per dataset come from the TabDiff study's source mapping (`study.SOURCES` → `hpc_transfer/configs/<src>.sh`): gnn type, layers, top_k and heads. The fixed top-k rule, dynamic graph, no global attention, `dim_t` 128 and the shared noise embedding are common to all. **`d_model` is 256 everywhere** (the source runs used 64/128), and the heads divide 256 on every dataset:

- magic ← wilt: GATv2 L2 k3 h8
- shoppers, adult ← adult: GCN L2 k5 (heads 4, unused by GCN)
- default ← default: GATv2 L3 k5 h8
- beijing, news ← king: GATv2 L3 k5 h8
- diabetes ← adult (GCN L2 k5). **This is our choice: diabetes has no TabDDPM source**, so it is not in the study's mapping. It is recorded in every diabetes `run_meta.json` (`gnn_source_note`, `gnn_source_is_our_choice`), the job header and `ROUTE_A.md`.

Flags:

- `--node_noise per_column` (off): each node embeds its own `log σ_j(t)/4`, read from the live learnable schedules through a weakref.
- `--geo_loss`, `--geo_edge_bias`: reserved hooks, default off, not implemented (enabling them raises an error).

Parameter counts of the denoiser come from `params.py`, on the real shapes. Both models add d_num + n_cat schedule parameters on top.

| dataset | N | max card. | AttnGNN d256 | TabDiff | ratio |
|---|---|---|---|---|---|
| magic | 11 | 2 | 3,478,403 | 10,589,231 | 0.33 |
| shoppers | 18 | 20 | 2,729,675 | 10,647,279 | 0.26 |
| adult | 15 | 42 | 2,746,353 | 10,622,989 | 0.26 |
| default | 24 | 11 | 5,237,465 | 10,696,637 | 0.49 |
| beijing | 12 | 31 | 5,226,705 | 10,598,109 | 0.49 |
| news | 48 | 7 | 5,222,031 | 10,893,027 | 0.48 |
| diabetes | 37 | 716 | 3,843,133 | 10,822,529 | 0.36 |

## Protocol

TabDiff's `tabdiff_configs.toml` is used unchanged:

- 8000 epochs at batch 4096.
- AdamW, lr 1e-3, ReduceLROnPlateau (factor 0.9, patience 50).
- Continuous-loss weight annealed from 1 to 0.
- EMA 0.997, updated once per epoch.
- Best EMA training loss after epoch 4000.
- 50 sampling steps with the stochastic sampler and second-order correction.

The run protocol follows GATD:

- **3 training seeds × 20 tables** of n_train rows each, with generation seeds 0..19. Seed 0 runs first; seeds 1-2 only after confirmation. The capacity arm runs seed 0 only.
- **No DCR in this study.** GATD's protocol does not use it, so no `_dcr` model is trained, sampled or evaluated. The TabDiff-study pipeline's DCR is untouched.

Evaluation of every table:

- TabDiff's evaluator, through `hpc_transfer/tabdiff_study/tabdiff_eval.py` (`report_tables` / `quality_tables`, digests and pins checked): Shape, Trend, MLE, C2ST, α and β.
- MLE-2 (F1 or R²), read from the same MLE call.
- wNMIS and C2ST (XGBoost) from [april-tools/tabpc](https://github.com/april-tools/tabpc) @ `b821a47` (`third_party/tabpc`, digests checked, its metric classes called unchanged).

Deviations, all recorded in `run_meta.json` and the analysis:

- The routine evaluation every 2000 epochs is off. It has no effect on training or checkpoint selection.
- The training seed is set explicitly.
- Table k is drawn after seeding with k.
- The entry point mirrors `tabdiff/main.py` instead of calling it. `main.py:185` hard-codes `UniModMLP`, writes inside the submodule and needs wandb. A test proves the transformer model is bit-identical to the one `main.py` builds.
- wandb is replaced by `train_log.csv`.

### Trainer: TabDiff's own, except news

TabDiff's Trainer cannot resume a run: it does not checkpoint the optimizer, the EMA, the LR scheduler or the best-loss tracking. Six datasets train within one job, with **TabDiff's Trainer unchanged**.

**News** needs more than 48 h, so it trains with `resumable.ResumableTrainer` across chained jobs. That class subclasses TabDiff's Trainer, and its `run_loop` is TabDiff's own, copied verbatim (`trainer.py` lines 124–295 @ `5ecdb33`, generated from the source), plus exactly two marked hook blocks:
- before the epoch loop: restore a saved run state;
- at the end of each epoch: save the run state every 100 epochs, and pause before the job's training budget runs out.

The run state (`ckpt/resume_state.pt`) holds:
- the denoiser and the learnable noise schedules;
- the EMA denoiser and the EMA schedules;
- the AdamW state and the ReduceLROnPlateau state;
- the epoch counter, which fixes the continuous-loss annealing position, plus patience and the loss weights;
- the best-loss and best-EMA-loss tracking, together with the exact bytes of the `best_model_*` and `best_ema_model_*` files at that epoch;
- the torch CPU and CUDA, numpy and python RNG states. The DataLoader's shuffling and worker seeds come from the torch RNG, so they are restored with it.

Evidence (`tests/test_tabdiff_gnn.py`):
- `test_resumable_run_loop_is_tabdiffs_verbatim`: the loop minus the two hook blocks equals TabDiff's source byte for byte, and every other method is TabDiff's own.
- `test_resume_is_bit_identical`: on CPU, magic (512-row subset, tiny GATv2), 40 epochs paused at epoch 15 with the RNG scrambled before the resume. Model, EMA, schedules, optimizer and LR-scheduler state come out **bit-identical** to an uninterrupted run and to TabDiff's own Trainer.
- `test_resume_is_bit_identical_through_best_checkpoints` (`RUN_SLOW=1`): the same check past TabDiff's epoch-4000 threshold (4012 epochs, paused at 4004), so the best-checkpoint files and the best-checkpoint choice are compared too.
- `test_resume_on_gpu_within_run_to_run_tolerance` (cluster): TabDiff trains without deterministic CUDA algorithms, so even two uninterrupted GPU runs need not be bit-identical. The tolerance used: the resumed run's max |Δ| from an uninterrupted run must be ≤ max(10 × the max |Δ| between two uninterrupted runs, 1e-6). Both numbers are printed.

Each news `run_meta.json` records `trainer: resumable copy …` and `trainer_evidence`, and lists the deviation. `train_record.json` lists the training segments: start and end epoch, wall time and SLURM job of each part. Every other dataset records `trainer: TabDiff's Trainer, unchanged`.

## Files

| file | role |
|---|---|
| `config.py` | datasets, arms, names, trees, Route-A + study TabDiff pins, tabpc pins, deviations |
| `denoiser.py` | `build_denoiser("transformer" \| "attngnn")`, `AttnGNNDenoiser` (V2), `t_from_c_noise` |
| `topk_log.py` | `KStatsLogger`: selected-neighbour count k per node and block, in 10 bins of t ∈ [0, 1] (adaptive arms) |
| `entry.py` | thin entry point, `train` / `sample`; mirrors `tabdiff/main.py` |
| `evaluate.py` | TabDiff evaluation (main / quality) via the study's `tabdiff_eval.py` |
| `tabpc_eval.py` | wNMIS and C2ST (XGB) with tabpc's own classes (tabpc venv) |
| `resumable.py` | `ResumableTrainer` (news only): TabDiff's `run_loop` verbatim + save/restore hooks |
| `runner.py` | one job = dataset × arm × seed; stages with done-markers; `--stages` for chains; `--dry_run` also builds the denoiser on the real shapes and runs one forward pass |
| `prepare_data.py` | `data/tabdiff_gnn/tabdiff_<ds>/` from TabDiff's processed output (no `_dcr` splits); downloads and processes diabetes with TabDiff's scripts |
| `params.py` | parameter counts from the data shapes → `hpc_transfer/jobs_tabdiff_gnn/params.csv` |
| `make_jobs.py` | writes `hpc_transfer/jobs_tabdiff_gnn/*.sh`, `walltime_estimates.csv`, `hpc_transfer/submit_tabdiff_gnn.sh` |
| `tests/test_tabdiff_gnn.py` | tests (run with the eval venv's python) |
| `bench_step_ratio.py` | CPU cost of each d256 AttnGNN relative to the A100-measured reference GNN → `jobs_tabdiff_gnn/step_ratio_cpu.csv` (drives the walltime estimates) |
| `analysis/tabdiff/compare.py --route_a` | `ROUTE_A.md`: a Reference note (TabDiff reported, not reproduced; GATD primary, paper secondary; the bit-identity tests), then reported TabDiff vs computed AttnGNN d256 per metric, then parameter counts |

Results go to `$TRANSFER_ROOT/exp_tabdiff_gnn/tabdiff_<ds>/<denoiser>/<arm>/seed<s>/` (the adaptive arms put the rule in `<arm>` in place of `topk<k>`):

- `route_a_spec.json`, `run_meta.json`
- `train_record.json`, `train_log.csv`, `ckpt/{config.pkl,best_ema_model_*}`
- `samples/samples_<k>.csv`, `sample_record.json`
- `tabdiff_eval/{all_results,avg_std,mle2,quality}.csv`, `eval_meta.json`
- `tabpc_eval/{tabpc_results,avg_std}.csv`, `tabpc_meta.json`
- `topk_k_stats.csv`, `topk_k_hist.csv` (adaptive arms only)
- `stages/*.done`

## Install (cluster, `module load python/3.10`)

Training and sampling run in the **eval venv** (`venvs/tabdiff_eval310`, see `hpc_transfer/tabdiff_study/README.md`). That venv already holds TabDiff's own package set: torch 2.2.2 cu118, category_encoders, icecream, tomli, sdmetrics. Route A needs a few more packages there and in the synthcity venv, and one new venv for tabpc:

```bash
module purge; module load python/3.10
HPC_ROOT=/work/pr019/atvoulo/Rompos
# eval venv (TabDiff training env): the adaptive_entmax15 rule, TabDiff's src.dump_config, figures
$HPC_ROOT/venvs/tabdiff_eval310/bin/pip install entmax==1.3 tomli_w matplotlib
# synthcity venv: torchvision matching torch 2.2.2, installed AFTER synthcity, without touching torch
$HPC_ROOT/venvs/tabdiff_synthcity310/bin/pip install --no-deps torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
# tabpc venv (new)
python -m venv $HPC_ROOT/venvs/tabpc310
P=$HPC_ROOT/venvs/tabpc310/bin/pip
$P install --upgrade pip
$P install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu   # tabpc's pin; MI is device-independent
$P install sdmetrics==0.24.0 sdv "numpy<2.0.0" pandas scikit-learn scipy tqdm torchmetrics xgboost \
           gputil prdc category_encoders                                    # tabpc requirements.txt
```

Verified locally (Python 3.11): sdmetrics 0.24.0, sdv 1.34.3, rdt 1.22.0, xgboost 3.2.0, torchmetrics 1.9.0. The versions actually used are recorded in `tabpc_meta.json` for every run.

## Upload list (the cluster has no git history)

Commit first, then:

```bash
echo "$(git rev-parse HEAD)$(git status --porcelain --untracked-files=no | grep -q . && echo -dirty)" \
    > tabdiff_gnn/UPLOAD_COMMIT
```

```
third_party/TabDiff/                       # unchanged (already uploaded for the TabDiff study)
third_party/tabpc/                         # NEW: whole working tree, no .git needed (~0.9 MB)
tabdiff_gnn/                               # all files incl. UPLOAD_COMMIT
hpc_transfer/jobs_tabdiff_gnn/             # *.sh + common_tabdiff_gnn.sh + walltime_estimates.csv + params.csv
hpc_transfer/submit_tabdiff_gnn.sh
hpc_transfer/tabdiff_study/study.py        # pins + GNN configs
hpc_transfer/tabdiff_study/tabdiff_eval.py # CHANGED: report_tables/quality_tables + mle2.csv
hpc_transfer/configs/{adult,default,king,wilt}.sh
tab_ddpm/                                  # __init__, modules, utils, gaussian_multinomial_diffsuion,
adaptive_adjacency.py                      #   graph_denoiser, graph_builder, gnn_layers, topk_logging
data/tabdiff_gnn/                          # 7 targets + manifest.json (~67 MB; no _dcr splits)
tests/test_tabdiff_gnn.py                  # optional (also needs data/tabdiff_<ds>/tabdiff/ and analysis/tabdiff/)
```

## Cluster tests (before any submission)

```bash
cd $HPC_ROOT/thesis_project
EV=$HPC_ROOT/venvs/tabdiff_eval310/bin/python
SY=$HPC_ROOT/venvs/tabdiff_synthcity310/bin/python
TP=$HPC_ROOT/venvs/tabpc310/bin/python
GPU="--partition=gpu --gres=${GRES} --cpus-per-task=26 --mem=120G"   # GRES: see sinfo -p gpu -o "%N %G"

# 1. login node, eval venv: digests, data, every job's spec, and each arm's denoiser
#    imported + built on the real shapes + one CPU forward pass (a missing package fails here)
#    (--check_all_rules: also imports entmax and builds the denoiser with fixed, jenks and entmax15)
for d in magic shoppers adult default beijing news diabetes; do
  $EV tabdiff_gnn/runner.py --dataset $d --denoiser attngnn --exp_root $HPC_ROOT/transfer_runs --dry_run \
      --check_all_rules | grep "DRY RUN"
done
$EV -m pytest -q tests/test_tabdiff_gnn.py            # if pytest is in the venv

# 2. tabpc venv: real-vs-real (a bootstrap of real.csv as the "synthetic" table); expect wNMIS ~0.98-0.99, C2ST(XGB) ~1
mkdir -p $HPC_ROOT/rvr_shoppers/samples
$EV -c "import pandas as pd; d=pd.read_csv('data/tabdiff_gnn/tabdiff_shoppers/real.csv'); \
d.sample(len(d),replace=True,random_state=0).to_csv('$HPC_ROOT/rvr_shoppers/samples/samples_0.csv',index=False)"
$TP tabdiff_gnn/tabpc_eval.py --run_dir $HPC_ROOT/rvr_shoppers --target tabdiff_shoppers --n_runs 1

# 3. GPU timing on the A100, throw-away tree: the full stage chain on magic (50 epochs, 2 tables;
#    the smoke model keeps its final EMA because TabDiff saves best-EMA only after epoch 4000)
srun $GPU --time=01:30:00 $EV tabdiff_gnn/runner.py --dataset magic --denoiser attngnn --seed 0 \
     --exp_root $HPC_ROOT/tabdiff_gnn_timing --device cuda:0 --train_python $EV --eval_python $EV \
     --synthcity_python $SY --tabpc_python $TP --smoke_epochs 50 --n_runs 2
#    training-only probes for the long ones (news, diabetes, default)
for d in news diabetes default; do
  srun $GPU --time=01:00:00 $EV tabdiff_gnn/entry.py train --target tabdiff_$d --denoiser attngnn --seed 0 \
       --run_dir $HPC_ROOT/tabdiff_gnn_timing/${d}_probe --device cuda:0 --smoke_epochs 30
done
#    s/epoch = train_record.json train_wall_s / epochs_completed. Then re-size the jobs with the
#    measured value(s), e.g.:  python3 tabdiff_gnn/make_jobs.py --probe news=<s/epoch>   (locally)
#    news needs ceil(8000 x s/epoch / 3600 x 1.3 / 46.5) training parts; other datasets must stay
#    <= 16.6 s/epoch to fit one 48 h job (else they are flagged OVER_CAP and blocked).

# 4. resumable trainer on the A100 (news uses it): resumed vs uninterrupted within run-to-run noise
ROUTE_A_RESUME_DEVICE=cuda:0 srun $GPU --time=00:30:00 $EV -m pytest -q -s tests/test_tabdiff_gnn.py \
     -k "resume_on_gpu or resumable_run_loop or resume_is_bit_identical"
```

## Jobs and submission

The job scripts in `hpc_transfer/jobs_tabdiff_gnn/` follow `hpc_transfer/jobs/` exactly:
- the same header: `--cpus-per-task=26`, `--mem=120G`, `--partition=gpu`;
- the same **TODO(GPU)** block, with the GRES line commented out (copied verbatim from `hpc_transfer/jobs`). The gpu partition needs `--gres`, so **every sbatch line in the submission loop passes `--gres=gpu:a100:1 --mem=24G --time=<estimate>`**; command-line flags override the scripts' `#SBATCH` directives;
- the cluster template's OMP and module block;
- paths overridable from the environment, `cd` to the repo root, and one sourced function (`run_route_a` from `common_tabdiff_gnn.sh`).

There is **one script per dataset and rule**: `<ds>_tabdiff_gnn.sh` (fixed), `<ds>_tabdiff_gnn_jenks.sh`, `<ds>_tabdiff_gnn_entmax15.sh`. The training seed is `SEED` (default 0), as in `hpc_transfer/jobs`.

News has two scripts per rule: `news_tabdiff_gnn[_<rule>]_train.sh`, which uses the resumable trainer with a 46.5 h budget and a 48 h walltime, and `news_tabdiff_gnn[_<rule>]_sample_eval.sh`.

`hpc_transfer/submit_tabdiff_gnn.sh` lists the sbatch lines, every one commented out, behind an `exit 0` guard, as `submit_all.sh` does. Uncomment one block at a time:

1. AttnGNN d256 seed 0: magic, shoppers, adult, beijing.
2. AttnGNN d256 seed 0: default, news, diabetes. **News is a chain**, each step with `--dependency=afterok`:
   - `news_tabdiff_gnn_train.sh` is submitted twice. The second submission gets its own job name and log (`_p2`), because SLURM truncates a reused log. Each part trains until its budget, saves the run state and exits 0; the next part resumes.
   - Then `news_tabdiff_gnn_sample_eval.sh`.

   If training needs more parts than planned, the sample_eval job refuses to start ("stage 'train' is not done"). Submit the train script once more (it resumes), then sample_eval. Or regenerate with `--probe`.
3. Rule arm adaptive_jenks, seed 0, then 4. adaptive_entmax15, seed 0, each in the order beijing, magic, shoppers, adult, default, diabetes, news (news as the same train ×2 → sample_eval chain).
5. Seeds 1 and 2 of every arm, double-commented (`##`), only after confirmation: `SEED=1 sbatch --job-name=<ds>_tabdiff_gnn_s1 --output=… --error=…`, so the seeds don't share one log.

No TabDiff (transformer) jobs, and no DCR.

Estimate per seed (hours on one A100, before the ×1.3 walltime margin; `jobs_tabdiff_gnn/walltime_estimates.csv`):

| dataset | trainer | train | sample + 3 evals | total | job walltime |
|---|---|---|---|---|---|
| magic | TabDiff | 2.7 | 0.5 | 3.2 | 48 h |
| shoppers | TabDiff | 2.3 | 0.3 | 2.6 | 48 h |
| adult | TabDiff | 3.3 | 0.9 | 4.2 | 48 h |
| beijing | TabDiff | 5.8 | 1.2 | 6.9 | 48 h |
| default | TabDiff | 11.4 | 1.1 | 12.5 | 48 h |
| news | **resumable** | 52.8 | 3.1 | 55.9 | 48 h + 48 h + 48 h (chain) |
| diabetes | TabDiff | 11.3 | 4.0 | 15.3 | 48 h |
| **seed 0** | | | | **≈ 101** | |
| seeds 1 + 2 | | | | ≈ 201 more | |

The fixed arm's scripts keep `#SBATCH --time=48:00:00` (unchanged since its seed-0 runs); the rule scripts carry the estimate × 1.3 (min 2 h; 48 h per resumable news part), and every submission line passes that `--time`. The per-job estimate is in each header comment. The train estimate is the A100-measured reference step time × the CPU-measured cost ratio (`step_ratio_cpu.csv`). Replace it with the probe's s/epoch via `--probe`.

## Adaptive neighbour-selection arms (adaptive_jenks, adaptive_entmax15)

`--topk_mode {fixed, adaptive_jenks, adaptive_entmax15}` (runner, entry, make_jobs; default `fixed`) selects the rule in every `DynamicAdjacency`. The rules are the thesis implementations (`adaptive_adjacency.py`, `tab_ddpm/graph_builder.py`), imported unchanged, with the settings of the thesis catfix runs (`transfer_runs/exp/*/ablation_adaptive_{jenks,entmax15}_catfix/*/run_meta.json`, all 16 identical):

| setting | adaptive_jenks | adaptive_entmax15 |
|---|---|---|
| selection | plain 2-class Jenks natural-breaks split of the sorted off-diagonal scores | support of entmax-1.5 over the off-diagonal scores |
| edge weights | sigmoid(score) on kept edges | sigmoid(score) on kept edges |
| k_i range | [1, F−2] (ties kept) | [k_min, F−1] |
| use_gate_rule / delta | False / — | — |
| entmax_alpha | — | 1.5 |
| k_min | 1 | 1 |
| topk_warmup_steps | 0 | 0 |
| TOPK_LOG_EVERY | 100 | 100 |
| cat_input / use_attention | prob / False (as the fixed arm) | prob / False |

**The source config's `top_k` is not used by either rule**: with `topk_warmup_steps 0` the fixed top-k is never applied (the thesis runs record it as "used only during warm-up and in the directory name"). Here it is not in the directory name either: the rule tag replaces `topk<k>`, so the arms never collide, and `route_a_spec.json` notes it as unused.

- **Paths:** fixed unchanged, `…/v2_gcn_layers2_dim256_topk5_heads4_dimt128_noise-shared/seed0`; rules `…/v2_gcn_layers2_dim256_<jenks|entmax15>_heads4_dimt128_noise-shared/seed0`.
- **`run_meta.json`** records `topk_mode`, `rule_settings` (all of the above) and `k_logging`. The fixed arm's spec has no rule fields, so its existing `route_a_spec.json` still matches.
- **Per-t k logging** (`topk_log.py`): TabDiff passes c_noise = log σ/4, not an integer step. The logger recovers t ∈ [0, 1] by inverting Precond's ρ-7 schedule and writes the selected k (mean/min/max per node, per block) in 10 bins of t, plus the distribution of k, to `topk_k_stats.csv` / `topk_k_hist.csv`. Phase `train` = every 100th training forward; `sample` = every sampling forward. Observation only: outputs are identical with logging on or off (tested). DynamicAdjacency's own per-t logging (which assumes integer t) gets `timesteps=None` for the rule arms; the fixed arm's call is unchanged.
- **entmax** (`entmax==1.3`) is needed in the eval venv. The job preflight imports it and `--dry_run --check_all_rules` builds the denoiser with each rule, so a missing package fails on the login node.

**Cost** (`bench_step_ratio.py --rules` → `jobs_tabdiff_gnn/step_ratio_rules_cpu.csv`; CPU, batch 1024, median of 3 alternating rounds, same dataset):

| dataset | jenks train | jenks forward | entmax15 train | entmax15 forward |
|---|---|---|---|---|
| magic | ×1.00 | ×1.01 | ×1.00 | ×1.02 |
| shoppers | ×0.99 | ×1.02 | ×1.01 | ×1.06 |
| adult | ×1.01 | ×1.01 | ×1.01 | ×1.03 |
| default | ×0.99 | ×1.00 | ×1.00 | ×1.02 |
| beijing | ×1.03 | ×1.00 | ×1.01 | ×1.01 |
| news | ×1.00 | ×1.02 | ×1.06 | ×1.01 |
| diabetes | ×1.02 | ×1.05 | ×1.03 | ×1.09 |

**GPU hours per seed** (A100 estimate before the ×1.3 margin → `--time`):

| dataset | fixed | jenks | entmax15 | rule job `--time` |
|---|---|---|---|---|
| beijing | 6.9 | 7.0 | 7.0 | 10 h |
| magic | 3.2 | 3.2 | 3.2 | 5 h |
| shoppers | 2.6 | 2.6 | 2.6 | 4 h |
| adult | 4.2 | 4.2 | 4.2 | 6 h |
| default | 12.5 | 12.4 | 12.5 | 17 h |
| diabetes | 15.3 | 15.6 | 15.8 | 21 h |
| news | 55.9 | 56.3 | 58.5 | 48 h + 48 h + 5 h (chain) |
| **seed 0** | 100.6 (done) | **101.3** | **103.8** | |

New seed-0 work: 205.1 GPU-h. Seeds 1 + 2 of all three arms: 611.3 GPU-h more.

## Diagnosis of the seed-0 results (beijing, magic) — CPU, no new runs

`analysis/tabdiff/route_a_diagnosis.py` runs in two steps, using the same two venvs as the evaluation:
- `nmi` (tabpc venv): per-pair NMI, using tabpc's own binning and MI. Each table's sum is cross-checked against tabpc's `MI_l1` wNMIS.
- `report` (eval venv):
  - per-column Shape and per-pair Trend, from TabDiff's own `QualityReport` on TabDiff's reordered frame;
  - for beijing: pm2.5's marginal and its dependencies on every other column; the time × weather η²; the Spearman correlations among the numerical columns; category coverage;
  - the training diagnostics: loss curves, the selected best-EMA epoch, the late-training slope and the learned ρ/k schedules;
  - the figures, and `analysis/tabdiff/route_a_diagnosis.md`.

```bash
# on the cluster login node (CPU), after the seed-0 runs:
$TP analysis/tabdiff/route_a_diagnosis.py nmi    --exp_root $HPC_ROOT/transfer_runs
$EV analysis/tabdiff/route_a_diagnosis.py report --exp_root $HPC_ROOT/transfer_runs
# or locally, after an rsync of each run's samples/ (or tabdiff_eval/all_samples/), train_log.csv,
# train_record.json and ckpt/best_ema_model_*.pt:  --run_dir beijing=PATH magic=PATH shoppers=PATH adult=PATH
```

TabDiff's Trainer computes the EMA loss but never logs it, so the report has no EMA-loss curve. It does have the selected checkpoint's EMA loss and epoch.

## Ablation: beijing and magic with a dense graph (not part of the main protocol)

`--ablation topk_dense` keeps every setting of the main beijing run except `top_k = N − 1 = 11`. With N = 12 nodes, `DynamicAdjacency` then masks nothing and every node attends to all the others.
- **Path:** `…/tabdiff_beijing/attngnn/v2_gatv2_layers3_dim256_topk11_heads8_dimt128_noise-shared/seed0/`.
- **`run_meta.json`** records `"ablation": {"name": "topk_dense", …, "main_run_dir": …}`.
- **`ROUTE_A.md`** never reads ablation runs.
- **Job:** `hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_ablation_topk_dense.sh`, in the "ABLATION" block of the submission loop, commented out.
- **Cost:** the same as the main beijing run. GATv2 scores all node pairs and then masks, and the CPU timing ratio is 1.00. That is about 6.9 GPU-h; the job asks for 48:00:00 like every job.

**magic** (same ablation, `topk_dense`): magic's own source config (`configs/wilt.sh`: GATv2, 2 layers, 8 heads; d_model 256) with `top_k = N − 1 = 10` (main run: 3).
- **Path:** `…/tabdiff_magic/attngnn/v2_gatv2_layers2_dim256_topk10_heads8_dimt128_noise-shared/seed0/`; `run_meta.json` links the main magic run (`…_topk3_…`).
- **Job:** `hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_ablation_topk_dense.sh`, ABLATION block, `--gres=gpu:a100:1 --mem=24G --time=05:00:00`.
- **Cost:** about 3.2 GPU-h (the main magic estimate × the 1.00 dense ratio measured on beijing; magic is GATv2 too).
