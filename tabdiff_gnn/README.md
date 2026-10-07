# Route A — AttnGNN as the denoiser inside TabDiff

TabDiff ([MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff) @ `5ecdb33`, `third_party/TabDiff`,
**never modified**, digests checked) runs through its own classes:

- `UnifiedCtimeDiffusion`: EDM for the numerical columns, masked diffusion for the categorical ones, learnable per-column schedules.
- `Model`/`Precond`.
- `Trainer`: training loop, EMA, best-EMA checkpoint after epoch 4000, 50-step sampler.
- `TabDiffDataset` and `TabMetrics`.

One thing is substituted: the denoiser. **One configuration is run: AttnGNN V2 with `d_model` 256 on all 7 datasets** (magic, shoppers, adult, default, beijing, news, diabetes).

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
| `denoiser.py` | `build_denoiser("transformer" \| "attngnn")`, `AttnGNNDenoiser` (V2) |
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

Results go to `$TRANSFER_ROOT/exp_tabdiff_gnn/tabdiff_<ds>/<denoiser>/<arm>/seed<s>/`:

- `route_a_spec.json`, `run_meta.json`
- `train_record.json`, `train_log.csv`, `ckpt/{config.pkl,best_ema_model_*}`
- `samples/samples_<k>.csv`, `sample_record.json`
- `tabdiff_eval/{all_results,avg_std,mle2,quality}.csv`, `eval_meta.json`
- `tabpc_eval/{tabpc_results,avg_std}.csv`, `tabpc_meta.json`
- `stages/*.done`

## Install (cluster, `module load python/3.10`)

Training and sampling run in the **eval venv** (`venvs/tabdiff_eval310`, see `hpc_transfer/tabdiff_study/README.md`). That venv already holds TabDiff's own package set: torch 2.2.2 cu118, category_encoders, icecream, tomli, sdmetrics. The synthcity venv is unchanged. One new venv is needed for tabpc:

```bash
module purge; module load python/3.10
HPC_ROOT=/work/pr019/atvoulo/Rompos
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
GPU="--partition=gpu --gres=gpu:a100:1 --cpus-per-task=26 --mem=24G"

# 1. login node, eval venv: digests, data, every job's spec, and each arm's denoiser
#    imported + built on the real shapes + one CPU forward pass (a missing package fails here)
for d in magic shoppers adult default beijing news diabetes; do
  $EV tabdiff_gnn/runner.py --dataset $d --denoiser attngnn --exp_root $HPC_ROOT/transfer_runs --dry_run | grep "DRY RUN"
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

## Submission

`hpc_transfer/submit_tabdiff_gnn.sh`: every `sbatch` line is commented out. Uncomment one block at a time:

1. AttnGNN d256 seed 0: magic, shoppers, adult, beijing.
2. AttnGNN d256 seed 0: default, news, diabetes. **News is a chain of three jobs**, each submitted with `--dependency=afterok`:
   - `gnn_news_s0_p1.sh`: training, 48 h walltime. It pauses and saves its run state at a 46.5 h training budget, then exits cleanly so the next part can start.
   - `gnn_news_s0_p2.sh`: training resumes, 48 h walltime.
   - `gnn_news_s0_p3.sh`: sample + evaluate, 5 h.

   If training needs more parts than planned, `p2` pauses as well and `p3` refuses to start ("stage 'train' is not done"). Resubmit `p2`, which resumes again, then `p3`, or regenerate with `--probe`.
3. Seeds 1 and 2, double-commented (`##`), only after confirmation.

No TabDiff (transformer) jobs, and no DCR.

Estimate per seed (hours on one A100, before the ×1.3 walltime margin; `jobs_tabdiff_gnn/walltime_estimates.csv`):

| dataset | trainer | train | sample + 3 evals | total | jobs |
|---|---|---|---|---|---|
| magic | TabDiff | 2.7 | 0.5 | 3.2 | 5 h |
| shoppers | TabDiff | 2.3 | 0.3 | 2.6 | 4 h |
| adult | TabDiff | 3.3 | 0.9 | 4.2 | 6 h |
| beijing | TabDiff | 5.8 | 1.2 | 6.9 | 10 h |
| default | TabDiff | 11.4 | 1.1 | 12.5 | 17 h |
| news | **resumable** | 52.8 | 3.1 | 55.9 | 48 h + 48 h + 5 h (chain) |
| diabetes | TabDiff | 11.3 | 4.0 | 15.3 | 20 h |
| **seed 0** | | | | **≈ 101** | |
| seeds 1 + 2 | | | | ≈ 201 more | |

The train estimate is the A100-measured reference step time × the CPU-measured cost ratio (`step_ratio_cpu.csv`). Replace it with the probe's s/epoch via `--probe`.
