# Adaptive budget on many-feature datasets (ARIS)

How many neighbours does each adaptive rule keep per node when the feature
graph is large? The manifest datasets (`hpc/`, `hpc_entmax/`) have 5–11 nodes
(gesture 32), which leaves little room between k = 1 and k = N − 1. This study
runs **Jenks, sparsemax and entmax-1.5** on the datasets with the most
features and reports the budget k against N.

**No ablation.** Each dataset gets one GATv2 configuration, taken from the
best GATv2 scenario of the swept datasets at a comparable size: 5 datasets ×
3 modes = **15 runs, one per job**.

Self-contained: nothing here modifies `hpc/` or `hpc_entmax/`. Results go to
their own tree, `wide_runs/`, never to `exp/`.

## Datasets

| dataset | train rows | N | numerical + y + categorical | k range | synthetic rows |
|---|---|---|---|---|---|
| fb-comments | 157 638 | 52 | 36 + 1 + 15 | 1..51 | 1 264 000 |
| miniboone | 83 240 | 50 | 50 + 0 + 0 | 1..49 | 664 000 |
| higgs-small | 62 751 | 28 | 28 + 0 + 0 | 1..27 | 502 000 |
| default | 19 200 | 23 | 14 + 0 + 9 | 1..22 | 76 800 |
| king | 13 832 | 21 | 17 + 1 + 3 | 1..20 | 27 600 |

N is computed exactly as `scripts/train.py` builds the graph (one node per
numerical column, plus y for regression without y conditioning, plus one per
categorical column), from the `.npy` headers. These are all datasets with
N ≥ 20 and an `ablation/<ds>_ablation.py` entry point; gesture (32) is covered
by `hpc/` + `hpc_entmax/`, and news (58) has no TabDDPM entry point.

## Configuration: GATv2, sized by the swept datasets

Best GATv2 run of each swept dataset (`tabddpm_ablation/`, best `score.mean`),
by training-set size:

| train rows | dataset | GATv2 runs | best GATv2 |
|---|---|---|---|
| 491 | diabetes | 8 | L3 d64 h4 |
| 856 | insurance | 8 | L2 d64 h4 |
| 2 672 | abalone | 8 | L3 d64 h4 |
| 3 096 | wilt | 4 | L2 d64 h8 |
| 6 400 | churn2 | 6 | L2 d128 h4 |
| 13 209 | california | 2 | **L3 d128 h8** |

Width grows with size (d_model 64 up to ~3k rows, 128 from ~6k), and at the
largest swept size (california, 13.2k rows) three blocks with eight heads won
— L3 beat L2 by 0.014, about 7× the seed std. **Every dataset here has
≥ 13.8k rows, so all five fall in that bracket and take the same
configuration:**

    gatv2, 3 blocks, d_model 128, 8 heads, no dense attention
    -> gnn_gatv2_layers3_dim128_topk5_heads8_noattn

Nothing wider is used for the 60k+ row datasets: no GATv2 run supports it, and
it would roughly double the cost. `top_k` only matters during warm-up (off)
and in the directory name. gesture's single GATv2 run is not a sweep and is
left out. The table and the choice live in `wide_manifest.py` (`CONFIGS`).

Training hyperparameters are the same as every other ablation run
(`ablation_runner`: 20 000 steps, lr 1e-4, batch 4096); the diffusion,
normalisation and sampling settings are each dataset's own tuned config.

## Two config fixes (outside `hpc_wide/`)

- **`ablation/default_ablation.py`**: `num_numerical_features` was 20
  (TabDDPM's config value), but `data/default/` has 14 numerical + 9
  categorical columns. `sample.py` splits the generated matrix at that index,
  so every `default` run crashed at sampling, after the full training. Now 14 —
  the value the Kaggle notebook already used. The preflight checks config vs
  data width for every dataset before anything runs.
- **`ablation/fb_comments_ablation.py`**: `sample_batch_size` 150 000 → 50 000.
  TabDDPM sized it for its MLP; with the graph denoiser a 150 000-row sampling
  batch peaks at ~38 GiB (measured on CPU, one forward, entmax15: 4.0 GiB +
  0.23 GiB per 1 000 rows), ~15 GiB at 50 000. Only the batching of sampling
  changes — `sample_all` trims to `num_samples` either way.

Measured peak memory at the configured sampling batch (CPU, one forward,
entmax15 — the largest of the three rules): higgs-small 11.1 GiB,
miniboone 10.1, king 5.9, default 5.6, fb-comments ~15 (after the fix);
training step at batch 4096 (fb-comments): 10.8 GiB.

## Protocol

`PROTOCOL` (default `budget`):

| | training | seeds | synthetic rows | results tree |
|---|---|---|---|---|
| `budget` | full 20 000 steps | 1 generation × 1 classifier | min(num_samples, 50 000) | `wide_runs/budget/exp/<ds>/ablation_<mode>/<combo>/` |
| `full` | full 20 000 steps | 5 × 10 (module defaults) | the dataset's own `num_samples` | `wide_runs/full/exp/...` |

`budget` is enough for the question: `topk_stats.csv` holds the complete
training and sampling budgets after one sampling pass, and the per-t-bucket k
curves stabilise long before 1.26 M rows. Its metrics are single-seed, on the
capped synthetic set, and **not comparable to the 5 × 10 protocol** — which is
why each protocol has its own tree.

The cap is applied in memory by `run_combo.py`, which is
`ablation/<ds>_ablation.py` (same parser, same single combination, same
`run_ablation` call) with a copy of `DATASET_CFG` whose `num_samples` is
capped (and `sample_batch_size` lowered to it when larger, so no rows are
generated only to be trimmed). No file under `ablation/` is changed for it;
without the cap it hands `run_ablation` exactly what the dataset script does
(verified). `run_meta.json` records `num_samples_config`,
`num_samples_used`, `num_samples_cap` and `sample_batch_size_used`:

| dataset | num_samples (config) | budget: used | budget: sampling batch |
|---|---|---|---|
| fb-comments | 1 264 000 | 50 000 | 50 000 |
| miniboone | 664 000 | 50 000 | 20 000 |
| higgs-small | 502 000 | 50 000 | 50 000 (config 60 000) |
| default | 76 800 | 50 000 | 10 000 |
| king | 27 600 | 27 600 | 20 000 |

## Submit

Needs, on the cluster: the adaptive-mode code (`adaptive_adjacency.py`,
`tab_ddpm/graph_builder.py`, `ablation/ablation_runner.py`), the `_encode` fix
(`scripts/eval_tabddpm_protocol.py`), the two config fixes above, and `entmax`
in the venv (only the entmax modes need it; see `hpc_entmax/README.md`).

```bash
cd /work/pr019/atvoulo/Rompos/thesis_project
module load python/3.10
PY=/work/pr019/atvoulo/Rompos/venvs/tabddpm310/bin/python
$PY hpc_wide/wide_manifest.py --repo_root "$PWD" --summary     # every line must end in OK
```

Estimated walltime per job: see **Walltime** below. Start small:

**1. king, Jenks only**

```bash
./hpc_wide/submit_all.sh adaptive_jenks king
```
Check the head of `jobs_out/wide_king_adaptive_jenks_budget_<jobid>.out`:
`protocol : budget`, `graph : N = 21 nodes`, `preflight : OK`,
`1 run(s) for king`. When it finishes,
`wide_runs/budget/exp/king/ablation_adaptive_jenks/gnn_gatv2_layers3_dim128_topk5_heads8_noattn/`
has `results_full_averaged.json` and a `topk_stats.csv` with `sample` rows.
Note its wall time — the others scale from it by the "× king" column under **Walltime**.

**2. The rest**

```bash
./hpc_wide/submit_all.sh adaptive_sparsemax,adaptive_entmax15 king
./hpc_wide/submit_all.sh all default higgs-small
./hpc_wide/submit_all.sh all miniboone fb-comments
squeue --me -o "%.10i %.16j %.8T %.10M %R"
```

Other forms: `./hpc_wide/submit_all.sh all all` (15 jobs),
`PROTOCOL=full ./hpc_wide/submit_all.sh all default`,
`DRY_RUN=1 ./hpc_wide/submit_all.sh all all` (print only). Job names are
`<tag>_<dataset>` with tag `jnk` / `spm` / `e15`; logs are
`jobs_out/wide_<dataset>_<mode>_<protocol>_<jobid>.{out,err}`.

Each job `cd`s to `$REPO_ROOT` itself: the evaluators read `tuned_models/`
relative to the working directory, and a SLURM job's working directory is
wherever `sbatch` ran.

Resume: resubmit the same command — a run with `results_full_averaged.json`
is skipped. A failing run is appended to
`hpc_wide/logs/<ds>_<mode>_<protocol>_failures.txt`.

**Plumbing test without a real run:** `SMOKE_STEPS=50 TOPK_MODE=adaptive_jenks
sbatch hpc_wide/jobs/king_wide.sh` trains for 50 steps and writes to
`wide_runs/smoke/`, which nothing reads and which can never be resumed as a
real result.

## Walltime (budget protocol, per job, A100)

| dataset | N | training | sampling | eval | **total, optimistic – pessimistic** | × king |
|---|---|---|---|---|---|---|
| fb-comments | 52 | 4.5 h | 0.55 h | ~0.5 h | **5.6 – 13.9 h** | 4.3 – 6.5 |
| miniboone | 50 | 4.2 h | 0.51 h | ~0.5 h | **5.2 – 13.0 h** | 4.0 – 6.0 |
| higgs-small | 28 | 1.1 h | 0.16 h | ~0.3 h | **1.5 – 3.8 h** | 1.7 – 1.8 |
| default | 23 | 0.7 h | 0.11 h | ~0.2 h | **1.0 – 2.6 h** | 1.2 – 1.4 |
| king | 21 | 0.6 h | 0.05 h | ~0.2 h | **0.9 – 2.1 h** | 1 |

None is expected to pass 20 h. **Without the row cap**, fb-comments would be
18.9 h and miniboone 11.4 h at the *optimistic* end — both over 20 h at the
pessimistic end — which is why the budget protocol caps the synthetic rows.

How these are estimated:
- GATv2 dominates: per layer it materialises a (B, H, N, N, d_head) pairwise
  tensor (B·N²·128 elements) and makes ~6 memory-bound passes over it (add,
  LeakyReLU, multiply, sum). Per row and forward pass that is 51.6 MB for
  fb-comments (N = 52) down to 8.7 MB for king (N = 21), over 3 blocks × 2
  layers; the adjacency rules and linear layers add < 3 %.
- Training = 20 000 steps × 4 096 rows × (forward + checkpoint recompute +
  backward ≈ 4× a forward; 5× for fb-comments and miniboone, whose pairwise
  tile at batch 4 096 exceeds the 2 GiB budget and is checkpointed again).
  Sampling = rows × 1 000 diffusion steps × one forward.
- Optimistic = that traffic at 1.3 TB/s (A100 40 GB, large elementwise
  kernels); pessimistic = 2.5× slower. Eval (CatBoost fit, DCR/MIA on ≤ 50 000
  synthetic rows) is a rough allowance.
- "× king": measured per-row costs on CPU (sampling forward and training step,
  final model, both modes within 3 %) / the traffic model. The GPU, being
  purely bandwidth-bound, should sit nearer the model's ratio.

**Calibrate on king**: its Jenks job is the first submission. If it takes more
than ~3 h, fb-comments could pass 20 h — hold fb-comments and miniboone and
lower `BUDGET_MAX_NUM_SAMPLES` in `common.sh` before submitting them.

## Read the budgets

```bash
python3 hpc_wide/budget_summary.py --repo_root "$PWD"
```

Reads every `topk_stats.csv` under `wide_runs/budget|full/` and, by default,
the manifest runs under `exp/` (wilt … gesture, `--no_main` to skip), so one
table covers N from 5 to 52. k is the number of kept off-diagonal edges per
node, averaged over nodes and samples; aggregates are `n_samples`-weighted.
Written to `hpc_wide/`:

- **`budget_runs.csv`** — one row per run: N, k at the start / end of
  training, k when sampling (overall, at t near 0, at t near T), k / (N − 1).
- **`budget_curves.csv`** — sampling-phase k per run × block × t-bucket.
- **`budget_training.csv`** — training-phase k per run × logged step.

## Files

| File | Purpose |
|---|---|
| `wide_manifest.py` | Datasets, the GATv2 configuration + evidence, N, config/data check |
| `jobs/<ds>_wide.sh` | One SLURM job per dataset; `TOPK_MODE` (default jenks), `PROTOCOL` (default budget) |
| `common.sh` | Mode/protocol checks, preflight, runner |
| `submit_all.sh` | `<mode[,mode]\|all> <dataset...\|all>` → one job per pair |
| `budget_summary.py` | k vs N tables and curves |
| `logs/` | failure logs, written at runtime |
