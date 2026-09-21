# Adaptive top-k re-run on ARIS

One SLURM job per dataset. Each job runs that dataset's adaptive-top-k
combinations sequentially on a single GPU, then you collect everything into
CSVs and merge them with the existing fixed-top-k baseline yourself.

Everything here writes under `HPC_ROOT`. Nothing writes to another user's
directory.

---

## 1. What has to be filled in before submitting

There is exactly one thing: **the `--gres` string**.

`gpu:a100:1` is rejected on this cluster (`invalid GRES format or GPU type`).
Find the accepted value:

```bash
sinfo -p gpu -o "%N %G"
```

Then either pass it at submit time (preferred, one place):

```bash
GRES=gpu:1 ./hpc/submit_all.sh
```

or uncomment the `#SBATCH --gres=` line in each `hpc/jobs/<dataset>_ablation.sh`
(clearly marked `TODO(GPU)`) and submit with `ALLOW_NO_GRES=1`.

A command-line `--gres` overrides the `#SBATCH` directive, so the submit-time
route means you only have to get it right once. `submit_all.sh` refuses to
submit without one, because a 24-hour job that silently lands without a GPU
wastes the whole allocation.

**If submission is rejected for a missing account**, uncomment this line in the
job scripts (association: `account=pr019022_gpu`, partitions `gpu` and `mig`):

```bash
##SBATCH --account=pr019022_gpu
```

## 2. Paths

Every script defines these once at the top and honours the environment:

| Variable | Default |
|---|---|
| `HPC_ROOT` | `/work/pr019/atvoulo/Rompos` |
| `REPO_ROOT` | `$HPC_ROOT/thesis_project` |
| `PYTHON_BIN` | `$HPC_ROOT/venvs/tabddpm310/bin/python` |

One unavoidable exception: the `#SBATCH --output` / `--error` directives spell
the path out literally, because SLURM parses those lines before any shell
expansion — they cannot reference `$HPC_ROOT`. If you move `HPC_ROOT`, update
those two lines in each job script too. They are the only duplicated path.

Results land in `$REPO_ROOT/exp/<dataset>/ablation_adaptive_jenks/<combo>/`,
deliberately a sibling of the baseline's `ablation/` tree so the two can never
be confused.

## 3. Smoke test first

GPU visibility from a compute node has not been confirmed yet, so
`smoke_test.sh` checks that **before** it runs anything: it prints
`torch.__version__`, `torch.cuda.is_available()` and
`torch.cuda.get_device_name(0)`, and aborts if no GPU is visible.

```bash
salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=26 \
       --mem=120G --time=01:00:00 --gres=<the string sinfo reports>

./hpc/smoke_test.sh              # wilt, first combination, 50 steps, 1x1 seeds
./hpc/smoke_test.sh gesture      # another dataset
GPU_CHECK_ONLY=1 ./hpc/smoke_test.sh   # just the GPU report
```

It goes through exactly the same code path as the real jobs, but writes to
`$HPC_ROOT/smoke_test_out` so it can never be mistaken for a real result. It
prints the elapsed time, which is what you scale to estimate a real run
(× steps/50 for training, × 50 for the 5 × 10 seed protocol).

## 4. Submit

```bash
GRES=gpu:1 ./hpc/submit_all.sh                  # all 7, cheapest first
GRES=gpu:1 ./hpc/submit_all.sh wilt buddy       # a subset
DRY_RUN=1 ./hpc/submit_all.sh                   # print sbatch lines only
```

Datasets are submitted cheapest-first, so a short job returns a measured
runtime before the long ones are committed:

| Order | Dataset | Adaptive runs | Baseline combos collapsed |
|---|---|---|---|
| 1 | buddy | 2 | 2 |
| 2 | gesture | 2 | 2 |
| 3 | wilt | 4 | 4 |
| 4 | insurance | 8 | 16 |
| 5 | abalone | 10 | 19 |
| 6 | churn2 | 10 | 10 |
| 7 | diabetes | 12 | 24 |
| | **total** | **48** | **77** |

The `gpu` partition has only two nodes (`a02`, `a03`), so queue time can be
long. That is why this is one job per dataset rather than many small ones.

## 5. Resume after a timeout

Just resubmit the same script. Each combination is skipped when its
`results_full_averaged.json` already exists.

**Resume granularity is one whole combination.** `ablation_runner` passes
`checkpoint_every=0`, so training writes no intermediate checkpoints: a
combination interrupted by the 24-hour walltime restarts from step 0. Only the
in-progress combination is lost — everything already finished is skipped, and
the job picks up where it left off.

A combination that fails does not kill the job. The failure is appended to
`hpc/logs/<dataset>_failures.txt` and the loop moves on.

## 6. Warm-up

`--topk_warmup_steps` is set from a single variable at the top of each job
script, default `0` (adaptive from step 0):

```bash
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"
```

Change it there, or override at submit time without editing anything:

```bash
TOPK_WARMUP_STEPS=2000 GRES=gpu:1 ./hpc/submit_all.sh
```

The counter is in **optimizer-step units**: `Trainer.run_loop` performs exactly
one model forward per step, verified empirically at a ratio of 1.000.

## 7. Collect

```bash
python3 hpc/collect_results.py --repo_root "$REPO_ROOT"
```

Writes three files into `hpc/`:

- **`results.csv`** — one row per adaptive run. The column layout matches
  `ablation_runner.summary_to_csv()`, the same flattening the baseline CSV
  uses, so the two concatenate directly. Adaptive-only columns
  (`topk_mode`, `baseline_top_ks`, `k_min`, `topk_warmup_steps`,
  `k_mean_train`, `k_mean_sample`, `gpu_name`, `git_commit`) are appended
  after the metric columns, where they come out empty for baseline rows.
- **`topk_curves.csv`** — every run's `topk_stats.csv` concatenated at full
  step resolution, for the budget-vs-t figure.
- **`topk_budget_summary.csv`** — mean adaptive budget per
  (dataset, block, t_bucket), weighted by `n_samples`. Also printed.

It reports missing and failed runs rather than silently skipping them, and
exits non-zero if either list is non-empty.

## 8. What each run actually executes

Per combination, literally:

```bash
srun "$PYTHON_BIN" "$REPO_ROOT/ablation/<dataset>_ablation.py" \
    --gnn_type <t> --n_layers <l> --d_model <d> --top_k <k> --n_heads <h> \
    --use_attention 0 \
    --topk_mode adaptive_jenks --k_min 1 --topk_warmup_steps 0 \
    --topk_log_every 100 \
    --repo_root "$REPO_ROOT" --data_root "$REPO_ROOT" --exp_root "$REPO_ROOT" \
    --device cuda:0 --seed 0
```

Two things about this command are load-bearing:

- `--use_attention 0` is **mandatory**. The per-dataset scripts build a
  five-element `single_combo`, and `run_ablation` defaults the missing
  attention axis to `True` — which would change both the architecture and the
  directory name away from the baseline's `_noattn`.
- `--data_root` is the **repo root**, not the `data/` directory.
  `ablation_runner` joins it with the per-dataset config's relative
  `data/<dataset>/` path.

`--steps`, `--n_gen_seeds`, `--n_clf_seeds`, `--evaluator` and
`--tabdiff_fidelity` are deliberately **not** passed, so the run inherits the
module constants the baseline was produced with: 20 000 steps, 5 generation
seeds × 10 classifier seeds, CatBoost, fidelity suite off.

## 9. Which combinations run

Taken from `tabddpm_ablation/*.json` — the baseline summaries — via
`ablation/existing_results.load_manifest()`, never a re-derived cartesian
product (several datasets were swept over different, partly incomplete grids).

In adaptive mode `top_k` is unused except during warm-up, so baseline
combinations differing only in `top_k` collapse onto one adaptive run. The
dedup key is `(gnn_type, n_layers, d_model, n_heads, use_attention)`; the
representative `top_k` is `min()` over the group, and every collapsed value is
recorded in `run_meta.json` as `baseline_top_ks` so you can see what each run
is comparable against.

```bash
python3 hpc/adaptive_manifest.py --repo_root "$REPO_ROOT" --summary
```

No new run-definition file is introduced; the baseline summaries stay the
single source of truth.

## 10. Environment

Verified on the cluster:

| Component | Version |
|---|---|
| Python | 3.10.15 (`module load python/3.10`) |
| torch | 2.1.2+cu118 |
| numpy | 1.23.5 |

`module load python/3.8` does **not** work: `lib/metrics.py` uses `list[str]`
annotations, which raise `TypeError: 'type' object is not subscriptable` on
3.8.

`rtdl` and `libzero` are installed with `--no-deps` because both pin
`torch<2`; both were verified working on torch 2.1.2.

`--mem=120G` is the maximum the partition accepts (`128G` is rejected: *"You
need to ask for upto 126976 Mb"*). `--nodes`, `--ntasks` and `--mem` must all
be present or ARIS rejects the job.

## 11. Files

| File | Purpose |
|---|---|
| `jobs/<dataset>_ablation.sh` | One SLURM job per dataset |
| `common.sh` | Shared per-combination runner, sourced by the job scripts |
| `adaptive_manifest.py` | Deduped run list derived from the baseline summaries |
| `submit_all.sh` | Submit jobs, print job ids, optional dataset subset |
| `smoke_test.sh` | GPU check, then one combination with a tiny budget |
| `collect_results.py` | Build `results.csv`, `topk_curves.csv`, `topk_budget_summary.csv` |
| `logs/` | `<dataset>_failures.txt`, written at runtime |
