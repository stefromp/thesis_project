# Transferred configurations for the un-ablated datasets (ARIS)

The GNN ablation (`tabddpm_ablation/`) covers 9 of the 16 TabDDPM datasets.
This directory predicts a configuration for the remaining 7 from those results,
and generates SLURM jobs to run it under the ablation's own protocol, so the
new numbers drop straight into the same table.

**39 job scripts, 39 runs, one configuration per job. What varies across a
dataset's jobs is only the k-selection rule.**

| rule | alpha | results tree | job script |
|---|---|---|---|
| `fixed` | — | `exp/<ds>/ablation/` | `<ds>_transfer.sh` (7 new datasets) |
| `adaptive_jenks` | — | `exp/<ds>/ablation_adaptive_jenks/` | `<ds>_transfer_jenks.sh` / `<ds>_jenks.sh` |
| `adaptive_entmax15` | 1.5 | `exp/<ds>/ablation_adaptive_entmax15/` | `<ds>_entmax.sh` (all 16) |

`ablation_runner` derives each sibling tree from `--topk_mode` itself
(`ablation_subdir = "ablation" if topk_mode == "fixed" else f"ablation_{topk_mode}"`),
so the rules of one configuration can never collide.

`adaptive_sparsemax` (alpha 2.0) is **not** part of this study — no config lists
it in `RUN_MODES` and no job script sets it.

### `entmax` is not in `requirements.txt`

entmax-1.5 needs it:

```bash
pip install entmax==1.3
```

`common.sh` preflights it before creating anything — it imports `entmax`,
checks `--topk_mode` is accepted, and drives one `DynamicAdjacency` forward in
the requested mode, naming the file to upload if any check fails. A missing
dependency therefore costs no allocation. Ported from `hpc_entmax/common.sh`.
Check it once interactively first:

```bash
./hpc_transfer/smoke_test.sh wilt adaptive_entmax15
```

### The two groups

- **Group A — the 9 ABLATED datasets (18 runs, 2 adaptive rules).**
  Each runs **its own ablation winner**, chosen by the repo's own
  `ablation/existing_results.load_best()` (highest `metrics.score.mean`:
  accuracy for classification, negative RMSE for regression). Their fixed
  baselines already exist in `tabddpm_ablation/`, so no `fixed` run is needed —
  and because each job keeps that configuration's own `top_k`, its output
  directory name matches the baseline exactly, so the adaptive/fixed pair
  lines up by name. Verified for all 9 by `preflight.py`.
- **Group B — the 7 NEW datasets (21 runs, all 3 rules).** No ablation results
  exist for these, so the configuration is the one predicted in
  `predictions.md`. The `fixed` run is the baseline they never had; the two
  adaptive rules are the comparison.

### Different scope from `hpc/` and `hpc_entmax/` — do not run both for the same (dataset, rule)

| study | datasets | rules | scope |
|---|---|---|---|
| `hpc/` | 7 ablated (excludes adult, california) | jenks | every recorded combo, 48 runs |
| `hpc_entmax/` | 7 ablated (same 7) | sparsemax, entmax15 | every recorded combo |
| `hpc_transfer/` Group A | **all 9** ablated | jenks, entmax15 | **1 run per dataset**, 18 runs |
| `hpc_transfer/` Group B | 7 new | fixed, jenks, entmax15 | 1 run per dataset, 21 runs |

All of them write `exp/<ds>/ablation_<mode>/<combo>/`. For a given
(dataset, rule) submit **one** study, not two. `hpc/` and `hpc_entmax/` are the
fuller studies over their 7 datasets; Group A is the narrow one that covers all
9, including the two `hpc/adaptive_manifest.py` excludes.

**Nothing here has been submitted.** `submit_all.sh` prints a message and
exits; every `sbatch` line in it is commented out.

---

## 1. What was decided, and how

Full derivation: **[predictions.md](predictions.md)**. In short:

- Leave-one-out on the ablated datasets says **nearest-neighbour transfer does
  not work here**. Mean utility regret: global best 0.158, per-axis vote 0.484,
  nearest-neighbour 0.714. That holds across every setting tried (donor-pool
  filter, dependency weight 1/2/4, both definitions of "best config"), so the
  global-best strategy is used for all seven targets.
- Scoring is on **utility only** (r2 for regression; accuracy, macro-F1 and
  ROC-AUC for classification). The ablation was run with
  `TABDIFF_FIDELITY = False` and the CatBoost evaluator, so Shape/Trend/C2ST/
  alpha-precision/beta-recall and XGBoost MLE do not exist for these runs.
- The `d_model` axis was **never saturated**: in 8 of 9 ablated datasets the
  best `d_model` is the largest value that dataset happened to test. The global
  best's `d64` is a ceiling of the shared abalone/diabetes/insurance sub-grid,
  not a measured optimum, and every target is larger than every donor.

Every target therefore runs the same configuration, in both modes:

    gatv2, 3 blocks, d_model 128, top_k 5, 8 heads, no dense attention
    -> gnn_gatv2_layers3_dim128_topk5_heads8_noattn

It is a configuration that was **actually measured** in the ablation: best on
churn2 and inside california's 1-sigma tie set, the two largest donors. `k5`
and `h8` are `d128`'s measured partners and are not mixed with `d64`'s `k3/h4`,
so no axis combination here is one that was never run. `hpc_wide/` reached the
same configuration independently, by a train-size bracket argument, for five of
these seven datasets.

## 2. The jobs

| job | mode | results tree | what it is |
|---|---|---|---|
| `jobs/<ds>_transfer.sh` | `fixed` | `exp/<ds>/ablation/<combo>/` | the baseline these datasets never had; directly comparable to `tabddpm_ablation/` |
| `jobs/<ds>_transfer_jenks.sh` | `adaptive_jenks` | `exp/<ds>/ablation_adaptive_jenks/<combo>/` | the adaptive-budget run; directly comparable to the `hpc/` re-run on the ablated datasets |
| `jobs/<ds>_jenks.sh` (9 ablated) | `adaptive_jenks` | `exp/<ds>/ablation_adaptive_jenks/<combo>/` | that dataset's own ablation winner, paired by name with its existing fixed baseline |

Every job runs exactly one configuration — there is no combination loop
anywhere in this directory. Each config file declares which modes it is run in
via `RUN_MODES`, and `collect_results.py` and `preflight.py` both read that
rather than assuming.

The Group A configurations come from `best_manifest.py`, a thin wrapper over
`ablation/existing_results.load_best()`. Nothing in `hpc/` is read differently
or modified.

The two job scripts differ in **exactly one functional line**, `TOPK_MODE`, and
share `configs/<ds>.sh` — the architecture is identical, only the top-k rule
changes, which is the point of the comparison. `ablation_runner` derives the
sibling results tree from `--topk_mode` itself
(`ablation_subdir = "ablation" if topk_mode == "fixed" else f"ablation_{topk_mode}"`),
so a fixed run and an adaptive run of the same combination can never collide.

In adaptive mode `top_k` is unused except during warm-up (default 0) and in the
directory name; it is kept at 5 so both runs of a dataset carry the same
`<combo>` name and line up in the tables.

The cluster configuration — every `#SBATCH` directive, the module loads, the
`OMP_NUM_THREADS` block and the `HPC_ROOT`/`REPO_ROOT`/`PYTHON_BIN` defaults —
is the same as `hpc/jobs/<ds>_ablation.sh`.

## 3. Protocol

Identical to the ablation, deliberately. `common.sh` does **not** pass
`--steps`, `--n_gen_seeds`, `--n_clf_seeds`, `--evaluator` or
`--tabdiff_fidelity`, so each run inherits the module constants the ablation
was produced with: 20 000 steps, 5 generation x 10 classifier seeds, CatBoost,
fidelity suite off. `--topk_mode` is left at its default `fixed`.

Two flags are load-bearing and carried over from `hpc/common.sh` unchanged:

- `--use_attention 0` is **mandatory**. The per-dataset scripts build a
  five-element `single_combo`, and `run_ablation` defaults the missing
  attention axis to `True`, which would change both the architecture and the
  directory name away from the ablation's `_noattn`.
- `--data_root` is the **repo root**, not `data/`. `ablation_runner` joins it
  with the per-dataset config's relative `data/<dataset>/` path.

## 4. Paths

| Variable | Default |
|---|---|
| `HPC_ROOT` | `/work/pr019/atvoulo/Rompos` |
| `REPO_ROOT` | `$HPC_ROOT/thesis_project` |
| `PYTHON_BIN` | `$HPC_ROOT/venvs/tabddpm310/bin/python` |
| `TRANSFER_ROOT` | `$HPC_ROOT/transfer_runs` |

Results land in `$TRANSFER_ROOT/exp/<dataset>/ablation/<combo>/`. **This study
never reads or writes `$REPO_ROOT/exp/`**, so it cannot disturb the ablation or
the adaptive-top-k runs — the same isolation `hpc_wide/` uses.

As in `hpc/`, the `#SBATCH --output`/`--error` directives spell the path out
literally, because SLURM parses them before any shell expansion. If you move
`HPC_ROOT`, update those two lines in each job script too.

## 5. Before submitting

**Fill in `--gres`.** `gpu:a100:1` is rejected on the `gpu` partition
(`invalid GRES format or GPU type`). Find the accepted value:

```bash
sinfo -p gpu -o "%N %G"
export GRES=gpu:1          # whatever sinfo reports
```

If submission is rejected for a missing account, uncomment
`##SBATCH --account=pr019022_gpu` in the job scripts.

**Run the preflight and the smoke test:**

```bash
python3 hpc_transfer/preflight.py          # no GPU needed; runs anywhere

salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=26 \
       --mem=120G --time=01:00:00 --gres="$GRES"
./hpc_transfer/smoke_test.sh                                  # king, fixed
./hpc_transfer/smoke_test.sh king adaptive_jenks
./hpc_transfer/smoke_test.sh wilt adaptive_entmax15           # cheapest entmax check
./hpc_transfer/smoke_test.sh fb-comments adaptive_entmax15    # the widest graph
GPU_CHECK_ONLY=1 ./hpc_transfer/smoke_test.sh
```

`preflight.py` checks, per target: the config parses; the entry point exists;
`num_numerical_features` matches the real width of `X_num_train.npy`; `d_model`
is divisible by `n_heads`; the node count matches `meta_features.csv`; and a
real `GraphAwareDenoiser` at the configured size does a forward **and**
backward pass on the true cardinalities. That third check is the one that
matters: a width mismatch is what made every `default` run crash at sampling,
*after* the full 20 000 training steps. It also checks that both job scripts
exist for each dataset and that each declares the `TOPK_MODE` it claims.

**Smoke-test in every rule you intend to submit.** The adaptive path goes
through `adaptive_adjacency.build_adjacency`, which the fixed path never
touches, and entmax-1.5 additionally needs the `entmax` package.

## 6. Walltime — read this before submitting

From `resource_estimates.py` (the traffic model documented in
`hpc_wide/README.md`, applied to this study's 5-generation-seed protocol).
The model reproduces `hpc_wide`'s own published fb-comments figure at its
budget settings, which is what makes the extrapolation worth anything.

Per job. The adaptive rule costs under 3 % on top of the same architecture, so
one row covers both modes — but note that means **two jobs** per dataset, so
the totals below are per job, not per dataset.

| dataset | N | synthetic rows | training | sampling (x5) | eval | **total, opt – pess** | fits 24 h? |
|---|---|---|---|---|---|---|---|
| king | 21 | 27 600 | 0.6 h | 0.3 h | 1.0 h | **1.8 – 3.1 h** | yes |
| house | 17 | 116 000 | 0.4 h | 0.7 h | 1.0 h | **2.1 – 3.7 h** | yes |
| default | 23 | 76 800 | 0.7 h | 0.8 h | 1.0 h | **2.5 – 4.8 h** | yes |
| cardio | 11 | 360 000 | 0.2 h | 0.9 h | 1.5 h | **2.5 – 4.1 h** | yes |
| higgs-small | 28 | 502 000 | 1.1 h | 8.0 h | 1.5 h | **10.6 – 24.2 h** | marginal |
| miniboone | 50 | 664 000 | 4.2 h | 33.8 h | 2.5 h | **40.5 – 97.5 h** | **NO** |
| fb-comments | 52 | 1 264 000 | 4.5 h | 69.7 h | 2.5 h | **76.7 – 188.0 h** | **NO** |

**Sampling dominates**, because the protocol generates each dataset's own
`num_samples` (about 8x its training rows) through 1 000 diffusion steps, once
per generation seed. `ablation_runner` overrides every dataset's own
`num_timesteps` with `ABLATION_NUM_TIMESTEPS = 1000`, so T is 1000 even for
king, whose tuned config says 100.

**There is one combination per job and no intermediate checkpoints**
(`ablation_runner` passes `checkpoint_every=0`), so unlike `hpc/`, a job killed
by the walltime loses the entire run and restarts from step 0. Resubmitting
does not help.

Options for miniboone and fb-comments, cheapest first — none is applied by
default, because the protocol was chosen to be comparable to the ablation and
changing it silently would defeat the point:

1. `--n_gen_seeds 2` instead of 5 puts miniboone at ~17–40 h and fb-comments at
   ~31–76 h. Point estimates stay on protocol; error bars widen and are no
   longer 5-seed comparable.
2. Cap the synthetic rows, as `hpc_wide/` does
   (`BUDGET_MAX_NUM_SAMPLES=50000`, applied in memory by `hpc_wide/run_combo.py`).
   Bounded and cheap, but the metrics are then computed on a capped synthetic
   set and are **not** comparable to `tabddpm_ablation/`.
3. A longer walltime on a partition that allows it, if one exists.

**Calibrate on king.** It is the cheapest target and structurally the closest
to a donor. If it takes much more than ~3 h, every estimate scales up and
higgs-small stops being marginal.

**Group A is cheap** — 1.0 to 3.5 h optimistic, gesture the worst at
3.5 – 7.1 h. Submit those first; `diabetes` or `wilt` (~1 h) is the fastest
calibration you can get. Where the backbone is GCN or GIN (adult, buddy,
california, diabetes) the figure is an **upper bound**, not an estimate: those
layers never build the pairwise tensor the model is derived from. Full table:
`python3 hpc_transfer/resource_estimates.py`.

**The rule barely changes cost** — the adjacency rules are a small part of the
traffic (`hpc_wide/README.md` puts them under 3 %), so one estimate covers all
rules of a dataset. But that means **2 or 3 jobs per dataset, not one**:
higgs-small is marginal in every rule, and miniboone and fb-comments miss in
every rule, so that is 3 and 6 jobs you cannot submit as-is. On memory,
entmax-1.5 is the heavier rule — the measured peaks quoted above are already
entmax-1.5 figures, so they cover the worst case.

Memory is not the binding constraint: measured peak is 5.6–15 GiB
(`hpc_wide/README.md`), well inside the `--mem=120G` the partition allows.

## 7. Collect

```bash
python3 hpc_transfer/collect_results.py \
    --repo_root "$REPO_ROOT" --transfer_root "$HPC_ROOT/transfer_runs"
```

Writes into `hpc_transfer/`:

Every config in `configs/` is collected, in the modes its `RUN_MODES` line
declares, with `group` (`transfer` / `ablated_best`) and `topk_mode` columns
separating them.

- **`transfer_ablation.json`** — one entry per run (39: 7 new x 3 rules + 9 ablated x 2), in exactly the schema of
  `tabddpm_ablation/<ds>_ablation.json`, so it can be read by
  `ablation/existing_results.py` without special-casing.
- **`results.csv`** — the same rows flattened by
  `ablation_runner.summary_to_csv()`, the function the ablation's own CSV uses,
  so the column layout is identical by construction.
- **`combined.csv`** — the ablation's 88 runs and these 39 in one table, with
  `study`, `group` and `topk_mode` columns separating them.
- **`run_meta.csv`** — provenance per run: strategy, neighbours, distance,
  confidence, GPU, git commit.

It reports missing and failed runs rather than skipping them silently, and
exits non-zero if either list is non-empty.

## 8. Reproduce the analysis

```bash
./hpc_transfer/run_analysis.sh
```

Runs Phases 1–3 in order and regenerates every CSV and `predictions.md`. Use
this rather than the individual scripts: `phase3_report.py` reads the CSVs
`phase3_transfer.py` writes, so a sensitivity sweep left running last would
otherwise leak its settings into the report.

## 9. Files

| File | Purpose |
|---|---|
| `predictions.md` | The derivation: Phase 1 winners, the `d_model` ceiling, LOO validation, per-target predictions |
| `phase1_best_config.py` | Best config per ablated dataset, tie sets, axis marginals |
| `meta_features.py` | Dataset meta-features from the **train split only** |
| `phase3_transfer.py` | Strategies A/B/C, leave-one-out, target predictions |
| `phase3_report.py` | Renders `predictions.md` from the CSVs |
| `resource_estimates.py` | Walltime and memory model; writes `resource_estimates.csv` |
| `preflight.py` | Per-target checks incl. a real forward/backward pass |
| `configs/<ds>.sh` | One config per target, with a provenance header; shared by both of that target's jobs |
| `jobs/<ds>_transfer.sh` | SLURM job, `topk_mode fixed` |
| `jobs/<ds>_transfer_jenks.sh` | SLURM job, `topk_mode adaptive_jenks` |
| `jobs/{adult,california}_jenks.sh` | SLURM job, adaptive-Jenks backfill of an ablated dataset |
| `best_manifest.py` | Each ablated dataset's own ablation winner, via `existing_results.load_best()` |
| `common.sh` | Shared runner, `run_transfer`: one dataset, one configuration, one rule; plus `entmax_preflight` |
| `smoke_test.sh` | GPU check, preflight, then one short capped run, in either mode |
| `submit_all.sh` | Commented submission lines — submits nothing |
| `collect_results.py` | Aggregate into the ablation's table format |
| `run_analysis.sh` | Regenerate Phases 1–3 in order |
| `logs/` | `<dataset>_failures.txt`, written at runtime |

## 10. Caveats

- The LOO result rests on **four** donor datasets (abalone, california, churn2,
  insurance). The other five ablated datasets either cannot separate any config
  from any other (diabetes snr 1.40, wilt 1.09) or have a grid too small for an
  argmax to mean anything (adult 1 config, buddy 2, gesture 2). Four is enough
  to show nearest-neighbour transfer failing badly; it is not enough to rank
  strategies finely.
- Every `std` in the ablation spans 5 generation x 10 classifier seeds of a
  **single trained model** — `ablation_runner` trains once per cell and then
  re-samples. Training-seed variance is not measured anywhere in this study,
  and these jobs do not add it (they run `--seed 0`, as the ablation did).
- All seven targets are extrapolations on size: every one has more rows than
  the largest donor, and five have a wider feature graph. fb-comments is 11.9x
  california's rows and 4.7x its node count.
- Group A covers adult and california, which `hpc/adaptive_manifest.py`
  excludes by design (`EXCLUDED_DATASETS`, commented only as "the baseline for
  these two is not being revisited"). Nothing in `hpc/` is modified, but if
  that exclusion existed for a reason beyond scope, these two jobs reinstate
  what it was avoiding.
- **On 4 of the 9 ablated datasets the "best config" is close to an arbitrary
  pick**: diabetes (snr 1.40), wilt (1.09), buddy (1.02) and adult (0.00, a
  single config) cannot separate any configuration from any other. Their
  config files say so in a warning block. The adaptive-vs-fixed comparison is
  still valid there — it is a paired run of the *same* configuration — but the
  choice of configuration is not evidence of anything.
- `beijing`, `magic`, `news` and `shoppers` have `data/` splits but no
  `exp/<ds>/` tuned config and no `ablation/<ds>_ablation.py`, so they are not
  covered here. Running them would require choosing lr, batch size, timesteps
  and `num_samples` from scratch.
