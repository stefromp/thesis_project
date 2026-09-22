# Sparsemax / entmax-1.5 adaptive top-k on ARIS

Second adaptive top-k rule, run on exactly the combinations the Jenks runs
used (`hpc/adaptive_manifest.py`), so **fixed top-k vs Jenks vs entmax** can be
compared head to head. Seven datasets have job scripts: wilt, abalone and
gesture (where Jenks results exist), plus buddy, churn2, diabetes and
insurance (baseline vs sparsemax vs entmax15; see step 6).

This directory is a self-contained sibling of `hpc/`. **Nothing here modifies,
overwrites or deletes anything under `hpc/`**: the Jenks job scripts on the
cluster stay exactly as they are. `hpc_entmax/` only *reads*
`hpc/adaptive_manifest.py` (for the combination list).

| | `hpc/` | `hpc_entmax/` |
|---|---|---|
| modes | `adaptive_jenks` | `adaptive_sparsemax` (default), `adaptive_entmax15` |
| job scripts | `jobs/<ds>_ablation.sh` | `jobs/<ds>_entmax.sh` (wilt, abalone, gesture, buddy, churn2, diabetes, insurance) |
| results | `exp/<ds>/ablation_adaptive_jenks/` | `exp/<ds>/ablation_adaptive_sparsemax/`, `exp/<ds>/ablation_adaptive_entmax15/` |
| failure logs | `hpc/logs/<ds>_failures.txt` | `hpc_entmax/logs/<ds>_<mode>_failures.txt` |
| job logs | `jobs_out/<ds>_ablation.{out,err}` | `jobs_out/<ds>_<mode>_<jobid>.{out,err}` |

`adaptive_jenks` and `fixed` are refused by every script here — those runs
belong to `hpc/`.

---

## 1. The method

`--topk_mode adaptive_sparsemax` (α = 2) or `adaptive_entmax15` (α = 1.5), both
fixed — no new trainable parameters. **Mask-only**: α-entmax decides *which*
edges exist; it never supplies the weights.

```
phi_off = phi with the diagonal set to -inf            # [B, F, F]
p       = entmax(phi_off, alpha, dim=-1)               # under no_grad, on phi.detach()
mask    = (p > 0) | top-k_min(phi_off)                 # the support = the kept edges
A       = sigmoid(phi) on mask, 0 elsewhere, diag = 1  # same gates as fixed / Jenks
k_i     = mask.sum(-1)                                 # the budget, logged
```

Warm-up (`TOPK_WARMUP_STEPS`, default 0), the dense fallback and
`topk_stats.csv` logging are shared with Jenks unchanged. Implementation:
`adaptive_adjacency.build_adjacency_entmax`; dispatch in
`tab_ddpm/graph_builder.py`.

`tests/test_topk_modes_bitexact.py` proves `fixed` and `adaptive_jenks` are
bit-identical (`torch.equal`, no tolerance) to the code that produced the
baseline and the finished Jenks runs (git tag `jenks-ref`), so the Jenks
results remain reproducible by the current code.

One property worth knowing when reading the budgets: the entmax support
depends on the *spread* of φ, whereas the Jenks split is invariant to a
positive rescaling of φ (apart from its μ_s > 0 gate).

## 2. Environment: install entmax

```bash
module load python/3.10
/work/pr019/atvoulo/Rompos/venvs/tabddpm310/bin/python -m pip install entmax==1.3
```

`entmax` 1.3 is a pure-Python wheel whose only dependency is `torch>=1.3`,
already satisfied by torch 2.1.2+cu118 — pip leaves torch untouched, so
`--no-deps` is not needed. (If the login node has no internet:
`pip download entmax==1.3 --no-deps` elsewhere, copy the `.whl`, and
`pip install` the file.)

### scikit-learn 1.1.3 vs the Kaggle baseline

The cluster venv uses **scikit-learn 1.1.3**; the Kaggle baseline
(`tabddpm_ablation/`) was produced with **scikit-learn ≥ 1.2**.
`scripts/eval_tabddpm_protocol.py::_encode` used
`OneHotEncoder(sparse_output=False)`, which only exists from 1.2, so on the
cluster every metric that one-hot encodes a categorical column raised on every
generation seed and came out NaN: `dcr_baseline` on wilt and gesture (the
class label is folded in as a category), and `dcr_tabddpm_mean`,
`dcr_baseline`, `mia_auc` on abalone (`Sex` is categorical).

`_encode` now falls back to `OneHotEncoder(sparse=False)` on sklearn < 1.2.
Both spellings produce the same dense one-hot matrix, so **the privacy metrics
are identical across the two environments** (verified on wilt, abalone and
gesture: sklearn 1.1.3 and 1.7.2 give the same values for all four protocol
metrics). Every `run_meta.json` written from here records `sklearn_version`
and this note under `privacy_metrics_note`.

The finished Jenks runs are repaired without retraining by
`backfill_privacy.py` (step 2 below).

## 3. Submission order

From the repo root on the cluster:

```bash
cd /work/pr019/atvoulo/Rompos/thesis_project
module load python/3.10
PY=/work/pr019/atvoulo/Rompos/venvs/tabddpm310/bin/python
```

**Step 1 — install entmax** (section 2), then:

```bash
$PY -c "import entmax, torch, sklearn; print(entmax.__version__, torch.__version__, sklearn.__version__)"
```
Check: `1.3 2.1.2+cu118 1.1.3`, and pip reported
`Requirement already satisfied: torch>=1.3` — torch was not reinstalled.

**Step 2 — repair the finished Jenks runs**

```bash
$PY hpc_entmax/backfill_privacy.py --repo_root "$PWD" --datasets wilt abalone gesture --check; echo "exit=$?"
$PY hpc_entmax/backfill_privacy.py --repo_root "$PWD" --datasets wilt abalone gesture
$PY hpc_entmax/backfill_privacy.py --repo_root "$PWD" --datasets wilt abalone gesture --check; echo "exit=$?"
ls exp/*/ablation_adaptive_jenks/*/results_full_averaged.orig.json | wc -l
```
Check: the first `--check` lists 13 finished runs with NaN/missing metrics and
exits 1. The backfill prints `runs rewritten : 13` and `not finished : 3`
(the wilt dim32/64/128 combos). The second `--check` ends with
`all finished runs have finite ...` and exits 0. There are 13 `.orig.json`
backups. CPU only, a few minutes; safe to re-run (a second run rewrites 0).

**Step 3 — finish the Jenks wilt runs, with the existing `hpc/` script**

```bash
cp /work/pr019/atvoulo/Rompos/jobs_out/wilt_ablation.out /work/pr019/atvoulo/Rompos/jobs_out/wilt_ablation.first_run.out
cp /work/pr019/atvoulo/Rompos/jobs_out/wilt_ablation.err /work/pr019/atvoulo/Rompos/jobs_out/wilt_ablation.first_run.err
sbatch hpc/jobs/wilt_ablation.sh
```
The `hpc/` script writes to the fixed name `wilt_ablation.out`, so the copy
keeps the first run's log. Check:
`grep -E "^\[[0-9]/|SKIP|\[OK\]|\[FAIL\]" /work/pr019/atvoulo/Rompos/jobs_out/wilt_ablation.out`
shows `[1/4]` → `[SKIP]` (dim512), then `[2/4]`… running. With the `_encode`
fix uploaded, these three get finite `dcr_baseline` directly.

**Step 4 — sparsemax on wilt alone (smoke test on a real run)**

```bash
./hpc_entmax/submit_all.sh adaptive_sparsemax wilt
```
Check, in order:
1. Within a minute of the job starting, the head of
   `jobs_out/wilt_adaptive_sparsemax_<jobid>.out` shows
   `topk_mode : adaptive_sparsemax`, `entmax : 1.3`,
   `torch / sklearn : 2.1.2+cu118 / 1.1.3`, `preflight : OK`,
   `4 run(s) for wilt`. A `PREFLIGHT FAILED` block names the file to upload.
2. During training, `tail exp/wilt/ablation_adaptive_sparsemax/gnn_gatv2_layers2_dim512_topk3_heads8_noattn/topk_stats.csv`
   has `train` rows with `k_mean` between 1 and 4 (wilt has N = 5 nodes).
3. When the first combination finishes:
   `$PY hpc_entmax/backfill_privacy.py --repo_root "$PWD" --datasets wilt --topk_mode adaptive_sparsemax --check`
   shows that combination with all four metrics `ok` (the other three
   `not finished`), and `hpc_entmax/logs/wilt_adaptive_sparsemax_failures.txt`
   does not exist.

**Step 5 — the rest of wilt / abalone / gesture, only after step 4 passes**

```bash
./hpc_entmax/submit_all.sh adaptive_sparsemax abalone gesture
./hpc_entmax/submit_all.sh adaptive_entmax15 wilt abalone gesture
squeue --me -o "%.10i %.14j %.8T %.10M %R"
```
Check: `squeue` shows `spm_abalone`, `spm_gesture`, `e15_wilt`,
`e15_abalone`, `e15_gesture` (plus `spm_wilt` if still running); each `.out`
header passes the same checks as step 4.

**Step 6 — buddy, churn2, diabetes, insurance (after step 4 passes)**

```bash
./hpc_entmax/submit_all.sh adaptive_sparsemax buddy churn2 diabetes insurance
./hpc_entmax/submit_all.sh adaptive_entmax15 buddy churn2 diabetes insurance
```
Check: `squeue` shows `spm_buddy`, `spm_churn2`, `spm_diabetes`,
`spm_insurance` and the `e15_*` counterparts; each `.out` header shows
`preflight : OK` and the run count (buddy 2, churn2 10, diabetes 12,
insurance 8). buddy, churn2 and insurance have categorical columns and
diabetes is a classifier, so all four rely on the `_encode` fix — the
preflight refuses to start if it is missing. No Jenks results exist for these
four, so they compare baseline vs sparsemax vs entmax15 (Jenks shows as
`0/N`). If Jenks runs are wanted too, the existing, unmodified
`sbatch hpc/jobs/<ds>_ablation.sh` produces them.

With 14 jobs (7 datasets x 2 modes) on the two-node `gpu` partition, expect
queueing; the order of steps 5 and 6 does not matter once step 4 has passed.

**When everything has finished**

```bash
$PY hpc_entmax/backfill_privacy.py --repo_root "$PWD" --datasets wilt abalone gesture --check
for m in adaptive_sparsemax adaptive_entmax15; do
    $PY hpc_entmax/backfill_privacy.py --repo_root "$PWD" --topk_mode $m --check \
        --datasets wilt abalone gesture buddy churn2 diabetes insurance
done
$PY hpc_entmax/compare_methods.py --repo_root "$PWD"      # all seven datasets by default
```

Run counts per mode: wilt 4, abalone 10, gesture 2, buddy 2, churn2 10,
diabetes 12, insurance 8 = **48** (16 on the three Jenks datasets; the Jenks
tree reaches 16 after step 3).

## 4. Job names and logs

`submit_all.sh <mode> <dataset>...` requires both arguments — nothing is
submitted by default — and passes on the `sbatch` command line (overriding the
`#SBATCH` directives):

- `--job-name=<tag>_<dataset>` with tag `spm` / `e15`. The tag comes first
  because `squeue`'s default NAME column is 8 characters wide
  (`spm_wilt`, `e15_abal`, `spm_gest`).
- `--output` / `--error` = `jobs_out/<dataset>_<mode>_%j.{out,err}`.
- `--export=ALL,TOPK_MODE=<mode>`.

The scripts' own `#SBATCH --output/--error` default to `%x_%j`, so even a
direct `sbatch` can never overwrite another job's log. The header is otherwise
identical to `hpc/jobs/*.sh`, including `--job-name=<ds>_ablation`: submitted
directly (`TOPK_MODE=adaptive_entmax15 sbatch hpc_entmax/jobs/wilt_entmax.sh`)
a job shows up under the same name as a Jenks job, so prefer `submit_all.sh`.

## 5. Preflight

Before the first combination, each job checks that the *uploaded* code can
run the mode, and exits without running anything otherwise:

- `entmax` imports;
- `ablation/ablation_runner.py` accepts `--topk_mode <mode>`;
- `DynamicAdjacency(..., topk_mode=<mode>)` runs a forward pass
  (`tab_ddpm/graph_builder.py` + `adaptive_adjacency.py`);
- `_encode` works on the installed sklearn with a categorical column
  (`scripts/eval_tabddpm_protocol.py`) — the NaN bug cannot come back.

## 6. Resume and failures

Resubmit the same `submit_all.sh` command: a combination with
`results_full_averaged.json` is skipped. Resume granularity is one whole
combination (training writes no intermediate checkpoints). A failing
combination is appended to `hpc_entmax/logs/<ds>_<mode>_failures.txt` and the
loop moves on. `srun --input none` keeps `srun` from consuming the loop's
input — without it only the first combination ran.

## 7. Comparison outputs (`compare_methods.py`)

Written to `hpc_entmax/` (the script refuses to write under `hpc/`):

- **`method_comparison.csv`** — one row per *baseline* combination (abalone:
  19), with `score`, `wasserstein_mean`, `mia_auc`, `dcr_tabddpm_mean` as
  mean/std for baseline, Jenks, sparsemax, entmax15, and the mean sampling-phase
  budget `k_sample_<method>`. The adaptive cells repeat across the `top_k`
  values they replace.
- **`method_comparison_wtl.csv`** — win/tie/loss per dataset × metric × pair.
  Tie: `|Δmean| < max(std_A, std_B)`. Higher is better for `score` (−RMSE for
  regression) and `dcr_tabddpm_mean`; lower for `wasserstein_mean` and
  `mia_auc`. Adaptive vs baseline is counted over the baseline rows; adaptive
  vs adaptive over the unique combinations. Rows with a missing side are
  counted as `n_na`, never as a result.
- **`budget_curves.csv`** — sampling-phase `k_mean` per t-bucket (0 = t near 0,
  9 = t near T), per dataset × method, averaged over combinations, with `k_max`
  and `k_frac` for plotting.

All comparisons are unpaired and descriptive (the baseline exists only as
mean ± std): no significance is claimed.

## 8. Files

| File | Purpose |
|---|---|
| `jobs/<ds>_entmax.sh` | One SLURM job per dataset (wilt, abalone, gesture, buddy, churn2, diabetes, insurance) |
| `common.sh` | Mode check, preflight, per-combination runner |
| `submit_all.sh` | `<mode> <dataset>...` → one job each, per-mode names/logs |
| `backfill_privacy.py` | Repair NaN/missing privacy metrics from saved `gen_seed_*/` |
| `compare_methods.py` | Baseline vs Jenks vs sparsemax vs entmax15 tables + budget curves |
| `logs/` | `<ds>_<mode>_failures.txt`, written at runtime |
