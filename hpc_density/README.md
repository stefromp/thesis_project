# Density-curve study (Stage 1)

The three adaptive rules land on very different mean densities (Jenks ~1–2.4
neighbours, sparsemax ~2–4, entmax-1.5 ~8–17), so comparing them directly mixes
*how many* edges a method keeps with *which* edges and *when*. This study
separates the two. For each dataset, with **one fixed GATv2 architecture**:

- fixed top-k at a grid of densities → a curve "metric vs k"
- the three adaptive rules → three points at their achieved mean k

An adaptive point **below** the fixed-k curve on Wasserstein (lower is better)
means adaptivity adds something beyond density; **on** the curve means the rule
is just choosing a density.

Self-contained: nothing here modifies `hpc/`, `hpc_entmax/` or `hpc_wide/`.
Results go to `density_runs/`, never to `exp/` or `wide_runs/`. Every run is
new — no earlier run is reused, so every comparison is paired.

## Design

| dataset | N | architecture (GATv2) | fixed k (dense = N−1) | runs |
|---|---|---|---|---|
| wilt | 5 | L2 d64 h8 | 1, 2, 3, dense (4) | 7 |
| abalone | 9 | L3 d64 h4 | 1, 2, 4, dense (8) | 7 |
| churn2 | 11 | L2 d128 h4 | 1, 3, 5, dense (10) | 7 |
| default | 23 | L3 d128 h8 | 1, 2, 6, 11, dense (22) | 8 |
| gesture | 32 | L2 d32 h4 | 1, 3, 8, 16, dense (31) | 8 |
| | | | | **37** |

Each dataset also runs `adaptive_jenks`, `adaptive_sparsemax` and
`adaptive_entmax15` (included in the run counts).

- **Architecture**: the best GATv2 run by `score.mean` in
  `tabddpm_ablation/` — a rule fixed before the study, deliberately not
  Wasserstein (the metric that judges it). Documented in `density_manifest.py`:
  wilt's four widths tie on score and d512 was better on Wasserstein (kept
  d64); abalone's L3 d64 ties with L2 d64 (kept L3); gesture has a single
  GATv2 run; default has no baseline and uses the `hpc_wide` configuration.
- **k grid**: k/(N−1) ≈ 10 %, 25 %, 50 %, rounded half up (churn2's 2.5 → 3),
  minimum 1, plus dense (`--top_k 0`, plotted at N−1); if fewer than 4 values
  result (wilt), every k in 1..N−2 plus dense. **k = 1 is added for default
  and gesture**, whose Jenks points (~1.1–2.7) would otherwise fall outside
  the curve.
- **Seeds**: training seed 0, generation seeds 0–4, classifier seeds 0–9 for
  every condition (the module defaults; nothing overrides them) — the pairing.
- **N** is computed as `scripts/train.py` builds the graph, from the `.npy`
  headers (`density_manifest.py --summary`).

## Per-seed persistence

`ablation/ablation_runner.py` gained one additive change (34 lines, nothing
removed): every run also writes `per_seed_scores.csv` — one row per
(generation seed, classifier seed) for the CatBoost metrics and one per
generation seed for DCR, Wasserstein and MIA, as `gen_seed, clf_seed, metric,
value`. `results_full_averaged.json` is unchanged: a before/after run was
byte-identical, and the per-seed rows reproduce its mean and std exactly. A
partial file from an interrupted attempt is reset at the start of a run.
Other directories (`hpc/`, `hpc_entmax/`, `hpc_wide/`) also get the file from
now on; nothing else changes for them.

## Cost

Calibrated on the finished ARIS jobs (wilt 67 min per combination, gesture
2.6–3.2 h): the GPU speed factor fitted on wilt predicts both gesture
combinations inside their measured range. 5 × 10 seeds per condition.

| dataset | per condition | runs | GPU-h |
|---|---|---|---|
| wilt | 0.20–0.26 h | 7 | 1.4–1.8 |
| abalone | 0.55–0.74 h | 7 | 3.8–5.2 |
| churn2 | 0.71–0.96 h | 7 | 5.0–6.7 |
| gesture | 2.2–3.2 h (≈40 % multiclass CatBoost eval) | 8 | 18–25 |
| default | 4.8–6.5 h | 8 | 38–52 |
| **total** | | **37** | **≈ 66–91** |

One job per (dataset, condition): conditions run in parallel and a timeout
loses one condition. The longest job (default) is ~6.5 h.

## Submit

```bash
cd /work/pr019/atvoulo/Rompos/thesis_project
module load python/3.10
PY=/work/pr019/atvoulo/Rompos/venvs/tabddpm310/bin/python
$PY hpc_density/density_manifest.py --repo_root "$PWD" --summary   # 37 runs, every line OK
DRY_RUN=1 ./hpc_density/submit_all.sh all all                      # prints 37 sbatch commands

./hpc_density/submit_all.sh fixed_k2 wilt                          # smoke test: one condition
./hpc_density/submit_all.sh all wilt abalone churn2                # then the cheap datasets
./hpc_density/submit_all.sh all gesture default                    # then the expensive ones
squeue --me -o "%.10i %.12j %.8T %.10M %R"
```

- First argument: conditions (comma-separated), or `all` | `fixed` |
  `adaptive`, expanded per dataset (`fixed_k16` exists for gesture only).
  Datasets, or `all` (cheapest first).
- Job names `<tag>_<dataset>`: `k<K>` / `kD` for fixed, `Dj` / `Ds` / `De`
  for adaptive — e.g. `k2_wilt`, `k16_gest`, `kD_defau`, `De_churn` in
  squeue's 8-character column; no prefix is shared with `hpc/`,
  `hpc_entmax/` or `hpc_wide/` jobs.
- Logs: `jobs_out/density_<dataset>_<condition>_<jobid>.{out,err}`.
- `MEM=40G ./hpc_density/submit_all.sh ...` lowers `--mem` (default 120G)
  once `sacct -j <id> --format=JobID,MaxRSS,Elapsed` shows what the jobs use.
- Resume: resubmit — a condition with `results_full_averaged.json` is
  skipped. On a restart the previous attempt's `topk_stats.csv` is moved to
  `topk_stats.stale_<time>.csv` (it is appended to, and would mix attempts).
- Each job `cd`s to `$REPO_ROOT` (the evaluators read `tuned_models/`
  relative to the working directory) and runs a preflight before anything
  else: entmax (entmax modes only), config vs data width, the `_encode`
  fix, the per-seed runner, and a forward pass of the condition's adjacency
  rule on the dataset's N nodes.

Check a finished run:
`density_runs/exp/<ds>/ablation[_<mode>]/<combo>/` holds
`results_full_averaged.json`, `per_seed_scores.csv` (one row per classifier
metric for each of the 50 generation × classifier seeds — 550 rows for
classification, fewer for regression — plus 5 × 4 per-generation rows for
DCR ×2, Wasserstein and MIA) and, for adaptive conditions,
`topk_stats.csv`; `run_meta.json` records condition, k, k/(N−1),
architecture, seeds and versions.

## Analysis

```bash
python3 hpc_density/density_curve.py --repo_root "$PWD"
```

- **Mean k**: the fixed k (dense = N−1), or for adaptive rules the
  sampling-phase `k_mean` from `topk_stats.csv`, `n_samples`-weighted (its
  std over generation seeds gives the x error bar).
- **Metrics**: mean ± std over generation seeds (score: the mean over the 10
  classifier seeds of each generation seed).
- **Verdict** per adaptive point: the fixed-k curve interpolated linearly in
  log k between the two bracketing fixed points (no extrapolation — outside
  the grid is reported as "outside the curve"); below / on / above, where
  "on" means the difference is smaller than the larger of the two stds; plus a
  paired comparison against the nearest fixed-k condition (mean paired
  difference and a sign count over generation seeds). No p-values.
- **Directions**: lower is better for `wasserstein_mean` (primary) and
  `mia_auc`; higher for `score` (secondary) and `dcr_tabddpm_mean` (context
  only, reported next to Wasserstein, never a privacy claim on its own).

Written to `hpc_density/results/`: `density_points.csv`,
`density_verdicts.csv`, `figures/density_<ds>.{png,pdf}` (Wasserstein and score
vs mean k) and `figures/density_all.{png,pdf}` (Wasserstein vs k/(N−1), one
panel per dataset). It refuses to write under `hpc/`, `hpc_entmax/` or
`hpc_wide/`.

## Files

| File | Purpose |
|---|---|
| `density_manifest.py` | Datasets, architectures (+ evidence), k grids, N, conditions |
| `jobs/<ds>_density.sh` | One SLURM job per (dataset, `CONDITION`) |
| `common.sh` | Condition check, preflight, runner, run_meta.json |
| `submit_all.sh` | `<conditions\|all\|fixed\|adaptive> <datasets\|all>` → one job each |
| `density_curve.py` | Points, verdicts and figures |
| `logs/` | failure logs, written at runtime |
