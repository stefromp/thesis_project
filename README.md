# Graph-Aware Denoiser for Tabular Diffusion

This thesis project adds a graph neural network (GNN) denoiser to
[TabDDPM](https://github.com/yandex-research/tab-ddpm) (Kotelnikov et al., 2023).
The original TabDDPM denoiser is an MLP that sees each table row as one flat
vector. Here, each feature (column) becomes a node in a graph of feature
dependencies, and the denoiser passes messages along that graph. The model
still generates ordinary table rows. The graph only shapes how the network
combines features.

The repository contains the model, the training and sampling pipeline, a GNN
ablation study over 20 datasets, and two evaluation protocols: one matching
TabDDPM and one matching [TabDiff](https://github.com/MinkaiXu/TabDiff).

## How the denoiser works

- **Nodes:** there is one node per numerical feature and one per categorical
  feature. Each categorical node gets its input from that feature's one-hot
  slice. The timestep and class-label embeddings are added to every node.
- **Edges:** two modes (see `tab_ddpm/graph_builder.py`).
  - `static`: edges are computed once from the training data, using |Pearson|
    for pairs of numerical features, Cramér's V for pairs of categorical
    features and normalised mutual information for mixed pairs. An edge is
    kept when its score is above a threshold.
  - `dynamic`: a soft adjacency matrix is learned from the node embeddings and
    recomputed in every block. It can optionally keep only the top-k
    neighbours of each node.
- **Backbones** (`tab_ddpm/gnn_layers.py`): GCN, GAT, GATv2, GIN, and the
  original graph-masked multi-head attention (`graphmha`).
- **Blocks:** the network stacks `n_layers` blocks. Each block has an optional
  dense self-attention sublayer followed by two GNN layers.
- **Output:** each node is projected back to its feature's space, and the
  outputs are concatenated. The Gaussian and multinomial diffusion process is
  the same as in TabDDPM.

When the graph is disabled, the pipeline uses the original `MLPDiffusion` and
gives the same results as upstream TabDDPM. For more detail, see
[docs/graph_denoiser.md](docs/graph_denoiser.md).

## Repository layout

| Path | Contents |
|---|---|
| `tab_ddpm/` | Diffusion model and denoisers. The graph code is in `graph_builder.py`, `gnn_layers.py` and `graph_denoiser.py`. |
| `lib/` | Data loading, preprocessing and metrics (from TabDDPM). |
| `scripts/` | Training, sampling and evaluation pipeline, hyperparameter tuning, dataset preparation and plotting. |
| `ablation/` | GNN ablation study: a shared runner, one script per dataset and the TabDiff-protocol evaluators. |
| `data/<ds>/` | Datasets stored as `.npy` train/val/test splits plus `info.json`, tracked with Git LFS. |
| `data/<ds>_tabdiff_split/` | Re-splits of adult, default, shoppers, magic, beijing and news that match TabDiff's split sizes. |
| `exp/<ds>/` | Configs and results for tuned TabDDPM (`ddpm_cb_best`, `ddpm_mlp_best`) and for the CTAB-GAN, CTAB-GAN+, SMOTE and TVAE baselines. |
| `tuned_models/` | Tuned hyperparameters for the CatBoost and MLP evaluation models. |
| `tabddpm_ablation/` | Finished ablation results, one JSON file per dataset. |
| `results_plots/` | Synthetic samples used to draw the figures. |
| `figures/` | Rendered figures: marginal distributions and correlation-difference heat maps. |
| `checkpoints/`, `model_ema/` | Saved model weights. |
| `tests/` | Correctness tests for the GNN layers. |

## Setup

```bash
git clone https://github.com/stefromp/thesis_project
cd thesis_project
git lfs pull                      # the datasets in data/ are stored with Git LFS
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Optional extras:

- `pip install sdmetrics==0.13.0 xgboost==2.0.3` is needed for the
  TabDiff-protocol evaluation.
- `torch_geometric` is needed for the PyG equivalence tests.

If `rtdl` or `libzero` fail to install on a recent Python version, the
ablation runner replaces them with small built-in stubs.

## Usage

### Train, sample and evaluate one model

```bash
python scripts/pipeline.py --config exp/adult/ddpm_cb_best/config.toml --train --sample --eval
```

To use the graph denoiser, add a `[model.graph]` section to the config:

```toml
[model.graph]
enabled        = true
mode           = "dynamic"   # "static" | "dynamic"
gnn_type       = "gatv2"     # graphmha | gcn | gat | gatv2 | gin
n_layers       = 2
d_model        = 64
n_heads        = 4
sparsity_top_k = 5           # dynamic mode only; 0 = dense
use_attention  = false       # dense self-attention sublayer in each block
threshold      = 0.1         # static mode only
```

### Run the ablation study

By default, the study uses dynamic adjacency and sweeps `gnn_type` ∈
{gcn, gatv2, gin}, `n_layers` ∈ {2, 3}, `d_model` ∈ {32, 64}, `top_k` ∈ {3, 5}
and `n_heads` = 4, with self-attention off. Each combination is trained for
20,000 steps. The `--grid_*` flags change the grid.

```bash
# Quick end-to-end check (200 steps, 1 seed; the numbers are not meaningful)
python ablation/run_all_ablation.py --datasets abalone --smoke --device cuda:0

# One combination on one dataset
python ablation/adult_ablation.py --gnn_type gatv2 --n_layers 2 --d_model 64 \
    --top_k 5 --n_heads 4 --use_attention 0 --device cuda:0

# Full grid on several datasets, then collect the results
python ablation/run_all_ablation.py --datasets adult abalone --device cuda:0
python ablation/run_all_ablation.py --datasets adult abalone --aggregate --csv
```

Results for each run go to `exp/<ds>/ablation/<run>/results_full_averaged.json`.
The summary for a dataset goes to `exp/<ds>/ablation/ablation_summary.{json,csv}`.

## Evaluation

The two protocols use different data splits and give different numbers, so
they are reported in separate tables and never combined.

| | TabDDPM protocol | TabDiff protocol |
|---|---|---|
| Data | `data/<ds>/` | `data/<ds>_tabdiff_split/` (flag: `--tabdiff_split`) |
| ML efficiency | CatBoost F1 / R², 5 sampling seeds × 10 evaluation-model seeds | XGBoost (`--evaluator xgboost`), a port of TabDiff's `eval/mle` |
| Fidelity and privacy | DCR, Wasserstein distance, membership-inference AUC | Shape, Trend, C2ST, TabDiff DCR, α-precision / β-recall (`--tabdiff_fidelity`) |
| Code | `scripts/eval_tabddpm_protocol.py` | `ablation/eval_xgboost.py`, `ablation/eval_tabdiff_fidelity.py` |

The two DCR metrics above measure different things and can't be compared with
each other. There's also a caveat within the TabDDPM protocol. Its DCR is the
mean nearest-neighbour distance with the label left out, while the baselines'
`exp/<ds>/ddpm_cb_best/privacy.json` stores the median distance with the label
included. To compare the ablation against those baselines, use the
`dcr_baseline` key in the ablation results, which follows the baseline recipe.

### Other scripts

- `scripts/eval_fidelity.py`: column-wise, pair-wise and joint fidelity scores.
- `scripts/resample_privacy.py`: DCR for the baselines.
- `scripts/tune_ddpm.py`, `scripts/tune_evaluation_model.py`: Optuna tuning of
  the diffusion model and of the evaluation models.
- `scripts/eval_seeds.py`: evaluation over several seeds.
- `scripts/process_new_datasets.py`: converts raw CSV files to the `data/`
  layout.
- `scripts/make_tabdiff_splits.py`, `scripts/apply_tabdiff_schema.py`: build
  the TabDiff-comparable splits.
- `scripts/plot_distributions.py`: marginal-distribution and
  correlation-difference figures, written from `results_plots/` to `figures/`.
- `scripts/fidelity_stats.py`: the KS, TVD and correlation numbers behind those
  figures.
- `ablation/count_params.py`: parameter counts for the MLP and GNN denoisers.
- `ablation/profile_gatv2_memory.py`: GPU memory profile of the GATv2 layer.

## Tests

```bash
python scripts/sanity_check_graph.py               # forward/backward smoke test (static + dynamic)
pytest tests/test_gnn_correctness.py -v            # each layer compared with a hand-written reference
pytest tests/test_gnn_pyg_equivalence.py -v        # message passing compared with torch_geometric
```

## Acknowledgements

The diffusion model, data pipeline and baseline experiments build on
[TabDDPM](https://github.com/yandex-research/tab-ddpm). The TabDiff evaluation
is ported from [TabDiff](https://github.com/MinkaiXu/TabDiff), and the privacy
metric is adapted from [CTAB-GAN](https://github.com/Team-TUD/CTAB-GAN).
