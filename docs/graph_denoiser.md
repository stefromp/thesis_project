# Graph-Aware Denoiser for TabDDPM

## Motivation

The default TabDDPM denoiser (`MLPDiffusion`) treats all features as a flat
vector with no explicit representation of inter-feature dependencies.  The
graph-aware extension adds a **feature-dependency graph as an inductive bias
inside the denoising network**.  The graph is *not* the generated object (as in
DiGress); the model still outputs a synthetic tabular row.

By structuring the denoiser around a graph, attention in each layer is
restricted to statistically or dynamically related features, which can improve
sample quality — especially on datasets with strong feature interactions.

---

## Architecture

Each tabular feature becomes a graph node:
- **Numerical features**: one node per feature; input is the scalar value projected to `d_model`.
- **Categorical features**: one node per feature; input is the one-hot slice projected to `d_model` via a per-feature linear layer.

Total nodes: `N = d_num + n_cat_features`.

The denoiser applies `n_layers` **graph-masked multi-head attention** blocks
(pre-norm transformer style) where attention weights are gated by the adjacency
matrix.  After the last layer, each node is projected back to its original
feature space (scalar for numerical, `K_i`-dim logits for categorical) and the
outputs are concatenated to reproduce the `d_in`-dimensional vector expected by
the diffusion process.

Timestep `t` and optional class labels `y` are injected by adding their
embeddings to every node's representation (broadcast over the node dimension),
identical to the conditioning in `MLPDiffusion`.

---

## Two Adjacency Modes

### `static`

Edges are derived **once from the training split** using statistical measures:

| Pair type   | Measure                       | Notes                              |
|-------------|-------------------------------|------------------------------------|
| num – num   | \|Pearson correlation\|       | Edge if score > `threshold`        |
| cat – cat   | Cramér's V                    | Edge if score > `threshold`        |
| num – cat   | Normalised mutual information | Edge if score > `threshold`        |

Self-loops are always included.  The resulting binary `(N, N)` tensor is
registered as a frozen buffer on the model and saved to `adj.pt` in the
experiment directory for use during sampling.

**When to use**: datasets with known or discoverable feature correlations;
cheaper to train than `dynamic`; fully deterministic across runs.

### `dynamic`

A learnable `(B, N, N)` soft adjacency is computed from the current node
embeddings at **every forward pass** via scaled dot-product attention
(`DynamicAdjacency` module).  Optionally the top-k neighbours per node can be
selected to encourage sparsity.

**When to use**: datasets where dependencies are complex or poorly captured by
pairwise statistics; allows the graph to specialise to the diffusion task.

---

## Configuration

Add a `[model.graph]` section to your experiment's TOML config:

```toml
[model.graph]
enabled       = true
mode          = "static"    # "static" | "dynamic"
threshold     = 0.1         # edge score threshold (static only)
n_layers      = 4           # number of graph attention layers
n_heads       = 4           # attention heads (must divide d_model)
d_model       = 128         # hidden dimension per node
sparsity_top_k = 0          # top-k neighbours per node (dynamic; 0 = dense)
```

**Backward compatibility**: when `enabled = false` (or the section is absent
entirely), the pipeline falls back to the original `MLPDiffusion` and produces
results identical to upstream TabDDPM.

### Example: static graph on the cardio dataset

```toml
model_type = "mlp"        # ignored when graph.enabled = true
num_numerical_features = 5

[model_params]
num_classes = 2
is_y_cond   = true

[model_params.rtdl_params]  # ignored when graph.enabled = true
d_layers = [256, 256]
dropout  = 0.0

[model.graph]
enabled   = true
mode      = "static"
threshold = 0.05
n_layers  = 4
n_heads   = 4
d_model   = 128
```

---

## Files Added or Modified

| Path | Change |
|---|---|
| `tab_ddpm/graph_builder.py` | New — `build_static_adjacency`, `DynamicAdjacency` |
| `tab_ddpm/graph_denoiser.py` | New — `GraphAttentionLayer`, `GraphAwareDenoiser` |
| `scripts/utils_train.py` | `get_model()` extended with `graph_params` / `adjacency` args |
| `scripts/train.py` | `graph_params` param; static adj build + logging before `get_model()` |
| `scripts/pipeline.py` | Extracts `raw_config['model']['graph']` and forwards it |
| `scripts/sample.py` | Loads saved `adj.pt`; passes `graph_params` to `get_model()` |
| `scripts/sanity_check_graph.py` | New — end-to-end check (static, dynamic, no-cat) |
| `exp/default/config.toml` | Added `[model.graph]` block with `enabled = false` |
| `docs/graph_denoiser.md` | This file |

---

## Running the Sanity Check

```bash
python scripts/sanity_check_graph.py
```

Expected output confirms:
1. Static adjacency shape and off-diagonal edge count.
2. Forward-pass output shape `== (B, d_in)` for both modes.
3. All model parameters receive gradients after a backward pass.
