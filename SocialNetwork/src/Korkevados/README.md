# Korkevados — Workspace Overview

This directory holds the GNN / data-pipeline work for the social-network
stock-market project. Notebooks are grouped by purpose: exploratory analysis,
ETL feeders, and two parallel GNN tracks (one tuned for the BGU cluster, one
for local experimentation).

## Folder layout

### `analysis notebooks/`
Exploratory and descriptive notebooks that do **not** train models. Used to
understand the underlying 13F data, profile investor behavior across quarters,
and load snapshots of the Δ-graph into Neo4j for visual inspection. This is
the first place to look when validating a new dataset slice or sanity-checking
graph statistics before feeding them into the GNN pipelines.

### `etl_data_pipes/`
Data-preparation notebooks that materialize the inputs the analysis and GNN
notebooks consume. Currently contains the fundamental-data insertion pipeline
that ingests per-CUSIP financial features (EPS, ROE, EV/EBITDA, …) into the
project database. Run these first whenever the upstream parquet / SQL inputs
change.

### `gnn_notebooks_cluster/`
GNN training notebooks designed to run on the BGU GPU cluster. Reads parquet
inputs from `~/13Fgnn/data/` and writes models, per-quarter result CSVs, and
CUSIP score parquets to `~/13Fgnn/`. Long-running sweeps live here — they're
resumable, GPU-aware, and write artifacts suffixed by run configuration so
parallel runs don't collide.

- **`old/`** — original reference implementations: bipartite tertile
  classifier, multi-quarter Approach A (block-diagonal union of K past
  quarters), and Approach B (unified temporal graph with time-stacked
  features). These are the baseline before any of the v2 changes; keep them
  around as a regression reference.
- **`new/`** — v2 versions of the same three notebooks with three substantive
  changes: (1) `WeightedSAGE` actually consumes `edge_weight` (with an
  automatic fallback to `GraphConv` if the installed PyG is too old to give
  `SAGEConv` an `edge_weight` arg), (2) `WeightedGAT` is replaced by GATv2
  whose attention conditions on a per-edge feature `z-score(w · log_aum)`,
  and (3) every quarterly evaluation appends a CUSIP ranking parquet
  (`cusip, year, quarter, score = P(top tertile)`) so downstream portfolio
  construction has a per-quarter signal. Each sweep loops over both edge-weight
  columns (`change_in_weight`, `change_in_adjusted_weight`); all output
  filenames are suffixed by the column name so the two runs stay separate.

### `gnn_notebooks_locally/`
Smaller, faster-iterating GNN notebooks meant for a local machine — useful for
developing model variants and sanity-checking ideas before committing to a
cluster sweep. Includes the original bipartite classifier and tertile
formulations alongside REV2-based stock-ranking experiments (and a
differentiable REV3 variant). Datasets here are typically a single quarter
or a small window so iteration cycles stay short.
