"""baseline_ml.py — non-graph ML baselines for stock tertile classification.

Why this exists
---------------
The slumrun/ pipeline only produces GNN results (GraphSAGE + GAT). To
answer "is the graph really helping?" we need standard tabular ML
classifiers trained on the same node features but with no message
passing. This file runs LogisticRegression, HistGradientBoosting, and
MLPClassifier (all sklearn, CPU-only, no extra deps) over the same
quarter set + same train/test split conventions as the GNN sweep, and
writes results into a parallel directory tree so baseline rows slot
directly next to GNN rows in the comparison tables.

Scope
-----
- Stocks only. CUSIP-tertile ranking is the headline output; investor
  accuracy is intentionally skipped.
- Three approaches mirrored from _common: bipartite, multiq_A, multiq_B.
  Train/test windows match each approach's GNN counterpart.
- Edge weights are NOT consumed (the whole point is to exclude graph
  signal). edges_col is only used to scope the available quarter set so
  (year, quarter) keys align with the GNN sweep.

Output layout
-------------
slumrun/results/baseline/<approach>/
    metrics__<edges_col>.csv          — one row per quarter, columns
                                          {model}_acc / _f1 / _stocks_acc /
                                          _train_s for each model.
    cusip_scores__<edges_col>.parquet — per-CUSIP P(top-tertile) ranks,
                                          one (year, quarter, model) block.

Resumable + atomic-write via _common.append_*. A quarter is "done" only
when every model's _acc column is populated.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# slurmrun/ on path so we can import _common (which lives in the parent dir).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

import SocialNetwork.src.Korkevados.storage.slumrun._common as _c

# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
# class_weight="balanced" matches the tertile-balanced GNN training. MLP
# hidden width 64 mirrors _common.HIDDEN_DIM for an apples-to-apples MLP-vs-GNN
# capacity comparison. early_stopping=True keeps MLP wall-time bounded.

MODELS = {
    "logreg": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", n_jobs=-1),
    "hgb": lambda: HistGradientBoostingClassifier(
        max_iter=200, class_weight="balanced", random_state=0),
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(64, 64), max_iter=200,
        random_state=0, early_stopping=True),
}


# ---------------------------------------------------------------------------
# feature/label builders — reuse _common's loaders
# ---------------------------------------------------------------------------

def _stock_features_indexed(year, quarter):
    """Z-scored stock features as a cusip-indexed DataFrame.
    _c.stock_features_for() already returns z-scored values, so no further
    normalisation is needed here.
    """
    return _c.stock_features_for(year, quarter).set_index("cusip")


def _stock_labels_for_target(target_year, target_quarter):
    """Tertile labels (0/1/2) on next-quarter log_return, keyed by cusip.

    Mirrors _common._assign_labels for stock nodes: tertiles are computed
    on the SAME quarter the GNN sees (the next_quarter return distribution).
    Unlabeled CUSIPs are dropped — caller filters its feature index against
    this label index.
    """
    ny, nq = _c.next_year_quarter(target_year, target_quarter)
    r_next = _c.load_returns(ny, nq).set_index("cusip")["log_return"]
    if r_next.empty:
        return pd.Series(dtype=np.int64)
    lab = _c.tertile_labels(r_next)
    return lab[lab >= 0].astype(np.int64)


def _build_single_quarter(year, quarter):
    """Per-CUSIP single-quarter (X, y, cusips) for stocks with both
    features AND a next-quarter label."""
    feats = _stock_features_indexed(year, quarter)
    labels = _stock_labels_for_target(year, quarter)
    common = feats.index.intersection(labels.index)
    X = feats.loc[common, _c.STOCK_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = labels.loc[common].to_numpy()
    cusips = np.array(common.tolist())
    return X, y, cusips


def _build_temporal_stack(target_year, target_quarter, K):
    """Per-CUSIP K-quarter feature concatenation.

    Mirrors _common.build_temporal_graph: for each CUSIP we concatenate K
    quarterly feature blocks (oldest..target). Quarters where a CUSIP has
    no feature row are zero-padded — same convention as
    `reindex(all_cusips).fillna(0.0)` in build_temporal_graph.
    Labels come from the TARGET quarter only (next-quarter return), which
    is also what the temporal GNN does.
    """
    window = []
    y, q = target_year, target_quarter
    for _ in range(K):
        window.insert(0, (y, q))
        y, q = _c.prev_year_quarter(y, q)

    feat_frames = [_stock_features_indexed(yy, qq) for (yy, qq) in window]
    all_cusips = pd.Index(sorted(set().union(*[df.index for df in feat_frames])))

    F_per_q = len(_c.STOCK_FEATURE_COLS)
    X = np.zeros((len(all_cusips), K * F_per_q), dtype=np.float32)
    for slot, df in enumerate(feat_frames):
        block = (df.reindex(all_cusips)[_c.STOCK_FEATURE_COLS]
                   .fillna(0.0).to_numpy(dtype=np.float32))
        X[:, slot * F_per_q:(slot + 1) * F_per_q] = block

    labels = _stock_labels_for_target(target_year, target_quarter)
    keep = all_cusips.intersection(labels.index)
    if len(keep) == 0:
        return np.zeros((0, K * F_per_q), dtype=np.float32), \
               np.zeros((0,), dtype=np.int64), np.array([])
    keep_pos = np.array([all_cusips.get_loc(c) for c in keep])
    return X[keep_pos], labels.loc[keep].to_numpy(), np.array(keep.tolist())


def build_quarter(approach, year, quarter, edges_col):
    """Returns (X_tr, y_tr, X_te, y_te, cusips_te, Fdim).

    Train/test windows match each GNN approach's _common.build_train_test:
    - bipartite: train on (y-1, q-1), test on (y, q), 10 features.
    - multiq_A : train on K=3 past-quarter rows concatenated, test on
                 (y, q), 10 features. Block-diagonal analogue.
    - multiq_B : per-CUSIP K-quarter feature stack at (y-1, q-1) for
                 train, same stack at (y, q) for test, 10*K=30 features.

    edges_col is unused inside (no graph signal); it's only here so the
    signature mirrors the GNN runner.
    """
    if approach == "bipartite":
        py, pq = _c.prev_year_quarter(year, quarter)
        X_tr, y_tr, _ = _build_single_quarter(py, pq)
        X_te, y_te, cusips_te = _build_single_quarter(year, quarter)
        return X_tr, y_tr, X_te, y_te, cusips_te, X_te.shape[1]

    if approach == "multiq_A":
        py, pq = _c.prev_year_quarter(year, quarter)
        train_quarters = _c.past_K_quarters(py, pq, _c.K)
        parts = [_build_single_quarter(yy, qq) for (yy, qq) in train_quarters]
        parts = [(X, y, c) for (X, y, c) in parts if X.shape[0] > 0]
        if not parts:
            X_tr = np.zeros((0, len(_c.STOCK_FEATURE_COLS)), dtype=np.float32)
            y_tr = np.zeros((0,), dtype=np.int64)
        else:
            X_tr = np.concatenate([p[0] for p in parts], axis=0)
            y_tr = np.concatenate([p[1] for p in parts], axis=0)
        X_te, y_te, cusips_te = _build_single_quarter(year, quarter)
        return X_tr, y_tr, X_te, y_te, cusips_te, X_te.shape[1]

    if approach == "multiq_B":
        py, pq = _c.prev_year_quarter(year, quarter)
        X_tr, y_tr, _ = _build_temporal_stack(py, pq, _c.K)
        X_te, y_te, cusips_te = _build_temporal_stack(year, quarter, _c.K)
        return X_tr, y_tr, X_te, y_te, cusips_te, X_te.shape[1]

    raise ValueError(f"unknown approach: {approach}")


# ---------------------------------------------------------------------------
# resume logic — quarter is "done" only when every model's _acc is populated
# ---------------------------------------------------------------------------

def _load_done_set(csv_path: Path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    needed = [f"{m}_acc" for m in MODELS]
    if not set(needed).issubset(df.columns):
        return set()
    df = df.dropna(subset=needed)
    return set(zip(df["year"].astype(int), df["quarter"].astype(int)))


# ---------------------------------------------------------------------------
# sweep orchestrator (mirrors _common.run_approach)
# ---------------------------------------------------------------------------

def run_baseline(approach, edges_col, results_root):
    out_dir = results_root / approach
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / f"metrics__{edges_col}.csv"
    scores_parq = out_dir / f"cusip_scores__{edges_col}.parquet"

    quarters = _c.list_available_quarters_for(approach, edges_col)
    if not quarters:
        print(f"  no quarters for {approach} / {edges_col}, skipping", flush=True)
        return
    if _c.SMOKE:
        quarters = [quarters[len(quarters) // 2]]

    done = _load_done_set(metrics_csv)
    remaining = [(y, q) for y, q in quarters if (y, q) not in done]
    print(f"\n{'=' * 70}", flush=True)
    print(f"  baseline | {approach} | edges_col={edges_col} | "
          f"quarters={len(quarters)} done={len(done)} remaining={len(remaining)}",
          flush=True)
    print(f"{'=' * 70}", flush=True)

    t_start = time.perf_counter()
    for i, (y, q) in enumerate(remaining, 1):
        ny, nq = _c.next_year_quarter(y, q)
        row = {
            "approach": approach, "edges_col": edges_col,
            "year": y, "quarter": q,
            "predicts_year": ny, "predicts_quarter": nq,
        }

        try:
            X_tr, y_tr, X_te, y_te, cusips_te, Fdim = build_quarter(
                approach, y, q, edges_col)
            row["n_train_nodes"] = int(X_tr.shape[0])
            row["n_test_nodes"] = int(X_te.shape[0])
            row["Fdim"] = int(Fdim)
        except Exception as e:
            print(f"  ! build failed {y}Q{q}: {type(e).__name__}: {e}", flush=True)
            _c.append_metrics_row(row, metrics_csv)
            continue

        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            print(f"  ! empty split {y}Q{q} (train={X_tr.shape[0]}, "
                  f"test={X_te.shape[0]}): skipping models", flush=True)
            _c.append_metrics_row(row, metrics_csv)
            continue

        for tag, factory in MODELS.items():
            try:
                t0 = time.perf_counter()
                model = factory()
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                proba = model.predict_proba(X_te)
                row[f"{tag}_acc"] = float(accuracy_score(y_te, pred))
                row[f"{tag}_f1"] = float(f1_score(
                    y_te, pred, average="macro", labels=[0, 1, 2], zero_division=0))
                # stocks_acc == acc here (stocks-only); kept for schema
                # parity with the GNN metrics CSV.
                row[f"{tag}_stocks_acc"] = row[f"{tag}_acc"]
                row[f"{tag}_train_s"] = time.perf_counter() - t0

                # P(top tertile) for CUSIP ranking. Some quarters can be
                # missing class 2 in y_tr (rare but possible), in which
                # case proba won't have it — guard via classes_.
                classes = list(model.classes_)
                if 2 in classes:
                    top_idx = classes.index(2)
                    top_score = proba[:, top_idx]
                else:
                    top_score = np.zeros(proba.shape[0], dtype=np.float32)

                scores_df = pd.DataFrame({
                    "cusip":   list(cusips_te),
                    "year":    int(y),
                    "quarter": int(q),
                    "score":   top_score.astype(np.float32),
                })
                scores_df = (scores_df.sort_values("score", ascending=False)
                                       .reset_index(drop=True))
                scores_df["rank"] = scores_df.index + 1
                _c.append_cusip_scores(scores_df, scores_parq, y, q, tag)
            except Exception as e:
                print(f"  ! {tag} {y}Q{q} failed: {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                sys.stdout.flush()

        _c.append_metrics_row(row, metrics_csv)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / max(i, 1) * (len(remaining) - i)
        accs = "  ".join(
            f"{m.upper()}={row.get(f'{m}_acc', float('nan')):.3f}" for m in MODELS)
        print(f"  [{len(done) + i:3d}/{len(quarters)}] {y}Q{q}  {accs}  "
              f"ETA {eta / 60:.1f}m", flush=True)

    print(f"  finished baseline / {approach} / {edges_col} in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min", flush=True)


def run_all(approaches=None):
    """Entry point shared by baseline_ml.py and the _A / _B drivers."""
    if approaches is None:
        approaches = _c.APPROACHES
    _c.install_signal_handlers()
    _c.load_all_parquets()

    # results/baseline/ is a SIBLING of slumrun/baseline/ (i.e. lives at
    # slumrun/results/baseline/, NOT under slumrun/baseline/results/). This
    # keeps it next to results/old/ and results/new/ produced by the GNN
    # sweeps, so downstream comparison code finds them in the same place.
    here = Path(__file__).resolve().parent
    results_root = here.parent / "results" / "baseline"
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"baseline runner", flush=True)
    print(f"approaches : {approaches}", flush=True)
    print(f"models     : {list(MODELS.keys())}", flush=True)
    print(f"OUT_DIR    : {_c.OUT_DIR}", flush=True)
    print(f"DATA_DIR   : {_c.DATA_DIR}", flush=True)
    print(f"results    : {results_root}", flush=True)
    print(f"smoke      : {_c.SMOKE}", flush=True)

    for approach in approaches:
        for edges_col in _c.EDGES_COLUMN_NAMES:
            run_baseline(approach, edges_col, results_root)

    print("all done.", flush=True)


def main():
    run_all()


if __name__ == "__main__":
    main()
