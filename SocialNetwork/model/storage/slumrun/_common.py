"""Shared utilities for old.py and new.py — 13F GNN tertile-prediction sweeps.

Encapsulates everything that is identical between the unweighted (v1) and
weighted (v2) runners: parquet loading, feature engineering, graph builders
(both flavors), training/eval loops, CUSIP scoring, atomic write/resume,
and the per-(approach × edges_col) orchestrator.

Two variants of the graph builders are exposed:
- weighted=False : matches the v1 `old/` notebooks. data.edge_weight is set,
                   data.edge_attr is None. Models that ignore edge_weight see
                   exactly the v1 behaviour.
- weighted=True  : matches the v2 `new/` notebooks. data.edge_attr =
                   z-score(w * log_aum_src), passed via forward_fn into models
                   that consume it.

Each runner injects (sage_cls, gat_cls, forward_fn) so this module never has
to know which model classes are in play.

Resumable + atomic writes follow the python_files/sweep_features_v4.py idiom:
load_done_set() seeds skip-set; append_metrics_row() and append_cusip_scores()
write to <path>.tmp then os.replace() to the final path.
"""
from __future__ import annotations

import contextlib
import gc
import os
import pickle
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# env / paths
# ----------------------------------------------------------------------------

OUT_DIR = Path(os.environ.get("FGNN_OUT_DIR", str(Path.home() / "13Fgnn"))).expanduser().resolve()
DATA_DIR = Path(os.environ.get("FGNN_DATA_DIR", str(OUT_DIR / "data"))).expanduser().resolve()
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

SMOKE = os.environ.get("FGNN_SMOKE", "0") == "1"

# bf16 autocast on CUDA cuts activation memory ~50% with no architecture change.
# RTX 4090/5090 (Ada/Blackwell) have native bf16. bf16 has fp32 dynamic range so
# no GradScaler is needed (unlike fp16). Disable via FGNN_NO_AMP=1.
USE_AMP = (torch.cuda.is_available()
           and os.environ.get("FGNN_NO_AMP", "0") != "1")


def _autocast_ctx():
    if USE_AMP:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

STOCK_FEATURE_COLS = [
    "diluted_eps", "roe", "ev_ebitda", "pe_ratio", "price_to_sales",
    "price_to_book", "debt_to_equity", "dividend_yield", "fcf_per_share",
    "log_return",
]
CIK_FEATS = ["log_aum", "n_holdings", "profitability"]

EDGES_COLUMN_NAMES = ["change_in_weight", "change_in_adjusted_weight"]
APPROACHES = ["bipartite", "multiq_A", "multiq_B"]

NUM_CLASSES = 3
HIDDEN_DIM = 32 if SMOKE else 64
NUM_LAYERS = 2
EPOCHS = 5 if SMOKE else 150
LR = 8e-4
DROPOUT = 0.5
GAT_HEADS = 4
K = 3

# ----------------------------------------------------------------------------
# parquet loaders
# ----------------------------------------------------------------------------

def _read_parquet_or_empty(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        print(f"  [warn] missing parquet: {p.name} (returning empty df)", flush=True)
        return pd.DataFrame()
    df = pd.read_parquet(p)
    print(f"  loaded {name:30s} {len(df):>10,} rows  {len(df.columns):>3} cols", flush=True)
    return df


def load_all_parquets():
    """Read all five inputs into module-level globals. Idempotent."""
    global CHANGED_HOLDINGS, STOCKS_RETURN, CIK_AUM, NORM_HOLDINGS, CUSIP_FIN
    if "CHANGED_HOLDINGS" in globals():
        return
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}. Set FGNN_DATA_DIR.")
    CHANGED_HOLDINGS = _read_parquet_or_empty("changed_holdings")
    STOCKS_RETURN    = _read_parquet_or_empty("stocks_return")
    CIK_AUM          = _read_parquet_or_empty("cik_aum")
    NORM_HOLDINGS    = _read_parquet_or_empty("normalized_holdings")
    CUSIP_FIN        = _read_parquet_or_empty("cusip_financial_data")


# ----------------------------------------------------------------------------
# date helpers + per-(year, quarter) data loaders
# ----------------------------------------------------------------------------

def next_year_quarter(year, quarter):
    return (year + 1, 1) if quarter == 4 else (year, quarter + 1)


def prev_year_quarter(year, quarter):
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def load_edges(year, quarter, edges_col_name):
    df = CHANGED_HOLDINGS
    if df.empty or edges_col_name not in df.columns:
        return pd.DataFrame(columns=["cik", "cusip", "w"])
    mask = (df["year"] == year) & (df["quarter"] == quarter) & df[edges_col_name].notna()
    return (df.loc[mask, ["cik", "cusip", edges_col_name]]
              .rename(columns={edges_col_name: "w"})
              .reset_index(drop=True))


def load_returns(year, quarter):
    df = STOCKS_RETURN
    if df.empty:
        return pd.DataFrame(columns=["cusip", "log_return"])
    mask = (df["year"] == year) & (df["quarter"] == quarter) & df["log_return"].notna()
    return df.loc[mask, ["cusip", "log_return"]].reset_index(drop=True)


def load_aum(year, quarter):
    df = CIK_AUM
    if df.empty:
        return pd.DataFrame(columns=["cik", "aum"])
    mask = (df["year"] == year) & (df["quarter"] == quarter) & (df["total"] > 0)
    return (df.loc[mask, ["cik", "total"]]
              .rename(columns={"total": "aum"})
              .reset_index(drop=True))


def load_stock_features(year, quarter):
    fin = CUSIP_FIN
    if fin.empty:
        fin = pd.DataFrame(columns=["cusip"] + STOCK_FEATURE_COLS)
    else:
        fin = fin.loc[(fin["year"] == year) & (fin["quarter"] == quarter)].copy()
    rets = load_returns(year, quarter)
    df = fin.merge(rets, on="cusip", how="outer")
    for c in STOCK_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[["cusip"] + STOCK_FEATURE_COLS]


def investor_profitability(year, quarter):
    ny, nq = next_year_quarter(year, quarter)
    h = NORM_HOLDINGS
    if h.empty:
        return pd.Series(dtype=float, name="profitability")
    h = h.loc[(h["year"] == year) & (h["quarter"] == quarter), ["cik", "cusip", "weight"]]
    r = load_returns(ny, nq)
    m = h.merge(r, on="cusip", how="inner")
    m["contrib"] = m["weight"] * m["log_return"]
    return m.groupby("cik")["contrib"].sum().rename("profitability")


# ----------------------------------------------------------------------------
# tertile labeller + z-score
# ----------------------------------------------------------------------------

def tertile_labels(values):
    s = values.astype(float)
    out = pd.Series(-1, index=s.index, dtype=np.int64)
    valid = s.dropna()
    if valid.empty:
        return out
    try:
        cats = pd.qcut(valid, q=3, labels=[0, 1, 2])
    except ValueError:
        q1, q2 = np.quantile(valid, [1 / 3, 2 / 3])
        cats = pd.cut(valid, bins=[-np.inf, q1, q2, np.inf], labels=[0, 1, 2], include_lowest=True)
    out.loc[valid.index] = cats.astype(np.int64)
    return out


def zscore(df, cols):
    out = df.copy()
    for c in cols:
        v = out[c].astype(float)
        v = v.replace([np.inf, -np.inf], np.nan).fillna(v.median() if v.notna().any() else 0.0)
        sd = v.std()
        out[c] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out


# ----------------------------------------------------------------------------
# feature pieces shared by all approaches
# ----------------------------------------------------------------------------

def cik_features_for(year, quarter, edges):
    """Z-scored CIK features [log_aum, n_holdings, profitability]."""
    aum = load_aum(year, quarter)
    py, pq = prev_year_quarter(year, quarter)
    try:
        past_prof = investor_profitability(py, pq).reset_index()
    except Exception:
        past_prof = pd.DataFrame(columns=["cik", "profitability"])
    cik_nh = edges.groupby("cik").size().rename("n_holdings").reset_index()
    cik_df = (aum.merge(cik_nh, on="cik", how="outer")
                  .merge(past_prof, on="cik", how="left"))
    cik_df["aum"] = cik_df["aum"].fillna(
        cik_df["aum"].median() if cik_df["aum"].notna().any() else 0.0)
    cik_df["log_aum"] = np.log(cik_df["aum"].clip(lower=1.0))
    cik_df["n_holdings"] = cik_df["n_holdings"].fillna(0)
    cik_df["profitability"] = cik_df["profitability"].fillna(0.0)
    return zscore(cik_df, CIK_FEATS)


def stock_features_for(year, quarter):
    return zscore(load_stock_features(year, quarter), STOCK_FEATURE_COLS)


def _load_label_sources(year, quarter):
    ny, nq = next_year_quarter(year, quarter)
    r_next = load_returns(ny, nq).set_index("cusip")["log_return"]
    prof_next = investor_profitability(year, quarter)
    return r_next, prof_next


# ----------------------------------------------------------------------------
# bipartite single-quarter graph builder (used by approach "bipartite" and as
# a sub-builder for approach A)
# ----------------------------------------------------------------------------

def _assemble_node_matrix(edges, cik_df, stock_df):
    cik_ids = pd.Index(edges["cik"].unique())
    cusip_ids = pd.Index(edges["cusip"].unique())
    cik_df = cik_df.set_index("cik").reindex(cik_ids).fillna(0.0)
    stock_df = stock_df.set_index("cusip").reindex(cusip_ids).fillna(0.0)
    F_cik = cik_df[CIK_FEATS].to_numpy()
    F_stk = stock_df[STOCK_FEATURE_COLS].to_numpy()
    Fdim = F_cik.shape[1] + F_stk.shape[1]
    X = np.zeros((len(cik_ids) + len(cusip_ids), Fdim), dtype=np.float32)
    X[:len(cik_ids), :F_cik.shape[1]] = F_cik
    X[len(cik_ids):, F_cik.shape[1]:] = F_stk
    cik_pos = {c: i for i, c in enumerate(cik_ids)}
    cusip_pos = {c: i + len(cik_ids) for i, c in enumerate(cusip_ids)}
    return X, cik_ids, cusip_ids, cik_pos, cusip_pos


def _assemble_edges(edges, cik_pos, cusip_pos, cik_df, *, weighted: bool):
    """Returns (edge_index, edge_weight, edge_attr-or-None).

    weighted=False mirrors the v1 notebooks (edge_attr=None).
    weighted=True  computes edge_attr = z-score(w * log_aum_src) per edge,
                   replicated for both directions.
    """
    src = edges["cik"].map(cik_pos).to_numpy()
    dst = edges["cusip"].map(cusip_pos).to_numpy()
    edge_index = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    w = edges["w"].to_numpy().astype(np.float32)
    edge_weight = np.concatenate([w, w])
    edge_attr = None
    if weighted:
        aum_map = cik_df.set_index("cik")["log_aum"]
        aum = edges["cik"].map(aum_map).fillna(0.0).to_numpy().astype(np.float32)
        attr = (w * aum).astype(np.float32)
        if attr.size > 0 and float(attr.std()) > 0:
            attr = (attr - attr.mean()) / attr.std()
        edge_attr = np.concatenate([attr, attr]).reshape(-1, 1)
    return edge_index, edge_weight, edge_attr


def _assign_labels(num_nodes, cusip_pos, cik_pos, r_next, prof_next):
    y = np.full(num_nodes, -1, dtype=np.int64)
    if not r_next.empty:
        stk_lbl = tertile_labels(r_next)
        for cusip, idx in cusip_pos.items():
            v = stk_lbl.get(cusip, -1)
            if v >= 0:
                y[idx] = int(v)
    if not prof_next.empty:
        inv_lbl = tertile_labels(prof_next)
        for cik, idx in cik_pos.items():
            v = inv_lbl.get(cik, -1)
            if v >= 0:
                y[idx] = int(v)
    return y


def build_graph(year, quarter, edges_col_name, *, weighted: bool):
    """Single-quarter bipartite graph. Returns (data, meta)."""
    edges = load_edges(year, quarter, edges_col_name)
    if edges.empty:
        raise RuntimeError(f"no Δ-edges for {year} Q{quarter}")
    cik_df = cik_features_for(year, quarter, edges)
    stock_df = stock_features_for(year, quarter)
    r_next, prof_next = _load_label_sources(year, quarter)
    X, cik_ids, cusip_ids, cik_pos, cusip_pos = _assemble_node_matrix(edges, cik_df, stock_df)
    edge_index, edge_weight, edge_attr = _assemble_edges(
        edges, cik_pos, cusip_pos, cik_df, weighted=weighted)
    y = _assign_labels(X.shape[0], cusip_pos, cik_pos, r_next, prof_next)

    kwargs = dict(
        x=torch.tensor(X),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight),
        y=torch.tensor(y),
    )
    if edge_attr is not None:
        kwargs["edge_attr"] = torch.tensor(edge_attr)
    data = Data(**kwargs)
    data.is_cik = torch.zeros(X.shape[0], dtype=torch.bool)
    data.is_cik[:len(cik_ids)] = True
    data.has_label = data.y >= 0
    meta = {
        "cik_ids": cik_ids, "cusip_ids": cusip_ids,
        "n_cik": len(cik_ids), "n_cusip": len(cusip_ids),
    }
    return data, meta


# ----------------------------------------------------------------------------
# Approach A — block-diagonal union of K past quarters
# ----------------------------------------------------------------------------

def past_K_quarters(target_year, target_quarter, K):
    out = []
    y, q = target_year, target_quarter
    for _ in range(K):
        out.insert(0, (y, q))
        y, q = prev_year_quarter(y, q)
    return out


def union_quarters(quarter_list, edges_col_name, *, weighted: bool):
    """Block-diagonal stack of build_graph(...) for each (y, q) in the list.
    Optionally concatenates edge_attr (when weighted=True)."""
    parts = [build_graph(y, q, edges_col_name, weighted=weighted) for y, q in quarter_list]
    datas = [d for d, _ in parts]

    offsets, n = [], 0
    for d in datas:
        offsets.append(n); n += d.num_nodes

    x = torch.cat([d.x for d in datas], dim=0)
    y = torch.cat([d.y for d in datas], dim=0)
    is_cik = torch.cat([d.is_cik for d in datas], dim=0)
    has_label = torch.cat([d.has_label for d in datas], dim=0)
    edge_index = torch.cat(
        [d.edge_index + off for d, off in zip(datas, offsets)], dim=1)
    edge_weight = torch.cat([d.edge_weight for d in datas], dim=0)

    kwargs = dict(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    if weighted:
        kwargs["edge_attr"] = torch.cat([d.edge_attr for d in datas], dim=0)
    out = Data(**kwargs)
    out.is_cik = is_cik
    out.has_label = has_label
    out.quarters = list(quarter_list)
    return out


# ----------------------------------------------------------------------------
# Approach B — temporal: shared nodes across K quarter slots
# ----------------------------------------------------------------------------

def quarters_window(target_year, target_quarter, K):
    out = []
    y, q = target_year, target_quarter
    for _ in range(K):
        out.insert(0, (y, q))
        y, q = prev_year_quarter(y, q)
    return out


def build_temporal_graph(target_year, target_quarter, K, edges_col_name, *, weighted: bool):
    """Temporal graph: same physical fund/CUSIP nodes shared across K slots,
    per-slot stacked features and per-slot edge_index (and edge_attr if weighted)."""
    window = quarters_window(target_year, target_quarter, K)

    pieces = []
    for (y, q) in window:
        edges = load_edges(y, q, edges_col_name)
        if edges.empty:
            raise RuntimeError(f"no Δ-edges for {y} Q{q}")
        cik_df = cik_features_for(y, q, edges)
        stock_df = stock_features_for(y, q)
        pieces.append((y, q, edges, cik_df, stock_df))

    all_ciks = pd.Index(sorted({c for _, _, e, _, _ in pieces for c in e["cik"].unique()}))
    all_cusips = pd.Index(sorted({c for _, _, e, _, _ in pieces for c in e["cusip"].unique()}))
    n_cik, n_cus = len(all_ciks), len(all_cusips)
    cik_pos = {c: i for i, c in enumerate(all_ciks)}
    cus_pos = {c: i + n_cik for i, c in enumerate(all_cusips)}
    n_nodes = n_cik + n_cus

    F_cik = len(CIK_FEATS)
    F_stk = len(STOCK_FEATURE_COLS)
    Fdim_per_q = F_cik + F_stk
    Fdim = K * Fdim_per_q
    X = np.zeros((n_nodes, Fdim), dtype=np.float32)
    for slot, (_, _, _, cik_df, stock_df) in enumerate(pieces):
        col_off = slot * Fdim_per_q
        cik_block = (cik_df.set_index("cik").reindex(all_ciks)[CIK_FEATS]
                          .fillna(0.0).to_numpy())
        X[:n_cik, col_off:col_off + F_cik] = cik_block
        stk_block = (stock_df.set_index("cusip").reindex(all_cusips)[STOCK_FEATURE_COLS]
                            .fillna(0.0).to_numpy())
        X[n_cik:, col_off + F_cik:col_off + Fdim_per_q] = stk_block

    src_l, dst_l, w_l, attr_l, off_l = [], [], [], [], []
    for slot, (_, _, edges, cik_df, _) in enumerate(pieces):
        offset = slot - (K - 1)
        s = edges["cik"].map(cik_pos).to_numpy()
        d = edges["cusip"].map(cus_pos).to_numpy()
        src_l.append(np.concatenate([s, d]))
        dst_l.append(np.concatenate([d, s]))
        w = edges["w"].to_numpy().astype(np.float32)
        w_l.append(np.concatenate([w, w]))
        if weighted:
            aum_map = cik_df.set_index("cik")["log_aum"]
            aum = edges["cik"].map(aum_map).fillna(0.0).to_numpy().astype(np.float32)
            attr = (w * aum).astype(np.float32)
            attr_l.append(np.concatenate([attr, attr]))
        off_l.append(np.full(2 * len(edges), offset, dtype=np.int8))

    edge_index = np.stack([np.concatenate(src_l), np.concatenate(dst_l)], axis=0)
    edge_weight = np.concatenate(w_l).astype(np.float32)
    edge_offset = np.concatenate(off_l)

    edge_attr = None
    if weighted:
        edge_attr_raw = np.concatenate(attr_l).astype(np.float32)
        if edge_attr_raw.size > 0 and float(edge_attr_raw.std()) > 0:
            edge_attr_raw = (edge_attr_raw - edge_attr_raw.mean()) / edge_attr_raw.std()
        edge_attr = edge_attr_raw.reshape(-1, 1)

    ny, nq = next_year_quarter(target_year, target_quarter)
    r_next = load_returns(ny, nq).set_index("cusip")["log_return"]
    prof_next = investor_profitability(target_year, target_quarter)
    y_arr = np.full(n_nodes, -1, dtype=np.int64)
    if not r_next.empty:
        stk_lbl = tertile_labels(r_next)
        for cusip, idx in cus_pos.items():
            v = stk_lbl.get(cusip, -1)
            if v >= 0:
                y_arr[idx] = int(v)
    if not prof_next.empty:
        inv_lbl = tertile_labels(prof_next)
        for cik, idx in cik_pos.items():
            v = inv_lbl.get(cik, -1)
            if v >= 0:
                y_arr[idx] = int(v)

    kwargs = dict(
        x=torch.tensor(X),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight),
        y=torch.tensor(y_arr),
    )
    if edge_attr is not None:
        kwargs["edge_attr"] = torch.tensor(edge_attr)
    data = Data(**kwargs)
    data.is_cik = torch.zeros(n_nodes, dtype=torch.bool)
    data.is_cik[:n_cik] = True
    data.has_label = data.y >= 0
    data.edge_offset = torch.tensor(edge_offset)

    meta = {
        "window": window, "K": K,
        "n_cik": n_cik, "n_cusip": n_cus,
        "cik_ids": all_ciks, "cusip_ids": all_cusips,
        "Fdim": Fdim, "Fdim_per_q": Fdim_per_q,
    }
    return data, meta


# ----------------------------------------------------------------------------
# train / eval — forward_fn lets the runner decide whether edge_attr is used
# ----------------------------------------------------------------------------

def train_one(model, data, train_mask, val_mask, *, forward_fn,
              epochs=None, lr=LR, verbose=False):
    if epochs is None:
        epochs = EPOCHS
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        with _autocast_ctx():
            logits = forward_fn(model, data)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad(), _autocast_ctx():
            logits = forward_fn(model, data)
            pred = logits.argmax(dim=-1)
            val_acc = (
                (pred[val_mask] == data.y[val_mask]).float().mean().item()
                if val_mask.any() else 0.0
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 25 == 0:
            print(f"  ep {ep:3d}  loss={loss.item():.4f}  val_acc={val_acc:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_subsets(model, data, mask, *, forward_fn):
    model.eval()
    with torch.no_grad(), _autocast_ctx():
        logits = forward_fn(model, data.to(DEVICE))
        pred = logits.argmax(dim=-1).cpu()
    y = data.y.cpu()
    out = {}
    for label, sel in [
        ("all", mask),
        ("stocks", mask & (~data.is_cik.cpu())),
        ("investors", mask & data.is_cik.cpu()),
    ]:
        if sel.sum() == 0:
            out[label] = {"n": 0}
            continue
        yt = y[sel].numpy()
        yp = pred[sel].numpy()
        out[label] = {
            "n": int(sel.sum()),
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
        }
    return out


def compute_cusip_scores(model, data, meta, year, quarter, *, forward_fn):
    """Score each test-graph CUSIP by P(top tertile next quarter)."""
    model.eval()
    with torch.no_grad(), _autocast_ctx():
        logits = forward_fn(model, data.to(DEVICE))
        # Cast back to fp32 before softmax — keeps probabilities numerically clean
        # in the parquet (autocast can leave logits in bf16).
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
    n_cik = meta["n_cik"]
    cusip_ids = meta["cusip_ids"]
    stock_probs = probs[n_cik:n_cik + len(cusip_ids)]
    df = pd.DataFrame({
        "cusip":   list(cusip_ids),
        "year":    int(year),
        "quarter": int(quarter),
        "score":   stock_probs[:, 2],
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ----------------------------------------------------------------------------
# atomic write + resume helpers (copied shape from sweep_features_v4.py)
# ----------------------------------------------------------------------------

def _atomic_replace(tmp_path: Path, target: Path):
    os.replace(tmp_path, target)


def append_metrics_row(row: dict, csv_path: Path):
    """Upsert one row into a metrics CSV, keyed by (year, quarter).

    A retry's outcome supersedes any prior row for the same quarter — without
    this, a quarter that fails GAT and gets re-tried later piles up multiple
    NaN-GAT rows. Mirrors the dedupe behaviour of `append_cusip_scores`.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if {"year", "quarter"}.issubset(df.columns):
            df = df[~((df["year"] == row["year"]) & (df["quarter"] == row["quarter"]))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    _atomic_replace(tmp, csv_path)


def append_cusip_scores(df_new: pd.DataFrame, parquet_path: Path,
                        year: int, quarter: int, model_tag: str):
    """Append CUSIP scores to a parquet via .tmp + os.replace, replacing any
    existing rows that match (year, quarter, model_tag)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = df_new.copy()
    df_new["model"] = model_tag
    if parquet_path.exists():
        prev = pd.read_parquet(parquet_path)
        prev = prev[~(
            (prev["year"] == year) &
            (prev["quarter"] == quarter) &
            (prev["model"] == model_tag)
        )]
        df_new = pd.concat([prev, df_new], ignore_index=True)
    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    df_new.to_parquet(tmp, index=False)
    _atomic_replace(tmp, parquet_path)


def load_done_set(csv_path: Path):
    """Quarters considered 'done' have BOTH sage_acc and gat_acc populated.

    Earlier versions keyed only on (year, quarter) — that meant a row written
    after a SAGE-only success (GAT failed) was treated as done forever, and
    re-runs skipped GAT silently. Requiring both ensures partial rows get
    retried instead of frozen.
    """
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    needed = {"sage_acc", "gat_acc"}
    if not needed.issubset(df.columns):
        return set()
    df = df.dropna(subset=["sage_acc", "gat_acc"])
    return set(zip(df["year"].astype(int), df["quarter"].astype(int)))


# ----------------------------------------------------------------------------
# quarter availability per approach
# ----------------------------------------------------------------------------

def list_available_quarters_bipartite(edges_col_name):
    """For approach 'bipartite' (train t-1, test t): need (prev, t, next) all available."""
    df = CHANGED_HOLDINGS
    sub = df.loc[df[edges_col_name].notna(), ["year", "quarter"]].drop_duplicates()
    avail = {(int(y), int(q)) for y, q in sub.itertuples(index=False)}
    out = []
    for y, q in sorted(avail):
        py, pq = prev_year_quarter(y, q)
        ny, nq = next_year_quarter(y, q)
        if (py, pq) in avail and (ny, nq) in avail:
            out.append((y, q))
    return out


def list_available_quarters_multiq(edges_col_name, K):
    """For approaches A and B: need next quarter and K past quarters all available."""
    df = CHANGED_HOLDINGS
    sub = df.loc[df[edges_col_name].notna(), ["year", "quarter"]].drop_duplicates()
    avail = {(int(y), int(q)) for y, q in sub.itertuples(index=False)}
    out = []
    for y, q in sorted(avail):
        ny, nq = next_year_quarter(y, q)
        if (ny, nq) not in avail:
            continue
        py, pq = prev_year_quarter(y, q)
        ok = True
        for _ in range(K):
            if (py, pq) not in avail:
                ok = False; break
            py, pq = prev_year_quarter(py, pq)
        if ok:
            out.append((y, q))
    return out


# ----------------------------------------------------------------------------
# graph-spec dispatch per approach
# ----------------------------------------------------------------------------

def _pad_to_dim(d: Data, Fdim: int) -> Data:
    if d.num_node_features < Fdim:
        pad_cols = torch.zeros(d.num_nodes, Fdim - d.num_node_features)
        d.x = torch.cat([d.x, pad_cols], dim=1)
    return d


def build_train_test(approach: str, year: int, quarter: int,
                     edges_col_name: str, *, weighted: bool):
    """Returns (train_data, test_data, test_meta, n_train_nodes, Fdim)."""
    if approach == "bipartite":
        py, pq = prev_year_quarter(year, quarter)
        train_data, _ = build_graph(py, pq, edges_col_name, weighted=weighted)
        test_data, test_meta = build_graph(year, quarter, edges_col_name, weighted=weighted)
        Fdim = max(train_data.num_node_features, test_data.num_node_features)
        train_data = _pad_to_dim(train_data, Fdim)
        test_data  = _pad_to_dim(test_data, Fdim)
        return train_data, test_data, test_meta, train_data.num_nodes, Fdim

    if approach == "multiq_A":
        py, pq = prev_year_quarter(year, quarter)
        train_quarters = past_K_quarters(py, pq, K)
        train_data = union_quarters(train_quarters, edges_col_name, weighted=weighted)
        test_data, test_meta = build_graph(year, quarter, edges_col_name, weighted=weighted)
        Fdim = max(train_data.num_node_features, test_data.num_node_features)
        train_data = _pad_to_dim(train_data, Fdim)
        test_data  = _pad_to_dim(test_data, Fdim)
        return train_data, test_data, test_meta, train_data.num_nodes, Fdim

    if approach == "multiq_B":
        py, pq = prev_year_quarter(year, quarter)
        train_data, train_meta = build_temporal_graph(py, pq, K, edges_col_name, weighted=weighted)
        test_data, test_meta = build_temporal_graph(year, quarter, K, edges_col_name, weighted=weighted)
        Fdim = max(train_meta["Fdim"], test_meta["Fdim"])
        return train_data, test_data, test_meta, train_data.num_nodes, Fdim

    raise ValueError(f"unknown approach: {approach}")


def list_available_quarters_for(approach: str, edges_col_name: str):
    if approach == "bipartite":
        return list_available_quarters_bipartite(edges_col_name)
    return list_available_quarters_multiq(edges_col_name, K)


# ----------------------------------------------------------------------------
# orchestrator: run a single (approach × edges_col) sweep with resume
# ----------------------------------------------------------------------------

_SHOULD_STOP = False


def _signal_handler(signum, _frame):
    global _SHOULD_STOP
    print(f"\n[signal {signum}] will stop after current quarter", flush=True)
    _SHOULD_STOP = True


def install_signal_handlers():
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _signal_handler)
        except (ValueError, OSError):
            pass


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_approach(*, approach: str, edges_col: str, sage_cls, gat_cls,
                 weighted: bool, forward_fn, results_root: Path,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 gat_heads: int = GAT_HEADS, hidden_dim: int = HIDDEN_DIM,
                 epochs: int = None):
    """One (approach × edges_col) corner of the sweep grid.

    Writes results to:
        <results_root>/<approach>/metrics__<edges_col>.csv
        <results_root>/<approach>/cusip_scores__<edges_col>.parquet

    Resumable: skips quarters already in the metrics CSV.
    Atomic writes: every CSV/parquet write goes through a .tmp file.
    """
    global _SHOULD_STOP
    out_dir = results_root / approach
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / f"metrics__{edges_col}.csv"
    scores_parq = out_dir / f"cusip_scores__{edges_col}.parquet"

    quarters = list_available_quarters_for(approach, edges_col)
    if not quarters:
        print(f"  no quarters for {approach} / {edges_col}, skipping", flush=True)
        return

    if SMOKE:
        # one quarter near the middle of the range
        quarters = [quarters[len(quarters) // 2]]

    done = load_done_set(metrics_csv)
    remaining = [(y, q) for y, q in quarters if (y, q) not in done]
    print(f"\n{'='*70}", flush=True)
    print(f"  {approach}  | edges_col={edges_col}  | quarters={len(quarters)}  "
          f"done={len(done)}  remaining={len(remaining)}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.perf_counter()
    for i, (y, q) in enumerate(remaining, 1):
        if _SHOULD_STOP:
            print(f"  [stop] received signal — exiting before {y}Q{q}", flush=True)
            break

        ny, nq = next_year_quarter(y, q)
        row = {
            "approach": approach, "edges_col": edges_col,
            "year": y, "quarter": q,
            "predicts_year": ny, "predicts_quarter": nq,
        }

        try:
            train_data, test_data, test_meta, n_train_nodes, Fdim = build_train_test(
                approach, y, q, edges_col, weighted=weighted)
            row["n_train_nodes"] = n_train_nodes
            row["n_test_nodes"]  = test_data.num_nodes
            row["n_test_edges"]  = test_data.num_edges
            row["Fdim"]          = Fdim
        except Exception as e:
            print(f"  ! build failed {y}Q{q}: {e.__class__.__name__}: {e}", flush=True)
            append_metrics_row(row, metrics_csv)
            continue

        for tag, model_cls, kw in [
            ("sage", sage_cls, dict(num_layers=num_layers, dropout=dropout)),
            ("gat",  gat_cls,  dict(num_layers=num_layers, heads=gat_heads, dropout=dropout)),
        ]:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                model = model_cls(Fdim, hidden_dim, **kw)
                model = train_one(
                    model, train_data, train_data.has_label, train_data.has_label,
                    forward_fn=forward_fn, epochs=epochs)
                results = eval_subsets(model, test_data, test_data.has_label,
                                        forward_fn=forward_fn)
                row[f"{tag}_train_s"]    = time.perf_counter() - t0
                row[f"{tag}_peak_gb"]    = (torch.cuda.max_memory_allocated() / 1e9
                                              if torch.cuda.is_available() else 0.0)
                row[f"{tag}_acc"]        = results["all"].get("accuracy")
                row[f"{tag}_f1"]         = results["all"].get("macro_f1")
                row[f"{tag}_stocks_acc"] = results["stocks"].get("accuracy")
                row[f"{tag}_inv_acc"]    = results["investors"].get("accuracy")

                scores_df = compute_cusip_scores(
                    model, test_data, test_meta, y, q, forward_fn=forward_fn)
                append_cusip_scores(scores_df, scores_parq, y, q, tag)
            except Exception as e:
                print(f"  ! {tag.upper()} {y}Q{q} failed: {e.__class__.__name__}: {e}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
            finally:
                # Move graphs and model back to CPU so empty_cache() can reclaim
                # the GPU before the next model (or quarter) allocates fresh.
                # Without this, train_data/test_data stay GPU-resident between
                # SAGE and GAT and memory steadily climbs across quarters,
                # OOM'ing GAT on multiq_A / multiq_B around quarter 25-30.
                try:
                    train_data = train_data.cpu()
                    test_data = test_data.cpu()
                except Exception:
                    pass
                try:
                    model = model.cpu()
                    del model
                except NameError:
                    pass
                cleanup()

        # release this quarter's graphs entirely before the next quarter's
        # build_train_test allocates fresh ones.
        del train_data, test_data, test_meta
        cleanup()

        append_metrics_row(row, metrics_csv)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / max(i, 1) * (len(remaining) - i)
        print(f"  [{len(done) + i:3d}/{len(quarters)}] {y}Q{q}  "
              f"SAGE={row.get('sage_acc', float('nan')):.3f}  "
              f"GAT={row.get('gat_acc', float('nan')):.3f}  "
              f"ETA {eta/60:.1f}m", flush=True)

    print(f"  finished {approach} / {edges_col} in {(time.perf_counter() - t_start)/60:.1f} min", flush=True)


def run_all(*, version: str, sage_cls, gat_cls, weighted: bool, forward_fn,
            approaches=None):
    """Entry-point shared by old.py / old_A.py / old_B.py.

    Iterates ``approaches`` × 2 edge columns. ``approaches`` defaults to all of
    APPROACHES; pass a subset (e.g. ["multiq_A"]) to scope the run to a single
    approach so two SLURM jobs can split the work. Outputs land under
        <FGNN_OUT_DIR>/notebooks/slumrun/results/<version>/<approach>/...
    Resolution prefers the script's own dir (sibling to results/) so the
    same code works on a laptop and on the cluster.
    """
    if approaches is None:
        approaches = APPROACHES
    install_signal_handlers()
    load_all_parquets()

    # results/ lives next to this _common.py file
    here = Path(__file__).parent.resolve()
    results_root = here / "results" / version
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"version    : {version}", flush=True)
    print(f"approaches : {approaches}", flush=True)
    print(f"OUT_DIR    : {OUT_DIR}", flush=True)
    print(f"DATA_DIR   : {DATA_DIR}", flush=True)
    print(f"results    : {results_root}", flush=True)
    print(f"device     : {DEVICE}", flush=True)
    print(f"smoke      : {SMOKE}", flush=True)

    for approach in approaches:
        if _SHOULD_STOP:
            break
        for edges_col in EDGES_COLUMN_NAMES:
            if _SHOULD_STOP:
                break
            run_approach(
                approach=approach,
                edges_col=edges_col,
                sage_cls=sage_cls,
                gat_cls=gat_cls,
                weighted=weighted,
                forward_fn=forward_fn,
                results_root=results_root,
            )

    print("all done.", flush=True)
