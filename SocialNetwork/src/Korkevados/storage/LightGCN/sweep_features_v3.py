#!/usr/bin/env python
"""sweep_features_v3.py — Multi-quarter LightGCN sweep (v3) with next-quarter evaluation.

Direct conversion of RobustLightGCN_features_sweep_v3.ipynb. v3 differences vs v2:
  - For each quarter Q, scores Q+1's actual Δ-edges using Q's trained model
    (restricted to the funds/stocks shared between Q and Q+1).
  - Reports OOS AUC, F1, Hit@K, NDCG@K — true forecasting metrics.
  - Computes Spearman corr(mean_score @ Q, log_return @ Q+1).
  - cusip_ranks_v3 rows carry both (year, quarter) [training] and
    (predicts_year, predicts_quarter) [= Q+1, the quarter being forecast].

Outputs (in --results-dir, default /results/LightGcnV3):
  - sweep_results_v3__<edges-col>.csv      one row per quarter (in-sample + OOS)
  - cusip_ranks_v3__<edges-col>.parquet    per-stock per-quarter rank, tagged

Resumable: re-running picks up from the last completed quarter (skips quarters
already in the CSV). All file writes are atomic (.tmp + os.replace).
SIGTERM/SIGINT cause a graceful stop after the current quarter saves.

Usage:
    python sweep_features_v3.py
    python sweep_features_v3.py --edges-col change_in_adjusted_weight
    python sweep_features_v3.py --quarters 2024Q1,2024Q2 --data-dir ../../data
"""
import argparse
import os
import random
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
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")


STOCK_FEATURE_COLS = [
    "diluted_eps", "roe", "ev_ebitda", "pe_ratio", "price_to_sales",
    "price_to_book", "debt_to_equity", "dividend_yield", "fcf_per_share",
    "log_return",
]


# =====================================================================
# Graceful shutdown
# =====================================================================

_SHOULD_STOP = False


def _on_signal(signum, _frame):
    global _SHOULD_STOP
    try:
        name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        name = str(signum)
    print(f"\n[signal] received {name}; will stop after the current quarter "
          f"finishes and saves.", flush=True)
    _SHOULD_STOP = True


def install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, AttributeError, OSError):
            pass


# =====================================================================
# CLI
# =====================================================================

def parse_quarters_arg(s):
    if s is None or s == "":
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip().upper()
        if "Q" not in tok:
            raise ValueError(f"bad quarter token: {tok!r}; expected e.g. 2024Q1")
        y, q = tok.split("Q")
        out.append((int(y), int(q)))
    return out


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--edges-col", default="change_in_weight",
                   choices=["change_in_weight", "change_in_adjusted_weight"])
    p.add_argument("--data-dir", default=None,
                   help="Path to parquet dir. Default: $FGNN_DATA_DIR -> "
                        "<script>/../../data -> ~/13Fgnn/data")
    p.add_argument("--results-dir", default=None,
                   help="Output dir. Default: $FGNN_RESULTS_DIR -> <script>/results/LightGcnV3")
    # v2 hyperparameters (defaults match the v3 notebook)
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--embed-dim",     type=int,   default=128)
    p.add_argument("--num-layers",    type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=16384,
                   help="Positives per mini-batch")
    p.add_argument("--num-negatives", type=int,   default=5,
                   help="Negatives per positive, resampled every batch")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--l2-emb",        type=float, default=1e-5)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--patience",      type=int,   default=25)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--top-k",         default="5,10,20,50",
                   help="Comma-separated K values for hit/ndcg metrics")
    p.add_argument("--quarters",      default=None,
                   help="Comma-separated, e.g. 2024Q1,2024Q2; default: all available")
    p.add_argument("--device",        default=None, choices=["cuda", "cpu"])
    p.add_argument("--model-tag",     default="WeightedLightGCN_v3",
                   help="Identifier saved in the ranks parquet 'model' column")
    return p.parse_args()


# =====================================================================
# Loaders
# =====================================================================

def make_loaders(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    def _read(name):
        p = data_dir / f"{name}.parquet"
        df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        print(f"  loaded {name:30s} {len(df):>12,} rows", flush=True)
        return df

    print(f"reading parquets from {data_dir}", flush=True)
    CHANGED = _read("changed_holdings")
    RETURNS = _read("stocks_return")
    AUM     = _read("cik_aum")
    NORM    = _read("normalized_holdings")
    FIN     = _read("cusip_financial_data")

    def load_edges(year, quarter, col):
        if CHANGED.empty or col not in CHANGED.columns:
            return pd.DataFrame(columns=["cik", "cusip", "w"])
        m = (CHANGED["year"] == year) & (CHANGED["quarter"] == quarter) & CHANGED[col].notna()
        return (CHANGED.loc[m, ["cik", "cusip", col]]
                .rename(columns={col: "w"}).reset_index(drop=True))

    def load_returns(year, quarter):
        if RETURNS.empty:
            return pd.DataFrame(columns=["cusip", "log_return"])
        m = (RETURNS["year"] == year) & (RETURNS["quarter"] == quarter) & RETURNS["log_return"].notna()
        return RETURNS.loc[m, ["cusip", "log_return"]].reset_index(drop=True)

    def load_aum(year, quarter):
        if AUM.empty:
            return pd.DataFrame(columns=["cik", "aum"])
        m = (AUM["year"] == year) & (AUM["quarter"] == quarter) & (AUM["total"] > 0)
        return AUM.loc[m, ["cik", "total"]].rename(columns={"total": "aum"}).reset_index(drop=True)

    def load_stock_features(year, quarter):
        if FIN.empty:
            fin = pd.DataFrame(columns=["cusip"] + STOCK_FEATURE_COLS)
        else:
            fin = FIN.loc[(FIN["year"] == year) & (FIN["quarter"] == quarter)].copy()
        rets = load_returns(year, quarter)
        df = fin.merge(rets, on="cusip", how="outer")
        for c in STOCK_FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0.0
        return df[["cusip"] + STOCK_FEATURE_COLS]

    def investor_profitability(year, quarter):
        ny, nq = next_year_quarter(year, quarter)
        if NORM.empty:
            return pd.Series(dtype=float, name="profitability")
        h = NORM.loc[(NORM["year"] == year) & (NORM["quarter"] == quarter),
                     ["cik", "cusip", "weight"]]
        r = load_returns(ny, nq)
        m = h.merge(r, on="cusip", how="inner")
        m["contrib"] = m["weight"] * m["log_return"]
        return m.groupby("cik")["contrib"].sum().rename("profitability")

    def list_available_quarters(col):
        sub = CHANGED.loc[CHANGED[col].notna(), ["year", "quarter"]].drop_duplicates()
        yq = sorted({(int(y), int(q)) for y, q in sub.itertuples(index=False)})
        avail = set(yq)
        return [(y, q) for (y, q) in yq if prev_year_quarter(y, q) in avail]

    return {
        "load_edges": load_edges,
        "load_returns": load_returns,
        "load_aum": load_aum,
        "load_stock_features": load_stock_features,
        "investor_profitability": investor_profitability,
        "list_available_quarters": list_available_quarters,
    }


def next_year_quarter(y, q): return (y + 1, 1) if q == 4 else (y, q + 1)
def prev_year_quarter(y, q): return (y - 1, 4) if q == 1 else (y, q - 1)


def zscore(df, cols):
    out = df.copy()
    for c in cols:
        v = out[c].astype(float).replace([np.inf, -np.inf], np.nan)
        v = v.fillna(v.median() if v.notna().any() else 0.0)
        sd = v.std()
        out[c] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out


def build_feature_graph(year, quarter, col, loaders):
    edges = loaders["load_edges"](year, quarter, col)
    if edges.empty:
        raise RuntimeError(f"no Δ-edges for {year} Q{quarter}")
    aum = loaders["load_aum"](year, quarter)
    py, pq = prev_year_quarter(year, quarter)
    try:
        past_prof = loaders["investor_profitability"](py, pq).reset_index()
    except Exception:
        past_prof = pd.DataFrame(columns=["cik", "profitability"])

    cik_nh = edges.groupby("cik").size().rename("n_holdings").reset_index()
    cik_df = aum.merge(cik_nh, on="cik", how="outer").merge(past_prof, on="cik", how="left")
    cik_df["aum"] = cik_df["aum"].fillna(
        cik_df["aum"].median() if cik_df["aum"].notna().any() else 0.0)
    cik_df["log_aum"] = np.log(cik_df["aum"].clip(lower=1.0))
    cik_df["n_holdings"] = cik_df["n_holdings"].fillna(0)
    cik_df["profitability"] = cik_df["profitability"].fillna(0.0)
    CIK_FEATS = ["log_aum", "n_holdings", "profitability"]
    cik_df = zscore(cik_df, CIK_FEATS)
    stock_df = zscore(loaders["load_stock_features"](year, quarter), STOCK_FEATURE_COLS)

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

    edges = edges.copy()
    edges["src"] = edges["cik"].map(cik_pos).astype(np.int64)
    edges["dst"] = edges["cusip"].map(cusip_pos).astype(np.int64)
    return {
        "X": X,
        "edges": edges[["src", "dst", "w"]].reset_index(drop=True),
        "n_cik": len(cik_ids),
        "n_cusip": len(cusip_ids),
        "cik_ids": cik_ids,
        "cusip_ids": cusip_ids,
        "cik_pos": cik_pos,
        "cusip_pos": cusip_pos,
    }


def split_edges(edges_df, tr, va, te, seed):
    df = edges_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df); n_tr = int(n * tr); n_va = int(n * va)
    train = df.iloc[:n_tr].reset_index(drop=True)
    val   = df.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    test  = df.iloc[n_tr + n_va:].reset_index(drop=True)
    train_nodes = set(train["src"]).union(train["dst"])

    def reflow(part):
        m = ~part["src"].isin(train_nodes) | ~part["dst"].isin(train_nodes)
        return part[~m].reset_index(drop=True), part[m]

    val,  vm = reflow(val)
    test, tm = reflow(test)
    if len(vm) + len(tm) > 0:
        train = pd.concat([train, vm, tm], ignore_index=True)
    return train, val, test


def edges_to_index(df, device):
    src = df["src"].to_numpy(); dst = df["dst"].to_numpy()
    w = df["w"].to_numpy().astype(np.float32)
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    ew = np.concatenate([np.abs(w), np.abs(w)])
    return (torch.tensor(ei, dtype=torch.long, device=device),
            torch.tensor(ew, dtype=torch.float, device=device))


# =====================================================================
# Negative sampling (per-batch)
# =====================================================================

def build_forbid_set(edges_df):
    return set(zip(edges_df["src"].astype(int).tolist(),
                   edges_df["dst"].astype(int).tolist()))


def sample_negatives_batch(num_pos, n_cik, n_cusip, num_negatives, forbid, rng):
    target = num_pos * num_negatives
    out = np.empty((target, 2), dtype=np.int64)
    filled = 0
    while filled < target:
        need = target - filled
        m = max(need * 2, 4096)
        u = rng.integers(0, n_cik, size=m)
        v = rng.integers(n_cik, n_cik + n_cusip, size=m)
        keep_u, keep_v = [], []
        for a, b in zip(u, v):
            if (int(a), int(b)) not in forbid:
                keep_u.append(a); keep_v.append(b)
                if len(keep_u) >= need:
                    break
        k = len(keep_u)
        out[filled:filled + k, 0] = keep_u
        out[filled:filled + k, 1] = keep_v
        filled += k
    return out


# =====================================================================
# Model + loss
# =====================================================================

class WeightedLightGCN(nn.Module):
    def __init__(self, in_feats, embedding_dim, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(in_feats, embedding_dim)
        nn.init.normal_(self.input_proj.weight, std=0.1)
        nn.init.zeros_(self.input_proj.bias)
        self.convs = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim,
                    improved=False, cached=False, add_self_loops=True)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_weight):
        x = self.input_proj(x)
        layers = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            layers.append(x)
        return torch.stack(layers, dim=0).mean(dim=0)


def bpr_loss_with_l2(pos_u, pos_v, neg_u, neg_v, l2_emb):
    pos_scores = (pos_u * pos_v).sum(dim=1)
    neg_scores = (neg_u * neg_v).sum(dim=1)
    bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
    reg = (pos_u.pow(2).sum() + pos_v.pow(2).sum()
           + neg_u.pow(2).sum() + neg_v.pow(2).sum()) / pos_u.size(0)
    return bpr + l2_emb * reg, bpr.item()


# =====================================================================
# Training
# =====================================================================

def train_model(model, X, train_ei, train_ew, trainval_ei, trainval_ew,
                train_pos_np, val_pos_np,
                n_cik, n_cusip,
                forbid_train, forbid_trainval,
                epochs, lr, weight_decay, l2_emb, patience,
                batch_size, num_negatives, seed):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    n_train = len(train_pos_np)
    n_val = len(val_pos_np)
    train_pos_t = torch.tensor(train_pos_np, dtype=torch.long, device=X.device)
    val_pos_t   = torch.tensor(val_pos_np,   dtype=torch.long, device=X.device)

    train_losses, val_losses = [], []
    best_val = float("inf"); best_state = None; no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=X.device)
        epoch_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            pos_batch = train_pos_t[idx]
            B = pos_batch.size(0)
            neg_np = sample_negatives_batch(B, n_cik, n_cusip, num_negatives,
                                            forbid_train, rng)
            neg_batch = torch.tensor(neg_np, dtype=torch.long, device=X.device)
            pos_rep = pos_batch.repeat_interleave(num_negatives, dim=0)

            opt.zero_grad()
            emb = model(X, train_ei, train_ew)
            pu = emb[pos_rep[:, 0]]; pv = emb[pos_rep[:, 1]]
            nu = emb[neg_batch[:, 0]]; nv = emb[neg_batch[:, 1]]
            loss, bpr_val = bpr_loss_with_l2(pu, pv, nu, nv, l2_emb)
            loss.backward(); opt.step()
            epoch_loss_sum += bpr_val
            n_batches += 1
        train_losses.append(epoch_loss_sum / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            v_neg_np = sample_negatives_batch(n_val, n_cik, n_cusip, 1,
                                              forbid_trainval, rng)
            v_neg = torch.tensor(v_neg_np, dtype=torch.long, device=X.device)
            emb = model(X, trainval_ei, trainval_ew)
            vps = (emb[val_pos_t[:, 0]] * emb[val_pos_t[:, 1]]).sum(dim=1)
            vns = (emb[v_neg[:, 0]]    * emb[v_neg[:, 1]]).sum(dim=1)
            v_loss = -F.logsigmoid(vps - vns).mean().item()
        val_losses.append(v_loss)

        if v_loss < best_val - 1e-5:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# =====================================================================
# Evaluation (in-sample)
# =====================================================================

def evaluate_test(model, X, edge_index, edge_weight, pos_pairs, neg_pairs,
                  threshold=0.5):
    model.eval()
    with torch.no_grad():
        emb = model(X, edge_index, edge_weight)
        pu = emb[pos_pairs[:, 0]]; pv = emb[pos_pairs[:, 1]]
        nu = emb[neg_pairs[:, 0]]; nv = emb[neg_pairs[:, 1]]
        ps = (pu * pv).sum(dim=1).sigmoid().cpu().numpy()
        ns = (nu * nv).sum(dim=1).sigmoid().cpu().numpy()
    scores = np.concatenate([ps, ns])
    labels = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    auc = roc_auc_score(labels, scores)
    pred = (scores > threshold).astype(int)
    return {
        "auc": auc,
        "precision": precision_score(labels, pred, zero_division=0),
        "recall":    recall_score(labels, pred, zero_division=0),
        "f1":        f1_score(labels, pred, zero_division=0),
        "scores": scores, "labels": labels,
    }


def find_optimal_threshold(scores, labels):
    p, r, ths = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / (p + r + 1e-10)
    idx = int(np.argmax(f1))
    th = ths[idx] if idx < len(ths) else 0.5
    return float(th), float(average_precision_score(labels, scores))


def evaluate_top_k(model, X, edge_index, edge_weight, test_edges_df,
                   seen_edges_df, n_cik, n_cusip, k_list):
    model.eval()
    with torch.no_grad():
        emb = model(X, edge_index, edge_weight)
    fund_to_test = {}
    for src, dst in zip(test_edges_df["src"].to_numpy(),
                        test_edges_df["dst"].to_numpy()):
        fund_to_test.setdefault(int(src), set()).add(int(dst))
    fund_to_seen = {}
    for src, dst in zip(seen_edges_df["src"].to_numpy(),
                        seen_edges_df["dst"].to_numpy()):
        fund_to_seen.setdefault(int(src), set()).add(int(dst))

    stock_indices = np.arange(n_cik, n_cik + n_cusip)
    stock_emb = emb[stock_indices]
    res = {k: {"hit_rate": 0.0, "ndcg": 0.0} for k in k_list}
    n_funds = len(fund_to_test)
    for fund_idx, true_stocks in fund_to_test.items():
        scores = (emb[fund_idx] * stock_emb).sum(dim=1).cpu().numpy()
        seen = fund_to_seen.get(fund_idx, set())
        if seen:
            seen_local = np.fromiter(
                (s - n_cik for s in seen if n_cik <= s < n_cik + n_cusip),
                dtype=np.int64,
            )
            if seen_local.size:
                scores[seen_local] = -np.inf
        order = np.argsort(-scores)
        for k in k_list:
            top_k_global = stock_indices[order[:k]]
            hits = [int(s) for s in top_k_global if int(s) in true_stocks]
            if hits:
                res[k]["hit_rate"] += 1
                dcg = sum(
                    1.0 / np.log2(i + 2)
                    for i, s in enumerate(top_k_global) if int(s) in true_stocks)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_stocks), k)))
                res[k]["ndcg"] += dcg / idcg if idcg > 0 else 0.0
    for k in k_list:
        res[k]["hit_rate"] /= max(n_funds, 1)
        res[k]["ndcg"]     /= max(n_funds, 1)
    return res


# =====================================================================
# v3: per-stock ranking + out-of-sample (next-quarter) evaluation
# =====================================================================

def compute_stock_ranking(model, X, edge_index, edge_weight,
                          n_cik, n_cusip, cusip_ids, year, quarter,
                          predicts_year, predicts_quarter,
                          chunk_funds=2048):
    """Per-stock mean attractiveness from Q's trained model.
    Tagged with the quarter being PREDICTED (= Q+1)."""
    model.eval()
    with torch.no_grad():
        emb = model(X, edge_index, edge_weight)
        fe = emb[:n_cik]; se = emb[n_cik:n_cik + n_cusip]
        accum = torch.zeros(n_cusip, device=emb.device)
        for start in range(0, n_cik, chunk_funds):
            end = min(start + chunk_funds, n_cik)
            block = torch.sigmoid(fe[start:end] @ se.T)
            accum += block.sum(dim=0)
        mean_score = (accum / max(n_cik, 1)).cpu().numpy()
    df = pd.DataFrame({
        "cusip":   list(cusip_ids),
        "year":    int(year),
        "quarter": int(quarter),
        "predicts_year":    int(predicts_year),
        "predicts_quarter": int(predicts_quarter),
        "mean_score": mean_score,
    }).sort_values("mean_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df


def evaluate_next_quarter(model, X, full_ei, full_ew, G,
                          next_year, next_quarter, edges_col,
                          k_list, seed, loaders):
    """Out-of-sample link prediction on Q+1's actual Δ-edges.
    Restricted to funds/stocks shared between Q and Q+1.
    Returns flat dict (NaN-filled if Q+1 unavailable)."""
    nan_result = {
        "oos_has_next": False, "oos_n_pos": 0,
        "oos_n_shared_funds": 0, "oos_n_shared_stocks": 0,
        "oos_auc": float("nan"), "oos_avg_precision": float("nan"),
        "oos_precision_05": float("nan"), "oos_recall_05": float("nan"),
        "oos_f1_05": float("nan"),
        **{f"oos_hit{k}":  float("nan") for k in k_list},
        **{f"oos_ndcg{k}": float("nan") for k in k_list},
    }
    try:
        next_edges_df = loaders["load_edges"](next_year, next_quarter, edges_col)
    except Exception:
        return nan_result
    if next_edges_df.empty:
        return nan_result

    cik_pos = G["cik_pos"]; cusip_pos = G["cusip_pos"]
    cik_set  = set(cik_pos.keys())
    cusip_set = set(cusip_pos.keys())
    shared_mask = next_edges_df["cik"].isin(cik_set) & next_edges_df["cusip"].isin(cusip_set)
    next_in_Q = next_edges_df[shared_mask].copy()
    if len(next_in_Q) == 0:
        nan_result["oos_has_next"] = True
        return nan_result
    next_in_Q["src"] = next_in_Q["cik"].map(cik_pos).astype(np.int64)
    next_in_Q["dst"] = next_in_Q["cusip"].map(cusip_pos).astype(np.int64)

    Q_pos_set = build_forbid_set(G["edges"])
    next_pos_set = build_forbid_set(next_in_Q)
    forbid = Q_pos_set | next_pos_set

    rng = np.random.default_rng(seed)
    n_pos = len(next_in_Q)
    neg_np = sample_negatives_batch(n_pos, G["n_cik"], G["n_cusip"], 1, forbid, rng)
    pos_pairs = torch.tensor(next_in_Q[["src", "dst"]].to_numpy(np.int64), device=X.device)
    neg_pairs = torch.tensor(neg_np, device=X.device)

    model.eval()
    with torch.no_grad():
        emb = model(X, full_ei, full_ew)
        ps = (emb[pos_pairs[:, 0]] * emb[pos_pairs[:, 1]]).sum(dim=1).sigmoid().cpu().numpy()
        ns = (emb[neg_pairs[:, 0]] * emb[neg_pairs[:, 1]]).sum(dim=1).sigmoid().cpu().numpy()
    scores = np.concatenate([ps, ns])
    labels = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    pred05 = (scores > 0.5).astype(int)

    oos_top_k = evaluate_top_k(model, X, full_ei, full_ew, next_in_Q, G["edges"],
                                G["n_cik"], G["n_cusip"], k_list)

    return {
        "oos_has_next": True,
        "oos_n_pos": int(n_pos),
        "oos_n_shared_funds":  int(next_in_Q["src"].nunique()),
        "oos_n_shared_stocks": int(next_in_Q["dst"].nunique()),
        "oos_auc": float(auc),
        "oos_avg_precision": float(ap),
        "oos_precision_05": float(precision_score(labels, pred05, zero_division=0)),
        "oos_recall_05":    float(recall_score(labels, pred05, zero_division=0)),
        "oos_f1_05":        float(f1_score(labels, pred05, zero_division=0)),
        **{f"oos_hit{k}":  float(oos_top_k[k]["hit_rate"]) for k in k_list},
        **{f"oos_ndcg{k}": float(oos_top_k[k]["ndcg"])     for k in k_list},
    }


def compute_rank_return_corr(rank_df, next_year, next_quarter, loaders):
    """Spearman corr(mean_score @ Q, log_return @ Q+1).
    Positive => higher predicted attractiveness aligns with higher next-quarter return."""
    nxt = loaders["load_returns"](next_year, next_quarter)
    if nxt.empty:
        return {"rank_return_spearman": float("nan"), "n_rank_return_pairs": 0}
    joined = rank_df[["cusip", "mean_score"]].merge(
        nxt[["cusip", "log_return"]], on="cusip", how="inner")
    if len(joined) < 5:
        return {"rank_return_spearman": float("nan"),
                "n_rank_return_pairs": int(len(joined))}
    rs = joined["mean_score"].rank()
    rr = joined["log_return"].rank()
    return {"rank_return_spearman": float(rs.corr(rr)),
            "n_rank_return_pairs": int(len(joined))}


# =====================================================================
# Per-quarter runner
# =====================================================================

def run_quarter(year, quarter, args, loaders, device):
    t0 = time.perf_counter()
    G = build_feature_graph(year, quarter, args.edges_col, loaders)
    train_e, val_e, test_e = split_edges(G["edges"], 0.8, 0.1, 0.1, args.seed)

    train_ei, train_ew = edges_to_index(train_e, device)
    trainval_df = pd.concat([train_e, val_e], ignore_index=True)
    tv_ei, tv_ew = edges_to_index(trainval_df, device)
    full_ei, full_ew = edges_to_index(G["edges"], device)
    X_t = torch.tensor(G["X"], dtype=torch.float, device=device)

    train_pos_np = train_e[["src", "dst"]].to_numpy(np.int64)
    val_pos_np   = val_e[["src", "dst"]].to_numpy(np.int64)

    forbid_train    = build_forbid_set(train_e)
    forbid_trainval = build_forbid_set(trainval_df)
    forbid_all      = build_forbid_set(pd.concat([trainval_df, test_e], ignore_index=True))

    model = WeightedLightGCN(G["X"].shape[1], args.embed_dim, args.num_layers).to(device)
    model, train_losses, val_losses = train_model(
        model, X_t, train_ei, train_ew, tv_ei, tv_ew,
        train_pos_np, val_pos_np,
        G["n_cik"], G["n_cusip"],
        forbid_train, forbid_trainval,
        args.epochs, args.lr, args.weight_decay, args.l2_emb, args.patience,
        args.batch_size, args.num_negatives, args.seed)

    # ---- in-sample test ----
    test_pos = torch.tensor(test_e[["src", "dst"]].to_numpy(np.int64), device=device)
    rng = np.random.default_rng(args.seed + 2)
    test_neg_np = sample_negatives_batch(
        len(test_e), G["n_cik"], G["n_cusip"], 1, forbid_all, rng)
    test_neg = torch.tensor(test_neg_np, device=device)

    res = evaluate_test(model, X_t, tv_ei, tv_ew, test_pos, test_neg, threshold=0.5)
    opt_th, ap = find_optimal_threshold(res["scores"], res["labels"])
    res_opt = evaluate_test(model, X_t, tv_ei, tv_ew, test_pos, test_neg, threshold=opt_th)

    k_list = [int(k) for k in args.top_k.split(",")]
    in_top_k = evaluate_top_k(
        model, X_t, tv_ei, tv_ew, test_e, trainval_df,
        G["n_cik"], G["n_cusip"], k_list)

    # ---- out-of-sample: predict Q+1 ----
    next_y, next_q = next_year_quarter(year, quarter)
    oos = evaluate_next_quarter(
        model, X_t, full_ei, full_ew, G,
        next_y, next_q, args.edges_col,
        k_list, args.seed + 3, loaders)

    # ---- per-stock ranking (tagged with Q+1 as predict target) ----
    rank_df = compute_stock_ranking(
        model, X_t, full_ei, full_ew,
        G["n_cik"], G["n_cusip"], G["cusip_ids"], year, quarter,
        predicts_year=next_y, predicts_quarter=next_q)

    # ---- rank-vs-next-return Spearman ----
    rrc = compute_rank_return_corr(rank_df, next_y, next_q, loaders)

    metrics = {
        "year": int(year), "quarter": int(quarter),
        "predicts_year": int(next_y), "predicts_quarter": int(next_q),
        "n_funds": int(G["n_cik"]), "n_stocks": int(G["n_cusip"]),
        "n_edges": int(len(G["edges"])),
        "n_train": int(len(train_e)), "n_val": int(len(val_e)),
        "n_test":  int(len(test_e)),
        "epochs_trained": int(len(train_losses)),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss":   float(val_losses[-1]),
        "best_val_loss":    float(min(val_losses)),
        # in-sample
        "auc":           float(res["auc"]),
        "avg_precision": float(ap),
        "opt_threshold": float(opt_th),
        "precision_05":  float(res["precision"]),
        "recall_05":     float(res["recall"]),
        "f1_05":         float(res["f1"]),
        "precision_opt": float(res_opt["precision"]),
        "recall_opt":    float(res_opt["recall"]),
        "f1_opt":        float(res_opt["f1"]),
        **{f"hit{k}":  float(in_top_k[k]["hit_rate"]) for k in k_list},
        **{f"ndcg{k}": float(in_top_k[k]["ndcg"])     for k in k_list},
        # out-of-sample
        **oos,
        # rank-return
        **rrc,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    del model, X_t, train_ei, train_ew, tv_ei, tv_ew, full_ei, full_ew
    del test_pos, test_neg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, rank_df


# =====================================================================
# Resumable, atomic I/O
# =====================================================================

def _atomic_write(write_fn, target_path):
    target_path = Path(target_path)
    tmp = target_path.with_name(target_path.name + ".tmp")
    try:
        write_fn(tmp)
        os.replace(tmp, target_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def append_metrics_row(row, csv_path):
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[warn] could not parse {csv_path.name} ({e!r}); "
                  f"backing up and starting fresh", flush=True)
            backup = csv_path.with_suffix(".csv.corrupt")
            os.replace(csv_path, backup)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    _atomic_write(lambda p: df.to_csv(p, index=False), csv_path)


def append_ranks(rank_df, parquet_path, model_tag):
    rank_df = rank_df.copy()
    rank_df["model"] = model_tag
    if parquet_path.exists():
        try:
            prev = pd.read_parquet(parquet_path)
            y = int(rank_df["year"].iloc[0]); q = int(rank_df["quarter"].iloc[0])
            prev = prev[~((prev["year"] == y) & (prev["quarter"] == q) &
                          (prev["model"] == model_tag))]
            rank_df = pd.concat([prev, rank_df], ignore_index=True)
        except Exception as e:
            print(f"[warn] could not parse {parquet_path.name} ({e!r}); "
                  f"backing up and starting fresh", flush=True)
            backup = parquet_path.with_suffix(".parquet.corrupt")
            os.replace(parquet_path, backup)
    _atomic_write(lambda p: rank_df.to_parquet(p, index=False), parquet_path)


def load_done_set(csv_path):
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[warn] could not parse {csv_path.name} ({e!r}); "
              f"treating as empty (no resume)", flush=True)
        return set()
    if "year" not in df.columns or "quarter" not in df.columns:
        return set()
    return set(zip(df["year"].astype(int), df["quarter"].astype(int)))


# =====================================================================
# Path resolution
# =====================================================================

def resolve_data_dir(arg_value):
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    if "FGNN_DATA_DIR" in os.environ:
        return Path(os.environ["FGNN_DATA_DIR"]).expanduser().resolve()
    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / ".." / "data",                              # cluster: 13Fgnn/python_files -> 13Fgnn/data
        script_dir / ".." / ".." / "data",                       # legacy
        script_dir / ".." / ".." / "Data" / "parquet_for_cluster",  # local Windows dev
        Path.home() / "13Fgnn" / "data",                         # cluster fallback
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(
        f"could not locate data dir; tried: {[str(c) for c in candidates]}\n"
        f"set --data-dir or $FGNN_DATA_DIR")


def resolve_results_dir(arg_value):
    if arg_value:
        d = Path(arg_value).expanduser().resolve()
    elif "FGNN_RESULTS_DIR" in os.environ:
        d = Path(os.environ["FGNN_RESULTS_DIR"]).expanduser().resolve()
    else:
        d = (Path(__file__).parent / "results" / "LightGcnV3").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


# =====================================================================
# Main
# =====================================================================

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    install_signal_handlers()

    args = get_args()
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = resolve_data_dir(args.data_dir)
    results_dir = resolve_results_dir(args.results_dir)

    print(f"=== sweep_features_v3.py ===", flush=True)
    print(f"started:        {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"PID:            {os.getpid()}", flush=True)
    print(f"device:         {device}", flush=True)
    if torch.cuda.is_available():
        try:
            print(f"  cuda device:  {torch.cuda.get_device_name(0)}", flush=True)
        except Exception:
            pass
    print(f"data_dir:       {data_dir}", flush=True)
    print(f"results_dir:    {results_dir}", flush=True)
    print(f"edges_col:      {args.edges_col}", flush=True)
    print(f"epochs:         {args.epochs}  (early-stop patience {args.patience})", flush=True)
    print(f"embed_dim:      {args.embed_dim}  layers: {args.num_layers}", flush=True)
    print(f"batch_size:     {args.batch_size}  num_negatives: {args.num_negatives}", flush=True)
    print(f"lr/wd/l2_emb:   {args.lr} / {args.weight_decay} / {args.l2_emb}", flush=True)
    print(f"top-K:          {args.top_k}", flush=True)
    print(f"model_tag:      {args.model_tag}", flush=True)
    print("", flush=True)

    loaders = make_loaders(data_dir)

    all_quarters = loaders["list_available_quarters"](args.edges_col)
    requested = parse_quarters_arg(args.quarters)
    if requested is not None:
        avail_set = set(all_quarters)
        quarters = [q for q in requested if q in avail_set]
        skipped = [q for q in requested if q not in avail_set]
        if skipped:
            print(f"  [skip] not available: {skipped}", flush=True)
    else:
        quarters = all_quarters

    results_csv  = results_dir / f"sweep_results_v3__{args.edges_col}.csv"
    ranks_parquet = results_dir / f"cusip_ranks_v3__{args.edges_col}.parquet"
    print(f"results CSV:    {results_csv}", flush=True)
    print(f"ranks parquet:  {ranks_parquet}", flush=True)

    done = load_done_set(results_csv)
    remaining = [yq for yq in quarters if yq not in done]
    print(f"available: {len(all_quarters)}  selected: {len(quarters)}  "
          f"done: {len(done)}  remaining: {len(remaining)}", flush=True)
    if remaining:
        print(f"  range: {remaining[0]} .. {remaining[-1]}", flush=True)
    print("", flush=True)

    t_start = time.perf_counter()
    failures = []
    completed_this_run = 0
    for i, (y, q) in enumerate(remaining, 1):
        if _SHOULD_STOP:
            print(f"\n[stop] received stop signal; "
                  f"finished {completed_this_run} quarter(s) this run, "
                  f"skipping the remaining {len(remaining) - i + 1}.", flush=True)
            break
        try:
            metrics, rank_df = run_quarter(y, q, args, loaders, device)
            append_metrics_row(metrics, results_csv)
            append_ranks(rank_df, ranks_parquet, args.model_tag)
            completed_this_run += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * max(len(remaining) - i, 0)
            oos_str = (f"oos_auc={metrics['oos_auc']:.4f} "
                       f"oos_hit10={metrics.get('oos_hit10', float('nan')):.4f} "
                       f"rrc={metrics.get('rank_return_spearman', float('nan')):.3f}"
                       if metrics.get("oos_has_next") else "oos=N/A")
            print(f"[{len(done)+i:2d}/{len(quarters)}] {y} Q{q}  "
                  f"in_auc={metrics['auc']:.4f}  {oos_str}  "
                  f"({metrics['epochs_trained']:>3}ep, {metrics['elapsed_s']:.0f}s)  "
                  f"ETA {eta/60:.1f}m", flush=True)
        except Exception as e:
            failures.append((y, q, repr(e)))
            print(f"  ! {y} Q{q} FAILED: {e.__class__.__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_min = (time.perf_counter() - t_start) / 60
    print(f"\nfinished in {total_min:.1f} min  "
          f"({completed_this_run} ok this run, {len(failures)} failed)", flush=True)
    if failures:
        print("failed quarters (will be retried on next run):", flush=True)
        for f in failures:
            print(f"  {f}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
