"""
Stock Market Social Network - JACCARD SIMILARITY Link Prediction
Lightweight script for weak servers with EXTENSIVE LOGGING.
Jaccard Similarity: Ratio of shared neighbors to total neighbors.
Score(A,B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|

Bipartite formulation for predicting fund → stock links:
  Score(f, s) = mean_{f' ≠ f, f' holds s in training} Jaccard(f, f')
  Path: fund → stock' → fund' → stock  (length-3)
"""

import os
import re
import gc
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score,
)
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_DIR = 'logs'
RESULTS_DIR = 'results'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f'jaccardsimilarity_{timestamp}.log')

def log_message(msg, to_console=True):
    """Write message to both console and log file."""
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    if to_console:
        print(msg)

# ============================================================================
# 1. DATA SETUP AND LOADING
# ============================================================================

def load_data():
    """Load reference data and quarterly holdings across ALL years."""
    personal_dir = os.path.expanduser('~')
    root = os.path.join(personal_dir, 'Social-Network-Stock-Market/SocialNetwork/parquet_files')
    output_dir = os.path.join(root, 'generated_combined_parquet')

    log_message("\n" + "=" * 80)
    log_message("STAGE 1: SYSTEM & PATH INITIALIZATION [JACCARD SIMILARITY]")
    log_message("=" * 80)
    log_message(f"[*] Base data directory set to: {root}")
    log_message(f"[*] Target parquet directory set to: {output_dir}")
    log_message(f"[*] Log file: {log_file}")

    log_message("\n" + "=" * 80)
    log_message("STAGE 2: LOADING HOLDINGS DATA (ALL YEARS)")
    log_message("=" * 80)

    combined_files = sorted([f for f in os.listdir(output_dir)
                             if f.startswith('holdings_filtered_new_period_start_') and f.endswith('.parquet')])

    if not combined_files:
        log_message("[!] ERROR: No processed files found in output_dir. Aborting.")
        return None

    log_message(f"[*] Found {len(combined_files)} potential parquet files.")

    all_dfs = []
    for file in combined_files:
        log_message(f"  [-] Reading file: {file}...")
        df_temp = pd.read_parquet(os.path.join(output_dir, file))

        date_col = 'period_start' if 'period_start' in df_temp.columns else 'PERIOD_DATE'

        if date_col in df_temp.columns:
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp['YEAR'] = df_temp[date_col].dt.year
            df_temp['QUARTER'] = 'Q' + df_temp[date_col].dt.quarter.astype(str)
        else:
            date_match = re.search(r'(\d{4})-\d{2}-\d{2}', file)
            if date_match:
                year_val = int(date_match.group(1))
                df_temp['YEAR'] = year_val
                month_match = re.search(r'\d{4}-(\d{2})-\d{2}', file)
                month_val = int(month_match.group(1)) if month_match else 1
                df_temp['QUARTER'] = f"Q{(month_val-1)//3 + 1}"
            else:
                log_message(f"  [!] Warning: Could not infer date for {file}, skipping.")
                continue

        year = df_temp['YEAR'].iloc[0]
        quarter_str = df_temp['QUARTER'].iloc[0]
        log_message(f"  [✓] Success: Loaded {len(df_temp):,} records assigned to {year} {quarter_str}")
        all_dfs.append(df_temp)

    log_message("\n[*] Concatenating all quarterly dataframes...")
    data = pd.concat(all_dfs, ignore_index=True)

    log_message("[*] Standardizing column names to uppercase...")
    data.columns = [c.upper() for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.reset_index(drop=True)

    if 'PERIOD_START' in data.columns:
        data = data.rename(columns={'PERIOD_START': 'PERIOD_DATE'})

    data['PERIOD_DATE'] = pd.to_datetime(data['PERIOD_DATE'])

    if 'VALUE' not in data.columns and 'SSHPRNAMT' in data.columns:
        log_message("[*] VALUE column missing. Imputing with SSHPRNAMT data.")
        data['VALUE'] = data['SSHPRNAMT']

    log_message(f"\n[✓] STAGE 2 COMPLETE. Total combined records loaded: {len(data):,}")
    return data

# ============================================================================
# 2. QUARTERLY GRAPH CONSTRUCTION
# ============================================================================

def build_quarterly_graphs(data):
    """Build separate bipartite graphs for each quarter."""
    log_message("\n" + "=" * 80)
    log_message("STAGE 3: BUILDING QUARTERLY GRAPHS")
    log_message("=" * 80)

    quarterly_graphs = {}

    for (year, quarter_str), group in data.groupby(['YEAR', 'QUARTER']):
        if isinstance(quarter_str, str):
            if '_' in quarter_str:
                quarter = int(quarter_str.split('_')[0][1])
            elif quarter_str.startswith('Q'):
                quarter = int(quarter_str[1])
            else:
                quarter = int(quarter_str)
        else:
            quarter = int(quarter_str)

        log_message(f"  [-] Building graph for {year} Q{quarter}...")
        funds  = group['CIK'].unique()
        stocks = group['CUSIP'].unique()

        G_bip = nx.Graph()
        G_bip.add_nodes_from(funds,  bipartite=0, node_type='fund')
        G_bip.add_nodes_from(stocks, bipartite=1, node_type='stock')

        # zip is ~10x faster than itertuples
        edges = list(zip(group['CIK'], group['CUSIP']))
        G_bip.add_edges_from(edges)

        quarterly_graphs[(year, quarter)] = G_bip
        log_message(f"  [✓] Graph constructed: {len(funds):,} funds, {len(stocks):,} stocks, {G_bip.number_of_edges():,} edges.")

    log_message(f"\n[✓] STAGE 3 COMPLETE. Generated {len(quarterly_graphs)} quarterly graphs.")
    return quarterly_graphs

# ============================================================================
# 3. SLIDING WINDOW & SAMPLING UTILITIES
# ============================================================================

def get_chronological_quarters(quarterly_graphs):
    return sorted(quarterly_graphs.keys())

def get_sliding_window_splits(chronological_quarters, train_window=8, test_offset=1):
    """
    Generate temporally correct train/test splits — no leakage.
    train_quarters: [i .. i+train_window-1]
    test_quarter:   i + train_window + test_offset - 1  (strictly after training window)
    """
    n = len(chronological_quarters)
    for i in range(n - train_window - test_offset + 1):
        train_quarters = chronological_quarters[i : i + train_window]
        test_quarter   = chronological_quarters[i + train_window + test_offset - 1]
        yield train_quarters, test_quarter

def get_shared_universe(G_train, G_test):
    """
    Extract shared universe (Q ∩ Q+1) for funds and stocks.
    Only entities appearing in BOTH last-train and test are evaluated.
    """
    funds_train  = {n for n, d in G_train.nodes(data=True) if d.get('node_type') == 'fund'}
    stocks_train = {n for n, d in G_train.nodes(data=True) if d.get('node_type') == 'stock'}
    test_funds   = {n for n, d in G_test.nodes(data=True)  if d.get('node_type') == 'fund'}
    test_stocks  = {n for n, d in G_test.nodes(data=True)  if d.get('node_type') == 'stock'}

    shared_funds  = funds_train  & test_funds
    shared_stocks = stocks_train & test_stocks
    return shared_funds, shared_stocks

def generate_test_samples(G_test, G_last_train, num_negatives_ratio=1.0, seed=42):
    """
    Generate balanced test set (positives + negatives) restricted to the
    shared universe (last train quarter ∩ test quarter) — no leakage.

    Negatives are sampled with numpy batch-random for speed.
    """
    log_message("    [-] Extracting positive holding edges from test graph...")

    shared_funds, shared_stocks = get_shared_universe(G_last_train, G_test)

    log_message(f"    [-] Train funds: {len({n for n, d in G_last_train.nodes(data=True) if d.get('node_type') == 'fund'}):,} | "
                f"Test funds: {len({n for n, d in G_test.nodes(data=True) if d.get('node_type') == 'fund'}):,} | "
                f"Shared: {len(shared_funds):,}")
    log_message(f"    [-] Train stocks: {len({n for n, d in G_last_train.nodes(data=True) if d.get('node_type') == 'stock'}):,} | "
                f"Test stocks: {len({n for n, d in G_test.nodes(data=True) if d.get('node_type') == 'stock'}):,} | "
                f"Shared: {len(shared_stocks):,}")

    if not shared_funds or not shared_stocks:
        log_message("    [!] Error: Empty shared universe.")
        return [], []

    # Positives: all edges in test within shared universe
    pos_edges = [(u, v) for u, v in G_test.edges()
                 if u in shared_funds and v in shared_stocks]

    if not pos_edges:
        log_message("    [!] Error: No positive edges found in shared universe.")
        return [], []

    log_message(f"    [-] Positive edges in shared universe: {len(pos_edges):,}")

    num_neg_target = int(len(pos_edges) * num_negatives_ratio)
    log_message(f"    [-] Beginning negative sampling (Ratio 1:{num_negatives_ratio}, target {num_neg_target:,})...")

    # Numpy batch random
    rng = np.random.default_rng(seed=seed)
    shared_funds_arr  = np.array(list(shared_funds))
    shared_stocks_arr = np.array(list(shared_stocks))

    neg_edges = []
    seen_neg  = set()
    oversample_factor = 5

    while len(neg_edges) < num_neg_target:
        batch_size = min((num_neg_target - len(neg_edges)) * oversample_factor, 500_000)
        fi = rng.integers(0, len(shared_funds_arr),  size=batch_size)
        si = rng.integers(0, len(shared_stocks_arr), size=batch_size)

        for f_idx, s_idx in zip(fi, si):
            f, s = shared_funds_arr[f_idx], shared_stocks_arr[s_idx]
            pair = (f, s)
            if pair not in seen_neg and not G_test.has_edge(f, s):
                neg_edges.append(pair)
                seen_neg.add(pair)
                if len(neg_edges) >= num_neg_target:
                    break

        if batch_size < oversample_factor:
            log_message("    [!] Warning: Could not reach negative target (graph too dense).")
            break

    log_message(f"    [✓] Generated {len(neg_edges):,} negative samples.")
    test_pairs = pos_edges + neg_edges
    y_true     = [1] * len(pos_edges) + [0] * len(neg_edges)

    return test_pairs, y_true, shared_funds, shared_stocks

# ============================================================================
# EVALUATION UTILITIES (shared with other baseline models)
# ============================================================================

def compute_ranking_metrics(test_pairs, y_true, y_prob, k_list=(5, 10, 20, 50)):
    """
    Compute Hit@k and NDCG@k by ranking candidate stocks per fund.

    For each fund:
      • Rank all candidate stocks by Jaccard score (descending)
      • Hit@k  : 1 if ANY true-positive stock is in top-k
      • NDCG@k : DCG@k / IDCG@k  (binary relevance)
    Returns averages over all funds with at least one positive.
    """
    fund_positives = {}
    fund_scores    = {}

    for (f, s), label, score in zip(test_pairs, y_true, y_prob):
        fund_positives.setdefault(f, set())
        fund_scores.setdefault(f, {})
        if label == 1:
            fund_positives[f].add(s)
        fund_scores[f][s] = float(score)

    hit_lists  = {k: [] for k in k_list}
    ndcg_lists = {k: [] for k in k_list}

    for fund, positives in fund_positives.items():
        if not positives:
            continue
        ranked = sorted(fund_scores[fund].items(), key=lambda x: -x[1])

        for k in k_list:
            top_k_stocks = {s for s, _ in ranked[:k]}
            hit_lists[k].append(1.0 if top_k_stocks & positives else 0.0)

            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, (s, _) in enumerate(ranked[:k])
                if s in positives
            )
            ideal_k = min(k, len(positives))
            idcg    = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
            ndcg_lists[k].append(dcg / idcg if idcg > 0 else 0.0)

    results = {}
    for k in k_list:
        results[f'hit{k}']  = float(np.mean(hit_lists[k]))  if hit_lists[k]  else 0.0
        results[f'ndcg{k}'] = float(np.mean(ndcg_lists[k])) if ndcg_lists[k] else 0.0
    return results


def evaluate_all_metrics(y_true, y_prob, test_pairs, model_name="Jaccard Similarity"):
    """
    Compute the full metrics suite (matches learned-model output format).

    final_train_loss, final_val_loss, best_val_loss  — NaN (no optimization)
    test_auc, test_avg_precision, opt_threshold
    test_precision/recall/f1_05   — fixed threshold 0.5
    test_precision/recall/f1_opt  — oracle threshold (max F1 on test)
    hit5/10/20/50, ndcg5/10/20/50
    rank_return_spearman           — Spearman(score, y_true)
    """
    log_message(f"    [-] Calculating full metrics suite for {model_name}...")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    has_both = len(np.unique(y_true)) > 1

    # 1. AUC & Average Precision
    test_auc           = roc_auc_score(y_true, y_prob)           if has_both else 0.5
    test_avg_precision = average_precision_score(y_true, y_prob)  if has_both else 0.0

    # 2. Optimal threshold (oracle: maximises F1 on test)
    if has_both:
        prec_c, rec_c, thr_c = precision_recall_curve(y_true, y_prob)
        f1_c = np.where(
            (prec_c[:-1] + rec_c[:-1]) > 0,
            2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1]),
            0.0
        )
        best_idx           = int(np.argmax(f1_c))
        opt_threshold      = float(thr_c[best_idx])
        test_precision_opt = float(prec_c[best_idx])
        test_recall_opt    = float(rec_c[best_idx])
        test_f1_opt        = float(f1_c[best_idx])
    else:
        opt_threshold      = 0.5
        test_precision_opt = 0.0
        test_recall_opt    = 0.0
        test_f1_opt        = 0.0

    # 3. Metrics at fixed threshold 0.5
    y_pred_05         = (y_prob >= 0.5).astype(int)
    test_precision_05 = precision_score(y_true, y_pred_05, zero_division=0)
    test_recall_05    = recall_score(y_true, y_pred_05,    zero_division=0)
    test_f1_05        = f1_score(y_true, y_pred_05,        zero_division=0)

    # 4. Ranking metrics
    ranking = compute_ranking_metrics(test_pairs, y_true.tolist(), y_prob.tolist())

    # 5. Rank-return Spearman (y_true as return proxy)
    corr, _ = spearmanr(y_prob, y_true)
    rank_return_spearman = float(corr) if not np.isnan(corr) else 0.0

    return {
        'final_train_loss':     np.nan,
        'final_val_loss':       np.nan,
        'best_val_loss':        np.nan,
        'test_auc':             test_auc,
        'test_avg_precision':   test_avg_precision,
        'opt_threshold':        opt_threshold,
        'test_precision_05':    test_precision_05,
        'test_recall_05':       test_recall_05,
        'test_f1_05':           test_f1_05,
        'test_precision_opt':   test_precision_opt,
        'test_recall_opt':      test_recall_opt,
        'test_f1_opt':          test_f1_opt,
        **ranking,
        'rank_return_spearman': rank_return_spearman,
    }

# ============================================================================
# 4. JACCARD SIMILARITY LINK PREDICTION
# ============================================================================

def jaccard_similarity_score(train_quarters, test_pairs, quarterly_graphs,
                              shared_funds=None, shared_stocks=None):
    """
    Bipartite Jaccard Similarity — memory-efficient, vectorized via scipy sparse.

    Score(f, s) = mean_{f' ≠ f, f' holds s in training}  Jaccard(f, f')
    where:
        Jaccard(f, f') = |stocks(f) ∩ stocks(f')| / |stocks(f) ∪ stocks(f')|
                       = B[f,:] · B[f',:] / (deg(f) + deg(f') - B[f,:] · B[f',:])

    Key memory optimization: We do NOT materialize the full F×F Jaccard matrix.
    Instead, for each test pair (f, s) we fetch only the sparse row B[f, :] and
    the rows B[other_idxs, :] (funds that hold s) — O(K × S) per pair, where K
    is typically small and S is the stock universe size.

    NOTE: All computation uses TRAINING data only — no leakage.
    """
    log_message("    [-] Computing Bipartite Jaccard Similarity scores (memory-efficient sparse)...")

    # ── Collect node IDs from training quarters ──────────────────────────────
    all_fund_ids  = set()
    all_stock_ids = set()
    for yq in train_quarters:
        G_q = quarterly_graphs.get(yq)
        if G_q:
            for n, d in G_q.nodes(data=True):
                if d.get('node_type') == 'fund':
                    all_fund_ids.add(n)
                else:
                    all_stock_ids.add(n)

    # Apply shared universe filter
    if shared_funds is not None:
        all_fund_ids = all_fund_ids & shared_funds
    if shared_stocks is not None:
        all_stock_ids = all_stock_ids & shared_stocks

    funds  = list(all_fund_ids)
    stocks = list(all_stock_ids)
    log_message(f"    [-] Restricted to shared universe: {len(funds):,} funds, {len(stocks):,} stocks")

    if not funds or not stocks:
        log_message("    [!] Error: Empty universe. Returning zeros.")
        return [0.0] * len(test_pairs)

    fund_to_idx  = {f: i for i, f in enumerate(funds)}
    stock_to_idx = {s: j for j, s in enumerate(stocks)}

    # ── Build sparse bi-adjacency matrix B (Funds × Stocks) ─────────────────
    # Build directly from quarterly graphs — no intermediate NetworkX object
    row_indices, col_indices = [], []
    for yq in train_quarters:
        G_q = quarterly_graphs.get(yq)
        if G_q:
            for u, v in G_q.edges():
                fi = fund_to_idx.get(u)
                si = stock_to_idx.get(v)
                if fi is not None and si is not None:
                    row_indices.append(fi)
                    col_indices.append(si)
                else:
                    fi = fund_to_idx.get(v)
                    si = stock_to_idx.get(u)
                    if fi is not None and si is not None:
                        row_indices.append(fi)
                        col_indices.append(si)

    if not row_indices:
        log_message("    [!] Error: No training edges. Returning zeros.")
        return [0.0] * len(test_pairs)

    B = csr_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
        shape=(len(funds), len(stocks))
    )
    B = (B > 0).astype(np.float32)  # Binarize (presence-only)

    # Fund degrees (number of stocks held)
    degrees = np.array(B.sum(axis=1)).flatten().astype(np.float64)

    # ── Precompute stock → holding-fund-indices map (CSC column access) ───────
    # Only for stocks that actually appear in test_pairs (saves time)
    B_csc = B.tocsc()
    test_stock_idxs = set()
    for f, s in test_pairs:
        si = stock_to_idx.get(s, -1)
        if si >= 0:
            test_stock_idxs.add(si)

    stock_fund_map = {si: B_csc[:, si].nonzero()[0] for si in test_stock_idxs}

    # ── Score extraction — sparse dot-product per test pair ──────────────────
    # For each (f, s):
    #   1. Get other_idxs = funds holding s  (from stock_fund_map)
    #   2. inter[k] = B[fi, :] · B[other_idxs[k], :]  (sparse dot)
    #   3. Jaccard[k] = inter[k] / (deg[fi] + deg[other_idxs[k]] - inter[k])
    #   4. score = mean(Jaccard)
    scores = np.zeros(len(test_pairs), dtype=np.float64)

    for i, (f, s) in enumerate(test_pairs):
        fi = fund_to_idx.get(f, -1)
        si = stock_to_idx.get(s, -1)
        if fi < 0 or si < 0:
            continue

        other_idxs = stock_fund_map.get(si, np.array([], dtype=np.int32))
        other_idxs = other_idxs[other_idxs != fi]  # Exclude self
        if len(other_idxs) == 0:
            continue

        # Sparse dot: B[fi, :] · B[other_idxs, :].T  → shape (1, K)
        inter = B[fi, :].dot(B[other_idxs, :].T).toarray().flatten()
        union = degrees[fi] + degrees[other_idxs] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            jac = np.where(union > 0, inter / union, 0.0)
        scores[i] = float(np.mean(jac))

    # ── Normalize to [0, 1] ──────────────────────────────────────────────────
    max_s = scores.max()
    if max_s > 0:
        scores = scores / max_s

    log_message(f"    [✓] Jaccard Similarity computed. "
                f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return scores.tolist()

# ============================================================================
# 5. BASELINE PIPELINE EXECUTION
# ============================================================================

def process_sliding_window_baselines(quarterly_graphs, train_window=8, test_offset=1):
    log_message("\n" + "=" * 80)
    log_message(f"STAGE 4: RUNNING JACCARD SIMILARITY MODEL OVER SLIDING WINDOWS")
    log_message(f"Settings: Train Window = {train_window} quarters | Test Offset = {test_offset} quarter")
    log_message("=" * 80)

    chrono_quarters = get_chronological_quarters(quarterly_graphs)
    log_message(f"[*] Total chronological quarters mapped: {len(chrono_quarters)}")
    all_results = []

    splits = list(get_sliding_window_splits(chrono_quarters, train_window=train_window, test_offset=test_offset))
    log_message(f"[*] Total sliding windows to process: {len(splits)}\n")

    for window_idx, (train_quarters, test_quarter) in enumerate(splits):
        test_year, test_quarter_num = test_quarter
        train_label = f"{train_quarters[0][0]}Q{train_quarters[0][1]} to {train_quarters[-1][0]}Q{train_quarters[-1][1]}"

        log_message(f"\n{'─' * 80}")
        log_message(f"WINDOW {window_idx + 1} / {len(splits)} | TEST QUARTER: {test_year}Q{test_quarter_num}")
        log_message(f"TRAIN WINDOW: {train_label}")
        log_message(f"{'─' * 80}")

        log_message("  [*] Fetching test graph from memory...")
        G_test = quarterly_graphs.get(test_quarter)
        if G_test is None or G_test.number_of_edges() == 0:
            log_message("  [!] WARNING: Test graph is empty. Skipping.")
            continue

        last_train_q = train_quarters[-1]
        G_last_train = quarterly_graphs.get(last_train_q)
        if G_last_train is None:
            log_message("  [!] WARNING: Last train graph not found. Skipping.")
            continue

        # Step 1: Generate test set (shared universe, no leakage)
        log_message("  [*] Step 1: Creating Evaluation Dataset...")
        result = generate_test_samples(G_test, G_last_train, num_negatives_ratio=1.0,
                                       seed=window_idx)
        if len(result) == 4:
            test_pairs, y_true, shared_funds, shared_stocks = result
        else:
            test_pairs, y_true = result
            shared_funds, shared_stocks = set(), set()

        if not test_pairs:
            log_message("  [!] WARNING: Could not generate valid test pairs. Skipping.")
            continue

        log_message(f"    [✓] Total test samples created: {len(test_pairs):,}")

        # Step 2: Jaccard scoring (training data only)
        log_message("\n  [*] Step 2: Executing Jaccard Similarity Link Prediction...")
        y_prob_js = jaccard_similarity_score(
            train_quarters, test_pairs, quarterly_graphs,
            shared_funds=shared_funds, shared_stocks=shared_stocks
        )

        # Step 3: Full metrics suite
        log_message("\n  [*] Step 3: Computing Full Metrics Suite...")
        metrics = evaluate_all_metrics(y_true, y_prob_js, test_pairs, "Jaccard Similarity")

        log_message("\n" + "." * 60)
        log_message(f"  [METRICS SUMMARY FOR WINDOW {window_idx + 1}]")
        log_message("  JACCARD SIMILARITY MODEL:")
        log_message(f"  -> AUC-ROC:              {metrics['test_auc']:.4f}")
        log_message(f"  -> Avg Precision:        {metrics['test_avg_precision']:.4f}")
        log_message(f"  -> Opt Threshold:        {metrics['opt_threshold']:.4f}")
        log_message(f"  -> Precision@0.5:        {metrics['test_precision_05']:.4f}")
        log_message(f"  -> Recall@0.5:           {metrics['test_recall_05']:.4f}")
        log_message(f"  -> F1@0.5:               {metrics['test_f1_05']:.4f}")
        log_message(f"  -> Precision@opt:        {metrics['test_precision_opt']:.4f}")
        log_message(f"  -> Recall@opt:           {metrics['test_recall_opt']:.4f}")
        log_message(f"  -> F1@opt:               {metrics['test_f1_opt']:.4f}")
        log_message(f"  -> Hit@5/10/20/50:       {metrics['hit5']:.4f} / {metrics['hit10']:.4f} / {metrics['hit20']:.4f} / {metrics['hit50']:.4f}")
        log_message(f"  -> NDCG@5/10/20/50:      {metrics['ndcg5']:.4f} / {metrics['ndcg10']:.4f} / {metrics['ndcg20']:.4f} / {metrics['ndcg50']:.4f}")
        log_message(f"  -> Rank-Return Spearman: {metrics['rank_return_spearman']:.4f}")
        log_message("." * 60)

        all_results.append({
            'window':               window_idx + 1,
            'test_quarter':         f"{test_year}Q{test_quarter_num}",
            'final_train_loss':     metrics['final_train_loss'],
            'final_val_loss':       metrics['final_val_loss'],
            'best_val_loss':        metrics['best_val_loss'],
            'test_auc':             metrics['test_auc'],
            'test_avg_precision':   metrics['test_avg_precision'],
            'opt_threshold':        metrics['opt_threshold'],
            'test_precision_05':    metrics['test_precision_05'],
            'test_recall_05':       metrics['test_recall_05'],
            'test_f1_05':           metrics['test_f1_05'],
            'test_precision_opt':   metrics['test_precision_opt'],
            'test_recall_opt':      metrics['test_recall_opt'],
            'test_f1_opt':          metrics['test_f1_opt'],
            'hit5':                 metrics['hit5'],
            'hit10':                metrics['hit10'],
            'hit20':                metrics['hit20'],
            'hit50':                metrics['hit50'],
            'ndcg5':                metrics['ndcg5'],
            'ndcg10':               metrics['ndcg10'],
            'ndcg20':               metrics['ndcg20'],
            'ndcg50':               metrics['ndcg50'],
            'rank_return_spearman': metrics['rank_return_spearman'],
        })

        log_message(f"\n  [*] Step 4: Window {window_idx + 1} memory cleanup...")
        del test_pairs, y_true, y_prob_js, metrics
        gc.collect()
        log_message("  [✓] Cleanup complete.")

    return pd.DataFrame(all_results)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    log_message(">>> INITIATING JACCARD SIMILARITY BASELINE EXPERIMENT <<<")
    log_message(f"Timestamp: {datetime.now()}")

    data = load_data()

    if data is not None:
        quarterly_graphs = build_quarterly_graphs(data)

        baseline_results_df = process_sliding_window_baselines(
            quarterly_graphs, train_window=8, test_offset=1
        )

        if not baseline_results_df.empty:
            log_message("\n\n" + "=" * 80)
            log_message("FINAL AGGREGATE EVALUATION SUMMARY - JACCARD SIMILARITY")
            log_message("=" * 80)

            display_cols = ['window', 'test_quarter', 'test_auc', 'test_avg_precision',
                            'test_f1_05', 'test_f1_opt', 'hit10', 'ndcg10']
            log_message("\nTabular Overview (key metrics per window):")
            log_message(baseline_results_df[display_cols].to_string(index=False))

            df = baseline_results_df
            log_message("\n\n*** AGGREGATE AVERAGES ACROSS ALL WINDOWS ***")
            log_message("JACCARD SIMILARITY BASELINE:")
            log_message(f"  AUC-ROC:              {df['test_auc'].mean():.4f}  (±{df['test_auc'].std():.4f})")
            log_message(f"  Avg Precision:        {df['test_avg_precision'].mean():.4f}  (±{df['test_avg_precision'].std():.4f})")
            log_message(f"  Opt Threshold:        {df['opt_threshold'].mean():.4f}  (±{df['opt_threshold'].std():.4f})")
            log_message(f"  ── Threshold = 0.5 ──────────────────────────────────")
            log_message(f"  Precision@0.5:        {df['test_precision_05'].mean():.4f}  (±{df['test_precision_05'].std():.4f})")
            log_message(f"  Recall@0.5:           {df['test_recall_05'].mean():.4f}  (±{df['test_recall_05'].std():.4f})")
            log_message(f"  F1@0.5:               {df['test_f1_05'].mean():.4f}  (±{df['test_f1_05'].std():.4f})")
            log_message(f"  ── Optimal Threshold ────────────────────────────────")
            log_message(f"  Precision@opt:        {df['test_precision_opt'].mean():.4f}  (±{df['test_precision_opt'].std():.4f})")
            log_message(f"  Recall@opt:           {df['test_recall_opt'].mean():.4f}  (±{df['test_recall_opt'].std():.4f})")
            log_message(f"  F1@opt:               {df['test_f1_opt'].mean():.4f}  (±{df['test_f1_opt'].std():.4f})")
            log_message(f"  ── Ranking ──────────────────────────────────────────")
            log_message(f"  Hit@5:                {df['hit5'].mean():.4f}  (±{df['hit5'].std():.4f})")
            log_message(f"  Hit@10:               {df['hit10'].mean():.4f}  (±{df['hit10'].std():.4f})")
            log_message(f"  Hit@20:               {df['hit20'].mean():.4f}  (±{df['hit20'].std():.4f})")
            log_message(f"  Hit@50:               {df['hit50'].mean():.4f}  (±{df['hit50'].std():.4f})")
            log_message(f"  NDCG@5:               {df['ndcg5'].mean():.4f}  (±{df['ndcg5'].std():.4f})")
            log_message(f"  NDCG@10:              {df['ndcg10'].mean():.4f}  (±{df['ndcg10'].std():.4f})")
            log_message(f"  NDCG@20:              {df['ndcg20'].mean():.4f}  (±{df['ndcg20'].std():.4f})")
            log_message(f"  NDCG@50:              {df['ndcg50'].mean():.4f}  (±{df['ndcg50'].std():.4f})")
            log_message(f"  ── Correlation ──────────────────────────────────────")
            log_message(f"  Rank-Return Spearman: {df['rank_return_spearman'].mean():.4f}  (±{df['rank_return_spearman'].std():.4f})")
            log_message("=" * 52)

            # Save results
            output_csv = os.path.join(RESULTS_DIR, f'jaccardsimilarity_scores_{timestamp}.csv')
            baseline_results_df.to_csv(output_csv, index=False)
            log_message(f"\n[✓] Detailed results saved to: {output_csv}")
            log_message("\n>>> EXPERIMENT COMPLETED SUCCESSFULLY <<<")
