"""
Stock Market Social Network - BASELINE Temporal Link Prediction
Lightweight script for weak servers with EXTENSIVE LOGGING.
Runs Persistence and Popularity baselines over sliding windows across ALL years.
Calculates and prints AUC, Precision, Recall, and F1.
"""

"""
This project presents an efficient temporal link prediction system designed to forecast institutional investment behavior using SEC 13F filings. Rather than relying on computationally heavy deep learning architectures, the system establishes a robust baseline using lightweight heuristic models—specifically Persistence (which assumes funds maintain their previous quarter's holdings) and Global Popularity (which assumes funds will acquire the market's most heavily held stocks). By evaluating these baselines across rolling temporal windows from 2013 to 2025, the system ensures strict causality with no future data leakage. The resulting metrics, including AUC-ROC, Precision, Recall, and F1-score, provide a rigorous performance floor that any advanced Graph Neural Network (GNN) must surpass to justify its computational complexity. Furthermore, the pipeline is highly optimized with aggressive memory management techniques, allowing it to process millions of historical records efficiently even on constrained server environments.
"""

import os
import re
import gc
import random
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA SETUP AND LOADING (Adapted from your original pipeline)
# ============================================================================

def load_data():
    """Load reference data and quarterly holdings across ALL years."""
    personal_dir = os.path.expanduser('~')
    root = os.path.join(personal_dir, 'Social-Network-Stock-Market/SocialNetwork/parquet_files')
    output_dir = os.path.join(root, 'generated_combined_parquet')

    print("\n" + "=" * 80)
    print("STAGE 1: SYSTEM & PATH INITIALIZATION")
    print("=" * 80)
    print(f"[*] Base data directory set to: {root}")
    print(f"[*] Target parquet directory set to: {output_dir}")

    print("\n" + "=" * 80)
    print("STAGE 2: LOADING HOLDINGS DATA (ALL YEARS)")
    print("=" * 80)

    combined_files = sorted([f for f in os.listdir(output_dir) 
                            if f.startswith('holdings_filtered_new_period_start_') and f.endswith('.parquet')])

    if not combined_files:
        print("[!] ERROR: No processed files found in output_dir. Aborting.")
        return None
    
    print(f"[*] Found {len(combined_files)} potential parquet files.")
    
    all_dfs = []
    for file in combined_files:
        print(f"  [-] Reading file: {file}...")
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
                print(f"  [!] Warning: Could not infer date for {file}, skipping.")
                continue

        year = df_temp['YEAR'].iloc[0]
        quarter_str = df_temp['QUARTER'].iloc[0]
        print(f"  [✓] Success: Loaded {len(df_temp):,} records assigned to {year} {quarter_str}")
        all_dfs.append(df_temp)
    
    print("\n[*] Concatenating all quarterly dataframes...")
    data = pd.concat(all_dfs, ignore_index=True)
    
    print("[*] Standardizing column names to uppercase...")
    data.columns = [c.upper() for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.reset_index(drop=True)
    
    if 'PERIOD_START' in data.columns:
        data = data.rename(columns={'PERIOD_START': 'PERIOD_DATE'})
    
    data['PERIOD_DATE'] = pd.to_datetime(data['PERIOD_DATE'])
    
    if 'VALUE' not in data.columns and 'SSHPRNAMT' in data.columns:
        print("[*] VALUE column missing. Imputing with SSHPRNAMT data.")
        data['VALUE'] = data['SSHPRNAMT']
    
    print(f"\n[✓] STAGE 2 COMPLETE. Total combined records loaded: {len(data):,}")
    return data

# ============================================================================
# 2. QUARTERLY GRAPH CONSTRUCTION
# ============================================================================

def build_quarterly_graphs(data):
    """Build separate bipartite graphs for each quarter."""
    print("\n" + "=" * 80)
    print("STAGE 3: BUILDING QUARTERLY GRAPHS")
    print("=" * 80)
    
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
        
        print(f"  [-] Building graph for {year} Q{quarter}...")
        funds = group['CIK'].unique()
        stocks = group['CUSIP'].unique()
        
        G_bip = nx.Graph()
        G_bip.add_nodes_from(funds, bipartite=0, node_type='fund')
        G_bip.add_nodes_from(stocks, bipartite=1, node_type='stock')
        
        edges = [(row.CIK, row.CUSIP) for row in group.itertuples(index=False)]
        G_bip.add_edges_from(edges)
        
        quarterly_graphs[(year, quarter)] = G_bip
        print(f"  [✓] Graph constructed: {len(funds):,} funds (nodes), {len(stocks):,} stocks (nodes), {G_bip.number_of_edges():,} holding relationships (edges).")
    
    print(f"\n[✓] STAGE 3 COMPLETE. Generated {len(quarterly_graphs)} quarterly graphs.")
    return quarterly_graphs

# ============================================================================
# 3. SLIDING WINDOW & SAMPLING UTILITIES
# ============================================================================

def get_chronological_quarters(quarterly_graphs):
    return sorted(quarterly_graphs.keys())

def get_sliding_window_splits(chronological_quarters, train_window=8, test_offset=1):
    """Generate splits. Default is 8 quarters (2 years) training, 1 quarter test."""
    n = len(chronological_quarters)
    for i in range(n - train_window - test_offset + 1):
        train_quarters = chronological_quarters[i : i + train_window]
        test_quarter = chronological_quarters[i + train_window + test_offset - 1]
        yield train_quarters, test_quarter

def generate_test_samples(G_test, G_last_train, num_negatives_ratio=1.0):
    """
    Lightweight negative sampling to create a balanced test set.
    FIXED: Samples negative nodes strictly from the training graph to avoid data leakage.
    """
    print("    [-] Extracting positive holding edges from test graph...")
    
    # FIX: Extract available funds and stocks from the LAST TRAIN GRAPH FIRST, not the test graph!
    funds_train = [n for n, d in G_last_train.nodes(data=True) if d.get('node_type') == 'fund']
    stocks_train = [n for n, d in G_last_train.nodes(data=True) if d.get('node_type') == 'stock']
    
    funds_train_set = set(funds_train)
    stocks_train_set = set(stocks_train)

    pos_edges = [(u, v) for u, v in G_test.edges() 
                if u in funds_train_set and v in stocks_train_set]
    
    if not funds_train or not stocks_train:
        print("    [!] Error: Empty train fund or stock sets. Cannot sample negatives.")
        return [], []

    print(f"    [-] Positive edges found: {len(pos_edges):,}")
    print(f"    [-] Beginning negative sampling process (Ratio 1:{num_negatives_ratio})...")
    
    pos_set = set(pos_edges)
    neg_edges = set()
    num_neg_target = int(len(pos_edges) * num_negatives_ratio)
    
    attempts = 0
    max_attempts = num_neg_target * 5
    
    while len(neg_edges) < num_neg_target and attempts < max_attempts:
        # FIX: Sample strictly from the training universe
        f = random.choice(funds_train)
        s = random.choice(stocks_train)
        
        # Ensure the sampled edge is not actually a true positive in the test set
        if not G_test.has_edge(f, s):
            neg_edges.add((f, s))
        attempts += 1
        
        # Logging progress every 10%
        if len(neg_edges) > 0 and len(neg_edges) % max(1, (num_neg_target // 10)) == 0 and len(neg_edges) != num_neg_target:
            print(f"      -> Sampled {len(neg_edges):,} / {num_neg_target:,} negative edges...")
            
    print(f"    [✓] Generated {len(neg_edges):,} negative samples.")
    test_pairs = pos_edges + list(neg_edges)
    y_true = [1]*len(pos_edges) + [0]*len(neg_edges)
    
    return test_pairs, y_true

def evaluate_predictions(y_true, y_prob, y_pred, model_name):
    """Calculates evaluation metrics and handles cases where predictions are flat."""
    print(f"    [-] Calculating evaluation metrics for {model_name}...")
    
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

# ============================================================================
# 4. BASELINE PIPELINE EXECUTION
# ============================================================================

def process_sliding_window_baselines(quarterly_graphs, train_window=8, test_offset=1):
    print("\n" + "=" * 80)
    print(f"STAGE 4: RUNNING BASELINE MODELS OVER SLIDING WINDOWS")
    print(f"Settings: Train Window = {train_window} quarters | Test Offset = {test_offset} quarter")
    print("=" * 80)
    
    chrono_quarters = get_chronological_quarters(quarterly_graphs)
    print(f"[*] Total chronological quarters mapped: {len(chrono_quarters)}")
    all_results = []
    
    splits = list(get_sliding_window_splits(chrono_quarters, train_window=train_window, test_offset=test_offset))
    print(f"[*] Total sliding windows to process: {len(splits)}\n")
    
    for window_idx, (train_quarters, test_quarter) in enumerate(splits):
        test_year, test_quarter_num = test_quarter
        train_label = f"{train_quarters[0][0]}Q{train_quarters[0][1]} to {train_quarters[-1][0]}Q{train_quarters[-1][1]}"
        
        print(f"\n{'─' * 80}")
        print(f"WINDOW {window_idx + 1} / {len(splits)} | TEST QUARTER: {test_year}Q{test_quarter_num}")
        print(f"TRAIN WINDOW: {train_label}")
        print(f"{'─' * 80}")
        
        print("  [*] Fetching test graph from memory...")
        G_test = quarterly_graphs.get(test_quarter)
        if G_test is None or G_test.number_of_edges() == 0:
            print("  [!] WARNING: Test graph is empty or not found. Skipping to next window.")
            continue
            
        # FIX: Fetch the last train graph BEFORE generating the test set
        last_train_q = train_quarters[-1]
        G_last_train = quarterly_graphs.get(last_train_q)
        
        if G_last_train is None:
            print("  [!] WARNING: Last train graph not found. Cannot generate negatives safely. Skipping.")
            continue

        # 1. Generate Test Set
        print("  [*] Step 1: Creating Evaluation Dataset...")
        # FIX: Pass G_last_train to the sampling function
        test_pairs, y_true = generate_test_samples(G_test, G_last_train, num_negatives_ratio=1.0)
        
        if not test_pairs:
            print("  [!] WARNING: Could not generate valid test pairs. Skipping.")
            continue
            
        print(f"    [✓] Total test samples created: {len(test_pairs):,} (50% positive, 50% negative)")
            
        # 2. Persistence Logic
        print("\n  [*] Step 2: Executing Persistence Baseline (Copy-Paste from previous quarter)...")
        print(f"    [-] Reference quarter for Persistence: {last_train_q[0]}Q{last_train_q[1]}")
        
        last_train_edges = set(G_last_train.edges())
        
        print("    [-] Making predictions...")
        y_pred_pers = [1 if (f, s) in last_train_edges or (s, f) in last_train_edges else 0 for f, s in test_pairs]
        res_pers = evaluate_predictions(y_true, y_pred_pers, y_pred_pers, "Persistence")
        
        # 3. Popularity Logic
        print("\n  [*] Step 3: Executing Global Popularity Baseline (Top 10% most held stocks)...")
        stock_degrees = {}
        
        print("    [-] Aggregating stock popularity across entire training window...")
        for yq in train_quarters:
            G_q = quarterly_graphs.get(yq)
            if not G_q: continue
            for n, d in G_q.nodes(data=True):
                if d.get('node_type') == 'stock':
                    stock_degrees[n] = stock_degrees.get(n, 0) + G_q.degree(n)
                    
        max_deg = max(stock_degrees.values()) if stock_degrees else 1
        threshold_deg = np.percentile(list(stock_degrees.values()), 90) if stock_degrees else 0 # Top 10%
        print(f"    [-] Max stock degree found: {max_deg}. Threshold for Top 10%: {threshold_deg:.2f} connections")
        
        print("    [-] Making predictions...")
        y_prob_pop = [stock_degrees.get(s, 0) / max_deg for _, s in test_pairs]
        y_pred_pop = [1 if stock_degrees.get(s, 0) >= threshold_deg else 0 for _, s in test_pairs]
        res_pop = evaluate_predictions(y_true, y_prob_pop, y_pred_pop, "Popularity")
        
        print("\n" + "." * 60)
        print(f"[METRICS SUMMARY FOR WINDOW {window_idx + 1}]")
        print("  PERSISTENCE MODEL:")
        print(f"  -> AUC-ROC:  {res_pers['AUC']:.4f}")
        print(f"  -> Precision: {res_pers['Precision']:.4f}")
        print(f"  -> Recall:   {res_pers['Recall']:.4f}")
        print(f"  -> F1-Score: {res_pers['F1']:.4f}")

        print("\n  GLOBAL POPULARITY MODEL:")
        print(f"  -> AUC-ROC:  {res_pop['AUC']:.4f}")
        print(f"  -> Precision: {res_pop['Precision']:.4f}")
        print(f"  -> Recall:   {res_pop['Recall']:.4f}")
        print(f"  -> F1-Score: {res_pop['F1']:.4f}")
        print("." * 60)
        
        all_results.append({
            'window': window_idx + 1,
            'test_quarter': f"{test_year}Q{test_quarter_num}",
            'pers_auc': res_pers['AUC'], 'pers_precision': res_pers['Precision'], 'pers_recall': res_pers['Recall'], 'pers_f1': res_pers['F1'],
            'pop_auc': res_pop['AUC'], 'pop_precision': res_pop['Precision'], 'pop_recall': res_pop['Recall'], 'pop_f1': res_pop['F1']
        })
        
        print(f"\n  [*] Step 4: Window {window_idx + 1} memory cleanup. Releasing variables...")
        del test_pairs, y_true, y_pred_pers, y_prob_pop, y_pred_pop, stock_degrees
        gc.collect()
        print("  [✓] Cleanup complete. Ready for next window.")
        
    return pd.DataFrame(all_results)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(">>> INITIATING BASELINE EXPERIMENT <<<")
    
    # Load all 86M rows
    data = load_data()
    
    if data is not None:
        quarterly_graphs = build_quarterly_graphs(data)
        
        baseline_results_df = process_sliding_window_baselines(quarterly_graphs, train_window=8, test_offset=1)
        
        if not baseline_results_df.empty:
            print("\n\n" + "=" * 80)
            print("FINAL AGGREGATE EVALUATION SUMMARY (ALL YEARS)")
            print("=" * 80)
            
            # Print tabular summary showing key metrics
            display_cols = ['window', 'test_quarter', 'pers_auc', 'pers_f1', 'pop_auc', 'pop_f1']
            print("\nTabular Overview (AUC & F1):")
            print(baseline_results_df[display_cols].to_string(index=False))
            
            print("\n\n*** AGGREGATE AVERAGES ACROSS ALL WINDOWS ***")
            print("PERSISTENCE BASELINE:")
            print(f"  Average AUC:       {baseline_results_df['pers_auc'].mean():.4f} (±{baseline_results_df['pers_auc'].std():.4f})")
            print(f"  Average Precision: {baseline_results_df['pers_precision'].mean():.4f} (±{baseline_results_df['pers_precision'].std():.4f})")
            print(f"  Average Recall:    {baseline_results_df['pers_recall'].mean():.4f} (±{baseline_results_df['pers_recall'].std():.4f})")
            print(f"  Average F1-Score:  {baseline_results_df['pers_f1'].mean():.4f} (±{baseline_results_df['pers_f1'].std():.4f})")
            
            print("\nGLOBAL POPULARITY BASELINE:")
            print(f"  Average AUC:       {baseline_results_df['pop_auc'].mean():.4f} (±{baseline_results_df['pop_auc'].std():.4f})")
            print(f"  Average Precision: {baseline_results_df['pop_precision'].mean():.4f} (±{baseline_results_df['pop_precision'].std():.4f})")
            print(f"  Average Recall:    {baseline_results_df['pop_recall'].mean():.4f} (±{baseline_results_df['pop_recall'].std():.4f})")
            print(f"  Average F1-Score:  {baseline_results_df['pop_f1'].mean():.4f} (±{baseline_results_df['pop_f1'].std():.4f})")
            print("*" * 45)
            
            # Save results
            os.makedirs('results', exist_ok=True)
            output_csv = 'results/baselines_scores_detailed_report.csv'
            baseline_results_df.to_csv(output_csv, index=False)
            print(f"\n[✓] Detailed baseline results saved to: {output_csv}")
            print("\n>>> EXPERIMENT COMPLETED SUCCESSFULLY <<<")