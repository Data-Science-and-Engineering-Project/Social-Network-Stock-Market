"""
Stock Market Social Network - JACCARD SIMILARITY Link Prediction
Lightweight script for weak servers with EXTENSIVE LOGGING.
Jaccard Similarity: Ratio of shared neighbors to total neighbors.
Score(A,B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|
"""

import os
import re
import gc
import random
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
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
        funds = group['CIK'].unique()
        stocks = group['CUSIP'].unique()
        
        G_bip = nx.Graph()
        G_bip.add_nodes_from(funds, bipartite=0, node_type='fund')
        G_bip.add_nodes_from(stocks, bipartite=1, node_type='stock')
        
        edges = [(row.CIK, row.CUSIP) for row in group.itertuples(index=False)]
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
    """Generate splits."""
    n = len(chronological_quarters)
    for i in range(n - train_window - test_offset + 1):
        train_quarters = chronological_quarters[i : i + train_window]
        test_quarter = chronological_quarters[i + train_window + test_offset - 1]
        yield train_quarters, test_quarter

def generate_test_samples(G_test, G_last_train, num_negatives_ratio=1.0):
    """Generate balanced test set with positives and negatives."""
    log_message("    [-] Extracting positive holding edges from test graph...")
    
    funds_train = [n for n, d in G_last_train.nodes(data=True) if d.get('node_type') == 'fund']
    stocks_train = [n for n, d in G_last_train.nodes(data=True) if d.get('node_type') == 'stock']
    
    funds_train_set = set(funds_train)
    stocks_train_set = set(stocks_train)

    pos_edges = [(u, v) for u, v in G_test.edges() 
                if u in funds_train_set and v in stocks_train_set]
    
    if not funds_train or not stocks_train:
        log_message("    [!] Error: Empty train fund or stock sets.")
        return [], []

    log_message(f"    [-] Positive edges found: {len(pos_edges):,}")
    log_message(f"    [-] Beginning negative sampling (Ratio 1:{num_negatives_ratio})...")
    
    neg_edges = set()
    num_neg_target = int(len(pos_edges) * num_negatives_ratio)
    attempts = 0
    max_attempts = num_neg_target * 5
    
    while len(neg_edges) < num_neg_target and attempts < max_attempts:
        f = random.choice(funds_train)
        s = random.choice(stocks_train)
        
        if not G_test.has_edge(f, s):
            neg_edges.add((f, s))
        attempts += 1
        
        if len(neg_edges) > 0 and len(neg_edges) % max(1, (num_neg_target // 10)) == 0 and len(neg_edges) != num_neg_target:
            log_message(f"      -> Sampled {len(neg_edges):,} / {num_neg_target:,} negative edges...")
            
    log_message(f"    [✓] Generated {len(neg_edges):,} negative samples.")
    test_pairs = pos_edges + list(neg_edges)
    y_true = [1]*len(pos_edges) + [0]*len(neg_edges)
    
    return test_pairs, y_true

def evaluate_predictions(y_true, y_prob, y_pred, model_name):
    """Calculate evaluation metrics."""
    log_message(f"    [-] Calculating evaluation metrics for {model_name}...")
    
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
# 4. JACCARD SIMILARITY LINK PREDICTION
# ============================================================================

def jaccard_similarity_score(G, train_quarters, test_pairs, quarterly_graphs):
    log_message("    [-] Fast Vectorized Bipartite Jaccard Score...")
    
    # 1. בניית הגרף המאוחד לחלון האימון
    G_train_agg = nx.Graph()
    for yq in train_quarters:
        G_q = quarterly_graphs.get(yq)
        if G_q:
            # קודם כל נוסיף את הצמתים יחד עם התכונות (node_type)
            G_train_agg.add_nodes_from(G_q.nodes(data=True))
            # לאחר מכן נוסיף את הקשתות
            G_train_agg.add_edges_from(G_q.edges())
            
    # בידוד רשימת הקרנות והמניות שקיימות בגרף האימון
    all_nodes = G_train_agg.nodes(data=True)
    funds = [n for n, d in all_nodes if G_train_agg.nodes[n].get('node_type') == 'fund' or str(n).startswith('CIK')] # התאמה לסוגי הצמתים שלך
    
    # 2. בניית מטריצת שכנות (Adjacency Matrix) - קרנות מול מניות
    # NetworkX יודע להוציא את זה ברגע כמטריצה דלילה (Sparse Matrix) מהירה ויעילה בזיכרון
    try:
        from networkx.algorithms import bipartite
        # אם הגרף מסומן כדו-צדדי בצורה מלאה
        B = bipartite.biadjacency_matrix(G_train_agg, row_order=funds)
        # B היא מטריצה שבה שורה = קרן, עמודה = מניה. ערך 1 אומר שיש החזקה.
        
        # 3. חישוב חיתוך ואיחוד בין כל הקרנות בבת אחת בעזרת מכפלת מטריצות!
        # מספר המניות המשותפות בין כל זוג קרנות
        intersection_matrix = B.dot(B.T).toarray() 
        
        # סך המניות של כל קרן (הדרגה שלה)
        degrees = np.array(B.sum(axis=1)).flatten()
        
        # איחוד: |A| + |B| - |A ∩ B|
        union_matrix = degrees[:, None] + degrees[None, :] - intersection_matrix
        
        # מטריצת הג'אקארד הסופית בין קרנות לקרנות
        with np.errstate(divide='ignore', invalid='ignore'):
            fund_jaccard = np.where(union_matrix > 0, intersection_matrix / union_matrix, 0)
            
        # מיפוי מהיר מקרן לאינדקס שלה במטריצה
        fund_to_idx = {fund: idx for idx, fund in enumerate(funds)}
        
        # 4. חילוץ הציונים עבור ה-test_pairs
        scores = []
        for f, s in test_pairs:
            if f in fund_to_idx and s in G_train_agg:
                # הקרנות האחרות שמחזיקות במניה s
                other_funds = list(G_train_agg.neighbors(s))
                other_idxs = [fund_to_idx[of] for of in other_funds if of in fund_to_idx and of != f]
                
                if other_idxs:
                    idx_f = fund_to_idx[f]
                    # ממוצע הדמיון של קרן f עם כל הקרנות שמחזיקות במניה s
                    scores.append(np.mean(fund_jaccard[idx_f, other_idxs]))
                else:
                    scores.append(0)
            else:
                scores.append(0)
                
    except Exception as e:
        log_message(f"    [!] Matrix acceleration failed: {str(e)}. Falling back to safe fast loops.")
        # פתרון גיבוי מהיר מבוסס מילונים (אם המטריצה הבי-פארטיטית נכשלת בבנייה)
        neighbors = {n: set(G_train_agg.neighbors(n)) for n in G_train_agg.nodes()}
        scores = []
        for f, s in test_pairs:
            if f in neighbors and s in neighbors:
                funds_holding_s = neighbors[s]
                if not funds_holding_s:
                    scores.append(0)
                    continue
                
                jaccard_sum = 0
                set_f = neighbors[f]
                count = 0
                for other_f in funds_holding_s:
                    if other_f != f and other_f in neighbors:
                        set_other = neighbors[other_f]
                        inter = len(set_f & set_other)
                        uni = len(set_f | set_other)
                        jaccard_sum += (inter / uni) if uni > 0 else 0
                        count += 1
                scores.append(jaccard_sum / count if count > 0 else 0)
            else:
                scores.append(0)

    # 5. קביעת ה-Threshold
    threshold = np.median(scores) if scores else 0
    if threshold == 0 and np.max(scores) > 0:
        threshold = np.mean(scores)
        
    y_pred = [1 if s >= threshold else 0 for s in scores]
    return scores, y_pred
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

        # Generate Test Set
        log_message("  [*] Step 1: Creating Evaluation Dataset...")
        test_pairs, y_true = generate_test_samples(G_test, G_last_train, num_negatives_ratio=1.0)
        
        if not test_pairs:
            log_message("  [!] WARNING: Could not generate valid test pairs. Skipping.")
            continue
            
        log_message(f"    [✓] Total test samples created: {len(test_pairs):,}")
            
        # Jaccard Similarity
        log_message("\n  [*] Step 2: Executing Jaccard Similarity Link Prediction...")
        y_prob_js, y_pred_js = jaccard_similarity_score(G_test, train_quarters, test_pairs, quarterly_graphs)
        res_js = evaluate_predictions(y_true, y_prob_js, y_pred_js, "Jaccard Similarity")
        
        log_message("\n" + "." * 60)
        log_message(f"  [METRICS SUMMARY FOR WINDOW {window_idx + 1}]")
        log_message("  JACCARD SIMILARITY MODEL:")
        log_message(f"  -> AUC-ROC:  {res_js['AUC']:.4f}")
        log_message(f"  -> Precision:{res_js['Precision']:.4f}")
        log_message(f"  -> Recall:   {res_js['Recall']:.4f}")
        log_message(f"  -> F1-Score: {res_js['F1']:.4f}")
        log_message("." * 60)
        
        all_results.append({
            'window': window_idx + 1,
            'test_quarter': f"{test_year}Q{test_quarter_num}",
            'js_auc': res_js['AUC'], 
            'js_precision': res_js['Precision'], 
            'js_recall': res_js['Recall'], 
            'js_f1': res_js['F1']
        })
        
        log_message(f"\n  [*] Step 3: Window {window_idx + 1} memory cleanup...")
        del test_pairs, y_true, y_prob_js, y_pred_js
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
        
        baseline_results_df = process_sliding_window_baselines(quarterly_graphs, train_window=8, test_offset=1)
        
        if not baseline_results_df.empty:
            log_message("\n\n" + "=" * 80)
            log_message("FINAL AGGREGATE EVALUATION SUMMARY - JACCARD SIMILARITY")
            log_message("=" * 80)
            
            display_cols = ['window', 'test_quarter', 'js_auc', 'js_f1']
            log_message("\nTabular Overview (AUC & F1):")
            log_message(baseline_results_df[display_cols].to_string(index=False))
            
            log_message("\n\n*** AGGREGATE AVERAGES ACROSS ALL WINDOWS ***")
            log_message("JACCARD SIMILARITY BASELINE:")
            log_message(f"  Average AUC:       {baseline_results_df['js_auc'].mean():.4f} (±{baseline_results_df['js_auc'].std():.4f})")
            log_message(f"  Average Precision: {baseline_results_df['js_precision'].mean():.4f} (±{baseline_results_df['js_precision'].std():.4f})")
            log_message(f"  Average Recall:    {baseline_results_df['js_recall'].mean():.4f} (±{baseline_results_df['js_recall'].std():.4f})")
            log_message(f"  Average F1-Score:  {baseline_results_df['js_f1'].mean():.4f} (±{baseline_results_df['js_f1'].std():.4f})")
            log_message("*" * 45)
            
            # Save results
            output_csv = os.path.join(RESULTS_DIR, f'jaccardsimilarity_scores_{timestamp}.csv')
            baseline_results_df.to_csv(output_csv, index=False)
            log_message(f"\n[✓] Detailed results saved to: {output_csv}")
            log_message("\n>>> EXPERIMENT COMPLETED SUCCESSFULLY <<<")
