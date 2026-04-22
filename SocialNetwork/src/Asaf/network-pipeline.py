"""
Stock Market Social Network - Temporal Link Prediction Pipeline

Build quarterly bipartite graphs of fund-stock holdings and perform temporal link prediction 
using sliding windows (2021-2024).

Key Features:
- Load holdings data 2021-2024 only
- Build separate bipartite graph per quarter
- Sliding window: train on 8 quarters (2 years), predict next quarter
- Strict temporal causality (no future leakage)
- Per-quarter evaluation metrics (AUC, Precision, Recall)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Core libraries
import pandas as pd
import numpy as np
import os
import re
import warnings
import glob
import pickle

# Graph libraries
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.centrality import degree_centrality, closeness_centrality
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.link_analysis.hits_alg import hits
import igraph as ig
import leidenalg as la

# ML libraries
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
import joblib

# Deep learning
import torch
from torch_geometric.nn import SAGEConv

warnings.filterwarnings('ignore')


# ============================================================================
# GPU SETUP
# ============================================================================

def check_cuda_compatibility():
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        test_tensor = torch.zeros(1).cuda()
        test_tensor = test_tensor + 1
        return True, "CUDA compatible"
    except Exception as e:
        return False, f"CUDA compatibility issue: {str(e)}"

cuda_compatible, cuda_message = check_cuda_compatibility()
print(f"CUDA Status: {cuda_message}")

if cuda_compatible:
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    print('Using CPU (GPU not available)')


# ============================================================================
# 1. DATA SETUP AND LOADING
# ============================================================================
# Load quarterly holdings data from processed parquet files (2021-2024 only).

def load_data():
    """Load reference data and quarterly holdings."""
    personal_dir = os.path.expanduser('~')
    root = os.path.join(personal_dir, 'Social-Network-Stock-Market/SocialNetwork/parquet_files')
    output_dir = os.path.join(root, 'generated_combined_parquet')

    print(f"Data directory: {root}")
    print(f"Output directory: {output_dir}")

    # Load reference data
    ticker_map = pd.read_parquet(f"{root}/ticker_to_cusip.parquet")
    prices = pd.read_parquet(f"{root}/ticker_prices.parquet")
    ticker_map.columns = [c.upper() for c in ticker_map.columns]
    prices.columns = [c.upper() for c in prices.columns]

    ticker_map["CUSIP"] = ticker_map["CUSIP"].astype(str) 
    prices["PERIOD_START"] = pd.to_datetime(prices["PERIOD_START"])

    print(f"✓ Ticker map: {ticker_map.shape}")
    print(f"✓ Prices: {prices.shape}")

    # Load all processed quarterly holdings files
    print("=" * 80)
    print("Loading quarterly holdings data (2021-2024)...")
    print("=" * 80)

    combined_files = sorted([f for f in os.listdir(output_dir) 
                            if f.startswith('holdings_filtered_new_period_start_') and f.endswith('.parquet')])

    if not combined_files:
        print("ERROR: No processed files found. Check output_dir path.")
        return None
    
    all_dfs = []
    for file in combined_files:
        df_temp = pd.read_parquet(os.path.join(output_dir, file))
        
        # זיהוי עמודת התאריך וחילוץ שנה ורבעון (תוקן לשמות העמודות בקבצים שלך)
        date_col = 'period_start' if 'period_start' in df_temp.columns else 'PERIOD_DATE'
        
        if date_col in df_temp.columns:
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp['YEAR'] = df_temp[date_col].dt.year
            df_temp['QUARTER'] = 'Q' + df_temp[date_col].dt.quarter.astype(str)
        else:
            # אם אין עמודת תאריך, ננסה לחלץ משם הקובץ (למשל: 2013-04-01)
            date_match = re.search(r'(\d{4})-\d{2}-\d{2}', file)
            if date_match:
                year_val = int(date_match.group(1))
                df_temp['YEAR'] = year_val
                # לוגיקה בסיסית לקביעת רבעון לפי חודש (אם קיים בשם הקובץ)
                month_match = re.search(r'\d{4}-(\d{2})-\d{2}', file)
                month_val = int(month_match.group(1)) if month_match else 1
                df_temp['QUARTER'] = f"Q{(month_val-1)//3 + 1}"
            else:
                print(f"  ⚠ Warning: Could not find date for {file}, skipping.")
                continue

        year = df_temp['YEAR'].iloc[0]
        quarter_str = df_temp['QUARTER'].iloc[0]
        print(f"  ✓ Loaded {file}: {len(df_temp):,} records ({year} {quarter_str})")
        all_dfs.append(df_temp)
    
    data = pd.concat(all_dfs, ignore_index=True)

    
    # 2. תיקון קריטי: הפיכת כל שמות העמודות לאותיות גדולות
    # זה יפתור את הבעיה ש-cik הופך ל-CIK ו-cusip ל-CUSIP
    data.columns = [c.upper() for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.reset_index(drop=True)
    
    # 3. התאמת שם עמודת התאריך אם היא נקראה period_start (עכשיו PERIOD_START)
    if 'PERIOD_START' in data.columns:
        data = data.rename(columns={'PERIOD_START': 'PERIOD_DATE'})
    
    # 4. המרה לפורמט תאריך
    data['PERIOD_DATE'] = pd.to_datetime(data['PERIOD_DATE'])
    
    # 5. יצירת עמודת VALUE במידה והיא חסרה (נשתמש בכמות המניות כמשקל)
    if 'VALUE' not in data.columns and 'SSHPRNAMT' in data.columns:
        data['VALUE'] = data['SSHPRNAMT']
    
    # 6. סינון שנים ורבעונים (התחלה מרבעון 3 של 2023)
    data['quarter_num'] = data['QUARTER'].astype(str).str.replace('Q', '').astype(int)
    data = data[((data['YEAR'] > 2023) | ((data['YEAR'] == 2023) & (data['quarter_num'] >= 3)))].copy()
    data = data.drop('quarter_num', axis=1)
    
    return data


# ============================================================================
# 2. QUARTERLY GRAPH CONSTRUCTION (2021-2024 only)
# ============================================================================
# Build separate bipartite graphs for each quarter from 2021-2024.

def build_quarterly_graphs(data):
    """
    Build separate bipartite graphs for each quarter.
    Data should already be filtered to desired time range.
    
    Args:
        data: DataFrame with columns [CIK, CUSIP, VALUE, SSHPRNAMT, PERIOD_DATE, YEAR, QUARTER]
    
    Returns:
        Dictionary: {(year, quarter): bipartite_graph}
    """
    quarterly_graphs = {}
    
    # Group by YEAR and extract quarter from QUARTER column
    for (year, quarter_str), group in data.groupby(['YEAR', 'QUARTER']):
        if isinstance(quarter_str, str):
            if '_' in quarter_str:
                quarter = int(quarter_str.split('_')[0][1])
            elif quarter_str.startswith('Q'):
                quarter = int(quarter_str[1])
            else:
                quarter = int(quarter_str)
        else:
            # אם זה כבר מספר (int64) כפי שמופיע בשגיאה
            quarter = int(quarter_str)
        
        funds = group['CIK'].unique()
        stocks = group['CUSIP'].unique()
        
        # Build bipartite graph
        G_bip = nx.Graph()
        G_bip.add_nodes_from(funds, bipartite=0, node_type='fund')
        G_bip.add_nodes_from(stocks, bipartite=1, node_type='stock')
        
        # Add edges with VALUE weight
        edges = [
            (row.CIK, row.CUSIP, {'value': row.VALUE, 'amount': row.SSHPRNAMT})
            for row in group.itertuples(index=False)
        ]
        G_bip.add_edges_from(edges)
        
        quarterly_graphs[(year, quarter)] = G_bip
        print(f"  {year} Q{quarter}: {len(funds):,} funds, {len(stocks):,} stocks, {G_bip.number_of_edges():,} edges")
    
    return quarterly_graphs


# ============================================================================
# 3. SLIDING WINDOW UTILITIES
# ============================================================================
# Implement temporal train/test splits with 8-quarter (2-year) training window.

def get_chronological_quarters(quarterly_graphs):
    """Get all quarters in chronological order."""
    return sorted(quarterly_graphs.keys())


def build_combined_training_graph(quarterly_graphs, quarters_list):
    """
    Combine multiple quarterly graphs into ONE training graph.
    This is the UNION of all edges across the training quarters.
    
    Args:
        quarterly_graphs: Dict {(year, quarter): graph}
        quarters_list: List of (year, quarter) tuples to combine
    
    Returns:
        Single bipartite graph with all edges from training quarters
    """
    if not quarters_list:
        raise ValueError("quarters_list cannot be empty")
    
    G_train = nx.Graph()
    
    # Add all nodes and edges from each quarter
    for yq in quarters_list:
        if yq not in quarterly_graphs:
            continue
        G_q = quarterly_graphs[yq]
        G_train.add_nodes_from(G_q.nodes(data=True))
        G_train.add_edges_from(G_q.edges(data=True))
    
    return G_train


def get_sliding_window_splits(chronological_quarters, train_window=3, test_offset=1):
    """
    Generate temporal train/test splits using sliding window.
    
    Args:
        chronological_quarters: Sorted list of (year, quarter) tuples
        train_window: Number of quarters for training
        test_offset: Quarters ahead to test (default: 1 = immediate next quarter)
    
    Yields:
        (train_quarters_list, test_quarter)
    """
    n = len(chronological_quarters)
    
    if n < train_window + test_offset:
        print(f"WARNING: Only {n} quarters available, need {train_window + test_offset}")
        return
    
    for i in range(n - train_window - test_offset + 1):
        train_quarters = chronological_quarters[i : i + train_window]
        test_quarter = chronological_quarters[i + train_window + test_offset - 1]
        yield train_quarters, test_quarter


# ============================================================================
# 4. GRAPH FEATURES: CENTRALITY & COMMUNITY DETECTION
# ============================================================================
# Compute topological features from training graph only (no future leakage).

def compute_fund_features(G_bip, funds):
    """
    Compute topological features for funds from bipartite graph.
    Features computed only from G_bip (no future information).
    
    Args:
        G_bip: Bipartite graph (fund-stock holdings)
        funds: List of fund CIKs
    
    Returns:
        DataFrame with features: degree, pagerank, hub, authority, closeness, community
    """
    if len(funds) == 0:
        return pd.DataFrame()
    
    # Project to fund-fund graph (shared stock holdings)
    try:
        G_fund = bipartite.weighted_projected_graph(G_bip, funds)
    except:
        G_fund = nx.Graph()
        G_fund.add_nodes_from(funds)
    
    # Centrality metrics
    degree_cent = degree_centrality(G_fund) if G_fund.number_of_nodes() > 0 else {}
    pagerank_cent = nx.pagerank(G_fund) if G_fund.number_of_nodes() > 0 else {}
    
    try:
        hubs, authorities = hits(G_fund)
    except:
        hubs = {f: 0 for f in funds}
        authorities = {f: 0 for f in funds}
    
    # Closeness on largest component
    closeness_cent = {}
    if G_fund.number_of_nodes() > 0:
        try:
            comps = list(nx.connected_components(G_fund))
            if comps:
                largest_cc = max(comps, key=len)
                closeness_cent = closeness_centrality(G_fund.subgraph(largest_cc))
        except:
            pass
    
    # Community detection (Leiden algorithm)
    communities = {}
    if G_fund.number_of_nodes() > 1:
        try:
            vertex_names = list(G_fund.nodes())
            vertex_to_idx = {v: i for i, v in enumerate(vertex_names)}
            edge_list = [(vertex_to_idx[u], vertex_to_idx[v]) for u, v in G_fund.edges()]
            
            if edge_list:
                ig_G = ig.Graph(n=len(vertex_names), edges=edge_list)
                ig_G.vs['_nx_name'] = vertex_names
                partition = la.find_partition(ig_G, la.ModularityVertexPartition)
                communities = {ig_G.vs[i]['_nx_name']: p for p, cl in enumerate(partition) for i in cl}
        except:
            communities = {f: 0 for f in funds}
    
    # Build feature dataframe
    fund_features = pd.DataFrame({
        'fund': funds,
        'degree': [degree_cent.get(f, 0) for f in funds],
        'pagerank': [pagerank_cent.get(f, 0) for f in funds],
        'hub': [hubs.get(f, 0) for f in funds],
        'authority': [authorities.get(f, 0) for f in funds],
        'closeness': [closeness_cent.get(f, 0) for f in funds],
        'community': [communities.get(f, -1) for f in funds]
    }).set_index('fund')
    
    return fund_features


# ============================================================================
# 5. GraphSAGE EMBEDDINGS
# ============================================================================
# Train GraphSAGE on training window to generate node embeddings.

class GraphSAGE(torch.nn.Module):
    """2-layer GraphSAGE model for bipartite graphs."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def train_graphsage_window(G_bip, num_epochs=30, embedding_dim=8, pretrained_model=None, fine_tune_lr=0.001):
    """
    Train GraphSAGE on bipartite graph with optional transfer learning.
    
    Args:
        G_bip: Bipartite graph
        num_epochs: Training epochs
        embedding_dim: Output embedding dimension
        pretrained_model: Optional pre-trained GraphSAGE model (for transfer learning)
        fine_tune_lr: Learning rate for fine-tuning (lower than training from scratch)
    
    Returns:
        (model, embeddings_numpy, funds, stocks)
    """
    nodes = list(G_bip.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Separate funds and stocks
    funds = [n for n in nodes if G_bip.nodes[n].get('bipartite') == 0]
    stocks = [n for n in nodes if G_bip.nodes[n].get('bipartite') == 1]
    
    num_nodes = len(nodes)
    
    # Initialize features on GPU
    x = torch.randn(num_nodes, 16, device=device)
    
    # Build edge index
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G_bip.edges()]
    if not edge_list:
        print("    WARNING: Graph has no edges")
        return None, np.zeros((num_nodes, embedding_dim)), funds, stocks
    
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    
    # Add reverse edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    # Model: use pretrained if available, otherwise create new
    if pretrained_model is not None:
        print("     → Using pretrained model (transfer learning)")
        model = pretrained_model.to(device)
        learning_rate = fine_tune_lr
    else:
        print("     → Training from scratch")
        model = GraphSAGE(16, 32, embedding_dim).to(device)
        learning_rate = 0.01
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    prev_loss = float('inf')
    patience, no_improve = 5, 0
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        
        # Link prediction loss
        pos_score = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=1).sigmoid()
        loss = -torch.log(pos_score + 1e-15).mean()
        loss.backward()
        optimizer.step()
        
        if abs(prev_loss - loss.item()) < 1e-6:
            no_improve += 1
            if no_improve >= patience:
                break
        else:
            no_improve = 0
        prev_loss = loss.item()
    
    # Extract embeddings
    model.eval()
    with torch.no_grad():
        emb = model(x, edge_index).cpu().numpy()
    
    torch.cuda.empty_cache()
    
    return model, emb, funds, stocks


# ============================================================================
# 6. LINK PREDICTION FEATURES & NEGATIVE SAMPLING
# ============================================================================
# Build training/test features for link prediction with proper negative sampling.

def create_link_prediction_features(G_bip_train, G_bip_test, embeddings_train, 
                                    fund_features, funds, stocks, fund_to_idx, stock_to_idx):
    """
    Create feature matrix for link prediction.
    
    Training edges: from G_bip_train (label=1) + hard negatives (label=0)
    Test edges: from G_bip_test (label=1) + negatives not in G_bip_train (label=0)
    
    Args:
        G_bip_train: Training bipartite graph
        G_bip_test: Test bipartite graph (for positive labels only)
        embeddings_train: Node embeddings (from GraphSAGE)
        fund_features: DataFrame with fund topological features
        funds, stocks: Lists of nodes
        fund_to_idx, stock_to_idx: Node to index mappings
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    fund_emb = embeddings_train[:len(funds)]
    stock_emb = embeddings_train[len(funds):]
    
    # ── TRAINING DATA ──
    # Positive edges from training graph
    pos_edges_train = [
        (fund_to_idx[u], stock_to_idx[v]) 
        for u, v in G_bip_train.edges() 
        if u in fund_to_idx and v in stock_to_idx
    ]
    
    # Hard negative sampling (use stock similarity)
    stock_sim = cosine_similarity(stock_emb)
    neg_edges_train = []
    
    for f_idx in range(len(funds)):
        fund_id = funds[f_idx]
        # Get connected stocks in training graph
        connected_stocks = {stock_to_idx[s] for s in G_bip_train.neighbors(fund_id) 
                           if s in stock_to_idx}
        
        if not connected_stocks:
            continue
        
        # Average similarity to connected stocks
        connected_list = list(connected_stocks)
        avg_sim = stock_sim[connected_list].mean(axis=0)
        
        # Hard negatives: high similarity but not connected
        hard_negs = np.argsort(-avg_sim)
        hard_neg_list = [
            s_idx for s_idx in hard_negs 
            if s_idx not in connected_stocks and len(neg_edges_train) < len(pos_edges_train)
        ]
        
        neg_edges_train.extend([(f_idx, s_idx) for s_idx in hard_neg_list[:20]])
    
    neg_edges_train = neg_edges_train[:len(pos_edges_train)]  # Balance classes
    
    # ── TEST DATA ──
    # Positive edges from test graph (only edges we didn't see in training)
    test_edges_train = set((fund_to_idx[u], stock_to_idx[v]) 
                          for u, v in G_bip_train.edges() 
                          if u in fund_to_idx and v in stock_to_idx)
    
    pos_edges_test = [
        (fund_to_idx[u], stock_to_idx[v]) 
        for u, v in G_bip_test.edges() 
        if u in fund_to_idx and v in stock_to_idx and (fund_to_idx[u], stock_to_idx[v]) not in test_edges_train
    ]
    
    # Test negatives: not in training OR test graphs
    all_possible = set((i, j) for i in range(len(funds)) for j in range(len(stocks)))
    test_edges_all = test_edges_train | set(pos_edges_test)
    neg_edges_test = list(all_possible - test_edges_all)
    neg_edges_test = neg_edges_test[:max(len(pos_edges_test), 1)]
    
    # Build feature vectors
    def build_features(edge_list):
        features = []
        for f_idx, s_idx in edge_list:
            fund_id = funds[f_idx]
            feat = np.concatenate([
                fund_emb[f_idx],
                stock_emb[s_idx],
                fund_features.loc[fund_id].values if fund_id in fund_features.index else np.zeros(6)
            ])
            features.append(feat)
        return np.array(features) if features else np.zeros((0, fund_emb.shape[1] + stock_emb.shape[1] + 6))
    
    X_train = np.vstack([
        build_features(pos_edges_train),
        build_features(neg_edges_train)
    ])
    y_train = np.hstack([np.ones(len(pos_edges_train)), np.zeros(len(neg_edges_train))])
    
    X_test = np.vstack([
        build_features(pos_edges_test),
        build_features(neg_edges_test)
    ])
    y_test = np.hstack([np.ones(len(pos_edges_test)), np.zeros(len(neg_edges_test))])
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# 7. TEMPORAL LINK PREDICTION: SLIDING WINDOW EVALUATION (2021-2024)
# ============================================================================
# Evaluate model per quarter with configurable training window and strict temporal causality.

def run_temporal_link_prediction(quarterly_graphs, train_window=3, test_offset=1, results_dir='results'):
    """
    Run temporal link prediction with sliding window evaluation.
    
    Args:
        quarterly_graphs: Dictionary of quarterly bipartite graphs
        train_window: Number of quarters for training window
        test_offset: Number of quarters ahead to test
        results_dir: Directory to save all results and models
    
    Returns:
        DataFrame with results per window, models_dir path
    """
    
    print(f"Sliding window configuration:")
    print(f"  Training window: {train_window} quarters")
    print(f"  Test offset: {test_offset} quarter(s) ahead")

    # Create directory structure for saved results
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, 'temporal_models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"  Results directory: {results_dir}/")
    print(f"  Models will be saved to: {models_dir}/")
    
    print("=" * 100)
    print(f"TEMPORAL LINK PREDICTION: SLIDING WINDOW EVALUATION (2021-2024)")
    print(f"Train window: {train_window} quarters | Test offset: {test_offset} quarter(s)")
    print("=" * 100)

    results_per_quarter = []
    chrono_quarters = get_chronological_quarters(quarterly_graphs)
    pretrained_graphsage = None  # Will store previous window's model

    # Run sliding window evaluation with transfer learning
    for window_idx, (train_quarters, test_quarter) in enumerate(
        get_sliding_window_splits(chrono_quarters, train_window=train_window, test_offset=test_offset)
    ):
        test_year, test_quarter_num = test_quarter
        train_label = ' → '.join([f"{y}Q{q}" for y, q in train_quarters])
        
        print(f"\n{'─' * 100}")
        print(f"WINDOW {window_idx + 1} | TEST: {test_year}Q{test_quarter_num}")
        print(f"TRAIN: {train_label}")
        print(f"{'─' * 100}")
        
        try:
            # 1. Build COMBINED training graph from all training quarters
            print("  1. Building combined training graph...")
            G_bip_train = build_combined_training_graph(quarterly_graphs, train_quarters)
            funds_train = [n for n in G_bip_train.nodes() if G_bip_train.nodes[n].get('bipartite') == 0]
            stocks_train = [n for n in G_bip_train.nodes() if G_bip_train.nodes[n].get('bipartite') == 1]
            
            print(f"     Funds: {len(funds_train):,} | Stocks: {len(stocks_train):,} | Edges: {G_bip_train.number_of_edges():,}")
            
            if G_bip_train.number_of_nodes() == 0:
                print("     WARNING: Training graph is empty, skipping...")
                continue
            
            # 2. Compute topological features
            print("  2. Computing topological features...")
            fund_features = compute_fund_features(G_bip_train, funds_train)
            print(f"     Feature shape: {fund_features.shape}")
            
            # 3. Train GraphSAGE (with transfer learning from previous window)
            print("  3. Training GraphSAGE embeddings...")
            if pretrained_graphsage is not None:
                print(f"     → Using pretrained model (transfer learning from window {window_idx})")
                graphsage_model, embeddings, funds_sage, stocks_sage = train_graphsage_window(
                    G_bip_train, num_epochs=30, embedding_dim=8, 
                    pretrained_model=pretrained_graphsage, fine_tune_lr=0.001
                )
            else:
                print("     → Training from scratch")
                graphsage_model, embeddings, funds_sage, stocks_sage = train_graphsage_window(
                    G_bip_train, num_epochs=30, embedding_dim=8, 
                    pretrained_model=None
                )
            
            print(f"     Embeddings shape: {embeddings.shape}")
            
            # 4. Get TEST quarter graph (individual quarter, not aggregated)
            print("  4. Loading test quarter graph...")
            G_bip_test = quarterly_graphs.get(test_quarter)
            if G_bip_test is None:
                print(f"     WARNING: Test quarter {test_quarter} not found, skipping...")
                continue
            
            funds_test = [n for n in G_bip_test.nodes() if G_bip_test.nodes[n].get('bipartite') == 0]
            stocks_test = [n for n in G_bip_test.nodes() if G_bip_test.nodes[n].get('bipartite') == 1]
            print(f"     Test funds: {len(funds_test):,} | Test stocks: {len(stocks_test):,} | Edges: {G_bip_test.number_of_edges():,}")
            
            # 5. Create mappings and link prediction features
            print("  5. Creating link prediction features...")
            fund_to_idx = {f: i for i, f in enumerate(funds_train)}
            stock_to_idx = {s: i for i, s in enumerate(stocks_train)}
            
            X_train, y_train, X_test, y_test = create_link_prediction_features(
                G_bip_train, G_bip_test, embeddings, fund_features,
                funds_train, stocks_train, fund_to_idx, stock_to_idx
            )
            print(f"     Train: {X_train.shape[0]:,} samples (pos: {y_train.sum():.0f}, neg: {(1-y_train).sum():.0f})")
            print(f"     Test:  {X_test.shape[0]:,} samples (pos: {y_test.sum():.0f}, neg: {(1-y_test).sum():.0f})")
            
            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                print("     WARNING: No training or test samples, skipping...")
                continue
            
            # 6. Train LightGBM
            print("  6. Training LightGBM...")
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1
            }
            
            bst = lgb.train(params, train_data_lgb, num_boost_round=100, valid_sets=[train_data_lgb])
            
            # 7. Evaluate
            print("  7. Evaluating...")
            y_pred = bst.predict(X_test)
            
            auc = roc_auc_score(y_test, y_pred)
            precision = precision_score(y_test, (y_pred > 0.5).astype(int), zero_division=0)
            recall = recall_score(y_test, (y_pred > 0.5).astype(int), zero_division=0)
            
            print(f"\n     ✓ AUC:       {auc:.4f}")
            print(f"     ✓ Precision: {precision:.4f}")
            print(f"     ✓ Recall:    {recall:.4f}")
            
            # 8. Save models to pickle
            print("  8. Saving models to pickle...")
            model_filename = f"window_{window_idx+1}_{train_label.replace(' → ', '_')}_test_{test_year}Q{test_quarter_num}.pkl"
            model_path = os.path.join(models_dir, model_filename)
            
            model_data = {
                'window': window_idx + 1,
                'graphsage_model': graphsage_model,
                'lgb_model': bst,
                'embeddings': embeddings,
                'fund_features': fund_features,
                'fund_to_idx': fund_to_idx,
                'stock_to_idx': stock_to_idx,
                'funds_train': funds_train,
                'stocks_train': stocks_train,
                'train_quarters': train_quarters,
                'test_quarter': test_quarter,
                'metrics': {
                    'auc': auc,
                    'precision': precision,
                    'recall': recall
                }
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"     ✓ Saved to: {model_path}")
            
            # Save GraphSAGE model for next window (transfer learning)
            pretrained_graphsage = graphsage_model
            
            results_per_quarter.append({
                'window': window_idx + 1,
                'test_year': test_year,
                'test_quarter': test_quarter_num,
                'train_quarters': train_label,
                'n_train_funds': len(funds_train),
                'n_train_stocks': len(stocks_train),
                'n_test_funds': len(funds_test),
                'n_test_stocks': len(stocks_test),
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'n_test_samples': X_test.shape[0],
                'model_path': model_path,
                'transfer_learned': window_idx > 0
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n\n{'=' * 100}")
    print(f"EVALUATION SUMMARY (2021-2024, {train_window}-Quarter Training Window)")
    print(f"{'=' * 100}")

    if results_per_quarter:
        results_df = pd.DataFrame(results_per_quarter)
        print(f"\nResults across {len(results_df)} windows:\n")
        display_cols = ['window', 'train_quarters', 'test_year', 'test_quarter', 'transfer_learned', 'auc', 'precision', 'recall']
        print(results_df[display_cols].to_string(index=False))
        
        print(f"\n\nAggregate Statistics:")
        print(f"  Average AUC:       {results_df['auc'].mean():.4f} (±{results_df['auc'].std():.4f})")
        print(f"  Average Precision: {results_df['precision'].mean():.4f} (±{results_df['precision'].std():.4f})")
        print(f"  Average Recall:    {results_df['recall'].mean():.4f} (±{results_df['recall'].std():.4f})")
        
        print(f"\n\nBy Year:")
        by_year = results_df.groupby('test_year')[['auc', 'precision', 'recall']].mean()
        print(by_year)
        
        # Save results to results directory
        csv_path = os.path.join(results_dir, 'temporal_link_prediction_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        print(f"✓ Models saved to {models_dir}/ directory ({len(results_df)} model files)")
        print(f"✓ Transfer learning: Windows 2+ use previous window's GraphSAGE model")
    else:
        print("No results generated.")
    
    return (results_df if results_per_quarter else None), models_dir, results_dir


# ============================================================================
# 8. NODE CONNECTION PREDICTION & INFERENCE
# ============================================================================
# Load trained models and predict connections for query nodes

class NodeConnectionPredictor:
    """
    Inference engine for predicting node connections using trained GraphSAGE + LightGBM models.
    """
    
    def __init__(self, model_path):
        """
        Load a saved model from pickle file.
        
        Args:
            model_path: Path to .pkl file containing trained models
        """
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.graphsage_model = self.model_data['graphsage_model']
        self.lgb_model = self.model_data['lgb_model']
        self.embeddings = self.model_data['embeddings']
        self.fund_features = self.model_data['fund_features']
        self.fund_to_idx = self.model_data['fund_to_idx']
        self.stock_to_idx = self.model_data['stock_to_idx']
        self.funds_train = self.model_data['funds_train']
        self.stocks_train = self.model_data['stocks_train']
        self.train_quarters = self.model_data['train_quarters']
        self.test_quarter = self.model_data['test_quarter']
        self.metrics = self.model_data['metrics']
        
        print(f"✓ Model loaded for quarters: {self.train_quarters} → {self.test_quarter}")
        print(f"  Funds: {len(self.funds_train)} | Stocks: {len(self.stocks_train)}")
        print(f"  Model metrics - AUC: {self.metrics['auc']:.4f}, Precision: {self.metrics['precision']:.4f}, Recall: {self.metrics['recall']:.4f}")
    
    def predict_connections(self, node_id, node_type='fund', top_k=10, threshold=0.5):
        """
        Predict connections for a given node (fund or stock).
        
        Args:
            node_id: CIK (fund) or CUSIP (stock) to query
            node_type: 'fund' or 'stock'
            top_k: Return top K predictions by probability
            threshold: Confidence threshold for positive predictions
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        
        results = {
            'query_node': node_id,
            'node_type': node_type,
            'found': False,
            'predictions': [],
            'high_confidence': [],
            'all_predictions': []
        }
        
        # Validate node exists in training set
        if node_type == 'fund':
            if node_id not in self.fund_to_idx:
                results['error'] = f"Fund {node_id} not found in training data"
                return results
            node_idx = self.fund_to_idx[node_id]
            results['found'] = True
            
            # Get fund embedding and features
            fund_emb = self.embeddings[node_idx]
            fund_feat = self.fund_features.loc[node_id].values if node_id in self.fund_features.index else np.zeros(6)
            
            # Score all stocks
            stock_emb = self.embeddings[len(self.funds_train):]
            scores = []
            
            for s_idx, stock_id in enumerate(self.stocks_train):
                feat = np.concatenate([fund_emb, stock_emb[s_idx], fund_feat])
                prob = self.lgb_model.predict([feat])[0]
                scores.append((stock_id, prob))
            
            # Sort by probability
            scores.sort(key=lambda x: x[1], reverse=True)
            
            results['all_predictions'] = [(stock_id, float(prob)) for stock_id, prob in scores]
            results['top_predictions'] = [(stock_id, float(prob)) for stock_id, prob in scores[:top_k]]
            results['high_confidence'] = [(stock_id, float(prob)) for stock_id, prob in scores if prob >= threshold]
            
        else:  # stock
            if node_id not in self.stock_to_idx:
                results['error'] = f"Stock {node_id} not found in training data"
                return results
            node_idx = self.stock_to_idx[node_id]
            results['found'] = True
            
            # Get stock embedding
            stock_emb = self.embeddings[len(self.funds_train) + node_idx]
            fund_emb_all = self.embeddings[:len(self.funds_train)]
            
            # Score all funds
            scores = []
            for f_idx, fund_id in enumerate(self.funds_train):
                fund_feat = self.fund_features.loc[fund_id].values if fund_id in self.fund_features.index else np.zeros(6)
                feat = np.concatenate([fund_emb_all[f_idx], stock_emb, fund_feat])
                prob = self.lgb_model.predict([feat])[0]
                scores.append((fund_id, prob))
            
            # Sort by probability
            scores.sort(key=lambda x: x[1], reverse=True)
            
            results['all_predictions'] = [(fund_id, float(prob)) for fund_id, prob in scores]
            results['top_predictions'] = [(fund_id, float(prob)) for fund_id, prob in scores[:top_k]]
            results['high_confidence'] = [(fund_id, float(prob)) for fund_id, prob in scores if prob >= threshold]
        
        return results
    
    def get_summary(self):
        """Get model summary information."""
        return {
            'train_quarters': self.train_quarters,
            'test_quarter': self.test_quarter,
            'n_funds': len(self.funds_train),
            'n_stocks': len(self.stocks_train),
            'embeddings_shape': self.embeddings.shape,
            'metrics': self.metrics
        }


def batch_predict_connections(models_dir='temporal_models', query_nodes=None, node_types=None, top_k=10):
    """
    Batch predict connections across all trained models.
    
    Args:
        models_dir: Directory containing saved .pkl model files
        query_nodes: List of node IDs to query (fund CIKs or stock CUSIPs)
        node_types: List of node types ('fund' or 'stock') corresponding to query_nodes
        top_k: Return top K predictions
    
    Returns:
        DataFrame with predictions across all models
    """
    
    if query_nodes is None:
        print("ERROR: query_nodes must be provided")
        return None
    
    if node_types is None:
        node_types = ['fund'] * len(query_nodes)
    
    # Find all model files
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
    
    if not model_files:
        print(f"ERROR: No .pkl files found in {models_dir}")
        return None
    
    print(f"Found {len(model_files)} trained models")
    print(f"Querying {len(query_nodes)} nodes across all models\n")
    
    all_results = []
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        try:
            predictor = NodeConnectionPredictor(model_path)
            
            for query_node, node_type in zip(query_nodes, node_types):
                pred_result = predictor.predict_connections(query_node, node_type=node_type, top_k=top_k)
                
                if pred_result['found']:
                    top_preds = pred_result.get('top_predictions', [])
                    high_conf = pred_result.get('high_confidence', [])
                    
                    all_results.append({
                        'model_file': model_file,
                        'query_node': query_node,
                        'node_type': node_type,
                        'top_predictions': top_preds,
                        'high_confidence_connections': high_conf,
                        'total_predictions': len(pred_result.get('all_predictions', [])),
                        'n_high_confidence': len(high_conf),
                        'model_auc': predictor.metrics['auc']
                    })
                else:
                    print(f"⚠ {query_node} not found in {model_file}")
        
        except Exception as e:
            print(f"ERROR loading {model_file}: {e}")
            continue
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        return results_df
    else:
        print("No predictions generated")
        return None


def print_prediction_report(prediction_result):
    """Print formatted report of predictions."""
    
    print("\n" + "=" * 80)
    print(f"CONNECTION PREDICTIONS FOR: {prediction_result['query_node']} ({prediction_result['node_type'].upper()})")
    print("=" * 80)
    
    if not prediction_result['found']:
        print(f"❌ Node not found in training data: {prediction_result.get('error', 'Unknown error')}")
        return
    
    print(f"\n📊 HIGH CONFIDENCE CONNECTIONS (threshold ≥ 0.5):")
    if prediction_result['high_confidence']:
        for idx, (conn_id, prob) in enumerate(prediction_result['high_confidence'][:20], 1):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {idx:2d}. {conn_id:20s} {bar} {prob:.4f}")
    else:
        print("  (None with confidence ≥ 0.5)")
    
    print(f"\n📈 TOP {len(prediction_result['top_predictions'])} PREDICTIONS:")
    for idx, (conn_id, prob) in enumerate(prediction_result['top_predictions'], 1):
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        confidence = "✓ HIGH" if prob >= 0.5 else "~ MEDIUM" if prob >= 0.3 else "○ LOW"
        print(f"  {idx:2d}. {conn_id:20s} {bar} {prob:.4f} [{confidence}]")
    
    print(f"\n📊 Statistics:")
    print(f"  Total possible connections: {len(prediction_result['all_predictions'])}")
    print(f"  High confidence (≥0.5): {len(prediction_result['high_confidence'])}")
    print(f"  Medium confidence (0.3-0.5): {len([p for p in prediction_result['all_predictions'] if 0.3 <= p[1] < 0.5])}")
    print("=" * 80)


def save_inference_model(models_dir='temporal_models', results_dir='results', output_file='all_temporal_models.pkl'):
    """
    Save all loaded models and metadata for quick inference later.
    
    Args:
        models_dir: Directory with trained models
        results_dir: Results directory to save the inference package
        output_file: Output pickle file name for inference
    """
    
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
    
    print(f"Packaging {len(model_files)} models for inference...")
    
    inference_package = {
        'models': {},
        'metadata': {
            'created': pd.Timestamp.now(),
            'n_models': len(model_files),
            'model_files': model_files,
            'models_dir': models_dir,
            'results_dir': results_dir
        }
    }
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            with open(model_path, 'rb') as f:
                inference_package['models'][model_file] = pickle.load(f)
            print(f"  ✓ Packaged {model_file}")
        except Exception as e:
            print(f"  ✗ Error with {model_file}: {e}")
    
    # Save to results directory
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(inference_package, f)
    
    print(f"\n✓ Inference package saved to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    return output_path

def export_all_predictions_to_csv(models_dir='results/temporal_models', output_path='results/all_predictions_scores.csv', threshold=0.1):
    """
    מחלצת ציוני חיזוי מכל המודלים, מחברת נתוני מניות (Name, Ticker) ושומרת ל-CSV באותיות גדולות.
    """
    import os
    import pandas as pd
    
    # 1. הגדרת נתיב וטעינת ה-Ticker Map (באותו פורמט של load_data בסקריפט)
    personal_dir = os.path.expanduser('~')
    root = os.path.join(personal_dir, 'Social-Network-Stock-Market/SocialNetwork/parquet_files')
    
    try:
        ticker_map = pd.read_parquet(os.path.join(root, "ticker_to_cusip.parquet"))
        # הפיכת עמודות המפה לאותיות גדולות לאחידות
        ticker_map.columns = [c.upper() for c in ticker_map.columns]
    except Exception as e:
        print(f"⚠ Warning: Could not load ticker_map for join: {e}")
        ticker_map = None

    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
    if not model_files:
        print(f"No models found in: {models_dir}")
        return

    all_rows = []
    for model_file in model_files:
        print(f"Processing scores for: {model_file}...")
        predictor = NodeConnectionPredictor(os.path.join(models_dir, model_file))
        test_q = f"{predictor.test_quarter[0]}Q{predictor.test_quarter[1]}"
        
        for fund_id in predictor.funds_train:
            preds = predictor.predict_connections(fund_id, node_type='fund', top_k=None)
            if preds['found']:
                for stock_id, score in preds['all_predictions']:
                    if score >= threshold:
                        all_rows.append({
                            'TEST_QUARTER': test_q,
                            'FUND_CIK': fund_id,
                            'STOCK_CUSIP': stock_id,
                            'PREDICTION_SCORE': round(score, 4)
                        })
    
    if not all_rows:
        print("No predictions found above threshold.")
        return

    df_results = pd.DataFrame(all_rows)

    # 2. ביצוע ה-JOIN (שימוש ב-STOCK_CUSIP מול CUSIP של המפה)
    if ticker_map is not None:
        # אנחנו מוודאים שגם העמודות שאנחנו מוסיפים הן ב-Uppercase
        df_results = df_results.merge(
            ticker_map[['CUSIP', 'NAME', 'TICKER']], 
            left_on='STOCK_CUSIP', 
            right_on='CUSIP', 
            how='left'
        )
        # הסרת העמודה הכפולה CUSIP (כי יש לנו STOCK_CUSIP)
        if 'CUSIP' in df_results.columns:
            df_results = df_results.drop(columns=['CUSIP'])
    
    # 3. שמירה סופית
    df_results.to_csv(output_path, index=False)
    print(f"✓ Success! Detailed report saved to: {output_path}")
    print(f"Total prediction records: {len(df_results):,}")
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    if data is not None:
        # Build quarterly graphs
        print("\nBuilding quarterly bipartite graphs (2021-2024 only)...")
        quarterly_graphs = build_quarterly_graphs(data)
        print(f"\nTotal quarters: {len(quarterly_graphs)}")
        if quarterly_graphs:
            min_q, max_q = min(quarterly_graphs.keys()), max(quarterly_graphs.keys())
            print(f"Date range: {min_q} to {max_q}")
        
        # Show example sliding windows
        chrono_quarters = get_chronological_quarters(quarterly_graphs)
        print(f"\nTotal quarters available: {len(chrono_quarters)}")
        print(f"All quarters: {chrono_quarters}")

        print("\nSliding window examples (train_window=3, test_offset=1):")
        for i, (train_q, test_q) in enumerate(list(get_sliding_window_splits(chrono_quarters, train_window=3))[:3]):
            print(f"  Window {i+1}:")
            print(f"    Train: {[f'{y}Q{q}' for y, q in train_q]}")
            print(f"    Test:  {test_q[0]}Q{test_q[1]}")
        
        # Run temporal link prediction - saves to results/ folder
        results_df, models_dir, results_dir = run_temporal_link_prediction(quarterly_graphs, train_window=3, test_offset=1, results_dir='results')
        
        export_all_predictions_to_csv(
            models_dir=models_dir, 
            output_path=os.path.join(results_dir, 'final_scores_report.csv'), 
            threshold=0.2
        )
        # ====================================================================
        # INFERENCE: NODE CONNECTION PREDICTION
        # ====================================================================
        # After training, use the saved models to predict connections for any node
        
        print("\n\n" + "=" * 100)
        print("INFERENCE: LOADING MODELS FOR CONNECTION PREDICTION")
        print("=" * 100)
        
        # Check if models were saved
        if models_dir and os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            if model_files:
                print(f"\n✓ Found {len(model_files)} saved models in {models_dir}/")
                
                # Example 1: Load first model and predict connections
                print("\n" + "-" * 100)
                print("EXAMPLE 1: Predict connections for a fund using the first model")
                print("-" * 100)
                
                first_model_path = os.path.join(models_dir, model_files[0])
                
                try:
                    predictor = NodeConnectionPredictor(first_model_path)
                    
                    # Get a sample fund from the model
                    sample_fund = predictor.funds_train[0]
                    
                    print(f"\nQuerying fund: {sample_fund}")
                    print(f"Model covers: {predictor.train_quarters} → {predictor.test_quarter}")
                    
                    # Predict top 15 stocks this fund might hold
                    prediction = predictor.predict_connections(sample_fund, node_type='fund', top_k=15, threshold=0.5)
                    print_prediction_report(prediction)
                    
                    # Example 2: Predict fund connections for a stock
                    print("\n" + "-" * 100)
                    print("EXAMPLE 2: Predict which funds will hold a specific stock")
                    print("-" * 100)
                    
                    sample_stock = predictor.stocks_train[0]
                    print(f"\nQuerying stock: {sample_stock}")
                    
                    prediction = predictor.predict_connections(sample_stock, node_type='stock', top_k=15, threshold=0.5)
                    print_prediction_report(prediction)
                    
                except Exception as e:
                    print(f"Error during inference example: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Example 3: Save all models for batch inference
                print("\n" + "-" * 100)
                print("EXAMPLE 3: Packaging all models for inference")
                print("-" * 100)
                
                try:
                    inference_file = save_inference_model(models_dir=models_dir, results_dir=results_dir, output_file='all_temporal_models.pkl')
                    print(f"\n✓ All models packaged and ready for inference!")
                except Exception as e:
                    print(f"Error packaging models: {e}")
            else:
                print("No .pkl model files found in temporal_models/")
        else:
            print(f"Models directory not found. Run training first.")
        
        print("\n" + "=" * 100)
        print("PIPELINE COMPLETE - ALL RESULTS SAVED")
        print("=" * 100)
        print(f"\n📁 Results Directory Structure:")
        print(f"   {results_dir}/")
        print(f"   ├── temporal_link_prediction_results.csv")
        print(f"   ├── all_temporal_models.pkl")
        print(f"   └── temporal_models/")
        print(f"       ├── window_1_...pkl")
        print(f"       ├── window_2_...pkl")
        print(f"       └── ...\n")
        print(f"✓ You can now use NodeConnectionPredictor to query any fund/stock!")
        print(f"  Example usage:")
        print(f"    predictor = NodeConnectionPredictor('{models_dir}/window_1_...pkl')")
        print(f"    results = predictor.predict_connections(fund_id, node_type='fund', top_k=10)")
        print(f"    print_prediction_report(results)")
