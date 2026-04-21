"""
Stock Market Social Network - Temporal Link Prediction Pipeline (OPTIMIZED)

Build quarterly bipartite graphs of fund-stock holdings and perform temporal link prediction 
using sliding windows (2023-2025).

Key Features:
- Load holdings data 2021-2024 only
- On-demand window graph building (memory efficient)
- Sliding window: train on configurable quarters, predict next quarter
- Strict temporal causality (no future leakage)
- Per-quarter evaluation metrics (AUC, Precision, Recall)
- Global CIK/CUSIP identity management with persistent embeddings

Architecture:
- HoldingsDataLayer: On-demand temporal data access
- BipartiteGraphSAGE: Persistent embedding model across windows
- Direct DataFrame → PyTorch tensor mapping (no NetworkX roundtrips)
- Preserved edge-level history (frequency, duration, VALUE statistics)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import os
import re
import warnings
import pickle

# Try Polars for maximum speed, fallback to Pandas
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("⚠ Polars not available, using Pandas (slower). Install with: pip install polars")

# ML libraries
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. HOLDINGS DATA LAYER (On-Demand Temporal Graph Building)
# ============================================================================

class HoldingsDataLayer:
    """
    On-demand temporal graph builder. Loads data once, builds windows as needed.
    
    Key features:
    - Global CIK/CUSIP ID mapping (persistent across windows)
    - Temporal feature aggregation (frequency, duration, VALUE stats)
    - Direct Polars/Pandas → PyTorch tensor mapping
    """
    
    def __init__(self, data, n_funds, n_stocks, device='cpu'):
        """
        Args:
            data: DataFrame (Polars or Pandas) with columns [CIK_ID, CUSIP_ID, VALUE, YEAR, QUARTER, ...]
            n_funds: Total number of unique funds
            n_stocks: Total number of unique stocks
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.n_funds = n_funds
        self.n_stocks = n_stocks
        self.n_total = n_funds + n_stocks
        self.device = device
        
        # Convert to native format for efficient partitioning
        if POLARS_AVAILABLE and hasattr(data, 'partition_by'):
            # Polars DataFrame
            self.data = data
            self._by_quarter = self._partition_polars()
        else:
            # Pandas DataFrame
            self.data = data
            self._by_quarter = self._partition_pandas()
        
        print(f"✓ DataLayer initialized: {len(self._by_quarter)} quarters")
    
    def _partition_polars(self):
        """Partition Polars DataFrame by (YEAR, QUARTER)."""
        partitions = {}
        for part in self.data.partition_by(["YEAR", "QUARTER"]):
            year = int(part["YEAR"][0])
            quarter = int(part["QUARTER"][0])
            partitions[(year, quarter)] = part
        return partitions
    
    def _partition_pandas(self):
        """Partition Pandas DataFrame by (YEAR, QUARTER)."""
        partitions = {}
        for (year, quarter), group in self.data.groupby(["YEAR", "QUARTER"]):
            partitions[(int(year), int(quarter))] = group
        return partitions
    
    def quarters(self):
        """Return sorted list of all available (year, quarter) tuples."""
        return sorted(self._by_quarter.keys())
    
    def _bipartite_edge_index(self, src_fund, dst_stock):
        """Create undirected edge index for bipartite graph.
        
        Packing: funds occupy [0, n_funds), stocks occupy [n_funds, n_funds + n_stocks)
        """
        dst = dst_stock + self.n_funds
        return torch.stack([
            torch.cat([src_fund, dst]),
            torch.cat([dst, src_fund]),
        ])
    
    def window_graph(self, quarters):
        """
        Aggregate quarters into one bipartite graph with temporal features.
        
        Args:
            quarters: List of (year, quarter) tuples
        
        Returns:
            Dict with keys:
            - 'edge_index': [2, 2E] edge tensor (undirected)
            - 'edge_attr': [E, 5] temporal features per edge (forward only)
            - 'src': [E] fund node IDs
            - 'dst': [E] stock node IDs
        """
        dfs = [self._by_quarter[q] for q in quarters if q in self._by_quarter]
        if not dfs:
            return None
        
        # Combine all quarters
        if POLARS_AVAILABLE:
            combined = pl.concat(dfs).sort(["YEAR", "QUARTER"])
        else:
            combined = pd.concat(dfs).sort_values(["YEAR", "QUARTER"]).reset_index(drop=True)
        
        # Convert year/quarter to quarter index for duration calculation
        if POLARS_AVAILABLE:
            combined = combined.with_columns(
                (pl.col("YEAR").cast(pl.Int32) * 4 + pl.col("QUARTER").cast(pl.Int32))
                .alias("q_idx")
            )
        else:
            combined['q_idx'] = combined['YEAR'].astype(int) * 4 + combined['QUARTER'].astype(int)
        
        # Aggregate edges: compute per-edge temporal features
        if POLARS_AVAILABLE:
            agg = combined.group_by(["CIK_ID", "CUSIP_ID"]).agg([
                pl.len().alias("frequency"),
                (pl.col("q_idx").max() - pl.col("q_idx").min() + 1).alias("duration"),
                pl.col("VALUE").mean().fill_null(0.0).alias("mean_value"),
                pl.col("VALUE").std().fill_null(0.0).alias("std_value"),
                pl.col("VALUE").last().fill_null(0.0).alias("last_value"),
            ])
            src_np = agg["CIK_ID"].to_numpy()
            dst_np = agg["CUSIP_ID"].to_numpy()
            
            # Extract Polars Series to numpy arrays (FIX: Polars Series don't have .values)
            freq_np = agg["frequency"].to_numpy().astype(np.float32)
            duration_np = agg["duration"].to_numpy().astype(np.float32)
            mean_value_np = agg["mean_value"].to_numpy().astype(np.float32)
            std_value_np = agg["std_value"].to_numpy().astype(np.float32)
            last_value_np = agg["last_value"].to_numpy().astype(np.float32)
        else:
            agg = combined.groupby(["CIK_ID", "CUSIP_ID"]).agg({
                "q_idx": ["min", "max"],
                "VALUE": ["mean", "std", "last"]
            }).reset_index()
            agg.columns = ["CIK_ID", "CUSIP_ID", "q_idx_min", "q_idx_max", 
                          "mean_value", "std_value", "last_value"]
            agg["frequency"] = combined.groupby(["CIK_ID", "CUSIP_ID"]).size().values
            agg["duration"] = agg["q_idx_max"] - agg["q_idx_min"] + 1
            agg["mean_value"] = agg["mean_value"].fillna(0.0)
            agg["std_value"] = agg["std_value"].fillna(0.0)
            agg["last_value"] = agg["last_value"].fillna(0.0)
            src_np = agg["CIK_ID"].values
            dst_np = agg["CUSIP_ID"].values
            
            # Extract Pandas Series to numpy arrays
            freq_np = agg["frequency"].values.astype(np.float32)
            duration_np = agg["duration"].values.astype(np.float32)
            mean_value_np = agg["mean_value"].values.astype(np.float32)
            std_value_np = agg["std_value"].values.astype(np.float32)
            last_value_np = agg["last_value"].values.astype(np.float32)
        
        src = torch.from_numpy(src_np.astype(np.int64)).to(self.device)
        dst = torch.from_numpy(dst_np.astype(np.int64)).to(self.device)
        
        # Temporal features: [frequency, duration, mean_value, std_value, last_value]
        feats = np.stack([
            freq_np,
            duration_np,
            mean_value_np,
            std_value_np,
            last_value_np,
        ], axis=1)
        
        # Log-compress heavy-tailed VALUE columns to prevent numerical instability
        feats[:, 2:5] = np.sign(feats[:, 2:5]) * np.log1p(np.abs(feats[:, 2:5]))
        
        return {
            "edge_index": self._bipartite_edge_index(src, dst),
            "edge_attr": torch.from_numpy(feats).to(self.device),  # [E, 5]
            "src": src,
            "dst": dst,
        }
    
    def quarter_edges(self, yq):
        """Return positive fund→stock edges for a single quarter.
        
        Used for test-time evaluation. Returns None if quarter not available.
        """
        df = self._by_quarter.get(yq)
        if df is None:
            return None
        
        if POLARS_AVAILABLE:
            src_np = df["CIK_ID"].to_numpy()
            dst_np = df["CUSIP_ID"].to_numpy()
        else:
            src_np = df["CIK_ID"].values
            dst_np = df["CUSIP_ID"].values
        
        src = torch.from_numpy(src_np.astype(np.int64)).to(self.device)
        dst = torch.from_numpy(dst_np.astype(np.int64)).to(self.device)
        
        return src, dst


# ============================================================================
# 2. BIPARTITE GRAPHSAGE MODEL (Persistent Embeddings)
# ============================================================================

class BipartiteGraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE over unified bipartite node space.
    
    Key feature: Embeddings indexed by global (CIK_ID / CUSIP_ID) persist
    across sliding windows, enabling true continual temporal learning.
    """
    
    def __init__(self, n_funds, n_stocks, emb_dim=32, hidden_dim=64, out_dim=32):
        """
        Args:
            n_funds: Total number of unique funds
            n_stocks: Total number of unique stocks
            emb_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
        """
        super().__init__()
        self.n_funds = n_funds
        self.n_stocks = n_stocks
        self.n_total = n_funds + n_stocks
        
        # Persistent embedding tables (indexed by global ID)
        self.fund_emb = nn.Embedding(n_funds, emb_dim)
        self.stock_emb = nn.Embedding(n_stocks, emb_dim)
        nn.init.normal_(self.fund_emb.weight, std=0.1)
        nn.init.normal_(self.stock_emb.weight, std=0.1)
        
        # Graph convolutions
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    
    def node_features(self):
        """Concatenate fund and stock embeddings into unified node feature matrix."""
        return torch.cat([self.fund_emb.weight, self.stock_emb.weight], dim=0)
    
    def forward(self, edge_index):
        """Forward pass: apply GraphSAGE convolutions."""
        x = self.node_features()
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
    def score_pairs(self, z, fund_ids, stock_ids):
        """Score fund-stock pairs via dot product."""
        stock_ids_adjusted = stock_ids + self.n_funds
        return (z[fund_ids] * z[stock_ids_adjusted]).sum(dim=-1)


# ============================================================================
# 3. SLIDING WINDOW UTILITIES
# ============================================================================

def get_sliding_window_splits(chronological_quarters, train_window=3, test_offset=1):
    """
    Generate temporal train/test splits using sliding window.
    
    Args:
        chronological_quarters: Sorted list of (year, quarter) tuples
        train_window: Number of quarters for training
        test_offset: Number of quarters ahead to test (default: 1)
    
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
# 4. TRAINING & EVALUATION
# ============================================================================

def train_window(model, graph, device, epochs=30, lr=0.01):
    """
    Train GraphSAGE on a single window with frequency-weighted loss.
    
    Args:
        model: BipartiteGraphSAGE instance
        graph: Output from HoldingsDataLayer.window_graph()
        device: PyTorch device
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        (z, loss): Node embeddings and final loss
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    edge_index = graph["edge_index"]
    src, dst = graph["src"], graph["dst"]
    
    # Weight positives by normalized edge frequency (more persistent holdings → stronger gradient)
    freq = graph["edge_attr"][:, 0]
    w = freq / freq.max().clamp(min=1.0)
    
    model.train()
    loss_val = float('nan')
    
    for epoch in range(epochs):
        opt.zero_grad()
        z = model(edge_index)
        
        # Positive edge scores
        pos_logits = (z[src] * z[dst + model.n_funds]).sum(-1)
        
        # Random negative sampling
        n = src.size(0)
        neg_src = torch.randint(0, model.n_funds, (n,), device=device)
        neg_dst = torch.randint(0, model.n_stocks, (n,), device=device)
        neg_logits = (z[neg_src] * z[neg_dst + model.n_funds]).sum(-1)
        
        # Weighted BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), weight=w, reduction='mean'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction='mean'
        )
        loss = pos_loss + neg_loss
        
        loss.backward()
        opt.step()
        loss_val = loss.item()
    
    model.eval()
    with torch.no_grad():
        z = model(edge_index)
    
    torch.cuda.empty_cache()
    return z, loss_val


def evaluate_window(model, z, data_layer, train_graph, test_q, device):
    """
    Evaluate on NEW links in the test quarter (not in training window).
    
    Returns:
        Dict with AUC, Precision, Recall, number of new links
    """
    test = data_layer.quarter_edges(test_q)
    if test is None:
        return None
    
    src_t, dst_t = test
    
    # Filter out edges that were in training window (no trivial hits)
    pack = data_layer.n_stocks
    train_pack = (train_graph["src"] * pack + train_graph["dst"]).to(device)
    test_pack = (src_t * pack + dst_t).to(device)
    mask = ~torch.isin(test_pack, train_pack)
    
    src_t, dst_t = src_t[mask], dst_t[mask]
    if src_t.numel() == 0:
        return None
    
    with torch.no_grad():
        # Positive predictions
        pos = torch.sigmoid(model.score_pairs(z, src_t, dst_t))
        
        # Negative sampling
        n = src_t.size(0)
        neg_src = torch.randint(0, model.n_funds, (n,), device=device)
        neg_dst = torch.randint(0, model.n_stocks, (n,), device=device)
        neg = torch.sigmoid(model.score_pairs(z, neg_src, neg_dst))
    
    y_true = torch.cat([torch.ones(n, device=device),
                        torch.zeros(n, device=device)]).cpu().numpy()
    y_pred = torch.cat([pos, neg]).cpu().numpy()
    y_hat = (y_pred > 0.5).astype(int)
    
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "n_new_links": int(n),
    }


# ============================================================================
# 5. DATA LOADING
# ============================================================================

def load_data():
    """
    Load reference data and quarterly holdings.
    
    Returns:
        DataFrame (Polars or Pandas) with global CIK_ID / CUSIP_ID mapping
    """
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
    print("Loading quarterly holdings data (2023-2025)...")
    print("=" * 80)
    
    combined_files = sorted([f for f in os.listdir(output_dir)
                            if f.startswith('holdings_filtered_new_period_start_') and f.endswith('.parquet')])
    
    if not combined_files:
        print("ERROR: No processed files found. Check output_dir path.")
        return None
    
    all_dfs = []
    for file in combined_files:
        df_temp = pd.read_parquet(os.path.join(output_dir, file))
        
        # Identify date column and extract year/quarter
        date_col = 'period_start' if 'period_start' in df_temp.columns else 'PERIOD_DATE'
        
        if date_col in df_temp.columns:
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp['YEAR'] = df_temp[date_col].dt.year
            df_temp['QUARTER'] = df_temp[date_col].dt.quarter
        else:
            date_match = re.search(r'(\d{4})-(\d{2})-\d{2}', file)
            if date_match:
                df_temp['YEAR'] = int(date_match.group(1))
                month_val = int(date_match.group(2))
                df_temp['QUARTER'] = (month_val - 1) // 3 + 1
            else:
                print(f"  ⚠ Warning: Could not find date for {file}, skipping.")
                continue
        
        year = df_temp['YEAR'].iloc[0]
        quarter = df_temp['QUARTER'].iloc[0]
        print(f"  ✓ Loaded {file}: {len(df_temp):,} records ({year} Q{quarter})")
        all_dfs.append(df_temp)
    
    data = pd.concat(all_dfs, ignore_index=True)
    
    # Normalize column names to UPPERCASE
    data.columns = [c.upper() for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.reset_index(drop=True)
    
    # Ensure consistent PERIOD_DATE column
    if 'PERIOD_START' in data.columns:
        data = data.rename(columns={'PERIOD_START': 'PERIOD_DATE'})
    
    data['PERIOD_DATE'] = pd.to_datetime(data['PERIOD_DATE'])
    
    # Create VALUE column if missing
    if 'VALUE' not in data.columns and 'SSHPRNAMT' in data.columns:
        data['VALUE'] = data['SSHPRNAMT']
    
    # Filter to 2023-2025 range for valid quarters
    data = data[(data['YEAR'] >= 2023) & (data['YEAR'] <= 2025)].copy()
    
    # Create global CIK_ID and CUSIP_ID mapping
    print("\nCreating global node IDs...")
    
    cik_unique = data['CIK'].unique()
    cusip_unique = data['CUSIP'].unique()
    
    cik_map = pd.DataFrame({'CIK': cik_unique, 'CIK_ID': np.arange(len(cik_unique))})
    cusip_map = pd.DataFrame({'CUSIP': cusip_unique, 'CUSIP_ID': np.arange(len(cusip_unique))})
    
    data = data.merge(cik_map, on='CIK', how='left')
    data = data.merge(cusip_map, on='CUSIP', how='left')
    
    data['CIK_ID'] = data['CIK_ID'].astype(int)
    data['CUSIP_ID'] = data['CUSIP_ID'].astype(int)
    
    N_FUNDS = len(cik_unique)
    N_STOCKS = len(cusip_unique)
    
    print(f"✓ Global fund IDs: {N_FUNDS:,}")
    print(f"✓ Global stock IDs: {N_STOCKS:,}")
    
    # Convert to Polars if available for better performance
    if POLARS_AVAILABLE:
        data = pl.from_pandas(data)
    
    return data, N_FUNDS, N_STOCKS


# ============================================================================
# 6. NODE CONNECTION PREDICTION & INFERENCE
# ============================================================================

class NodeConnectionPredictor:
    """Inference engine for predicting node connections using trained models."""
    
    def __init__(self, model_path):
        """Load a saved model from pickle file."""
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.embeddings = model_data['embeddings']
        self.train_quarters = model_data['train_quarters']
        self.test_quarter = model_data['test_quarter']
        self.metrics = model_data['metrics']
        self.n_funds = model_data['n_funds']
        self.n_stocks = model_data['n_stocks']
        
        print(f"✓ Model loaded for quarters: {self.train_quarters} → {self.test_quarter}")
        print(f"  Funds: {self.n_funds:,} | Stocks: {self.n_stocks:,}")
        print(f"  AUC: {self.metrics['auc']:.4f} | Precision: {self.metrics['precision']:.4f} | Recall: {self.metrics['recall']:.4f}")
    
    def predict_connections(self, fund_id, top_k=10, threshold=0.5):
        """Predict stock connections for a given fund."""
        if fund_id < 0 or fund_id >= self.n_funds:
            return {
                'fund_id': fund_id,
                'found': False,
                'predictions': []
            }
        
        z = torch.from_numpy(self.embeddings).to(device)
        fund_emb = z[fund_id:fund_id+1]
        stock_embs = z[self.n_funds:self.n_funds + self.n_stocks]
        
        scores = torch.sigmoid((fund_emb * stock_embs).sum(dim=-1)).cpu().detach().numpy()
        top_indices = np.argsort(-scores)[:top_k]
        
        predictions = [
            {'stock_id': int(idx), 'score': float(scores[idx])}
            for idx in top_indices
            if scores[idx] > threshold
        ]
        
        return {
            'fund_id': fund_id,
            'found': True,
            'predictions': predictions
        }


# ============================================================================
# 7. TEMPORAL LINK PREDICTION PIPELINE
# ============================================================================

def run_temporal_link_prediction(data, n_funds, n_stocks, train_window=3, test_offset=1, 
                                 results_dir='results_optimized/', epochs_per_window=30):
    """
    Run temporal link prediction with sliding window evaluation.
    
    Args:
        data: DataFrame from load_data()
        n_funds, n_stocks: Global node counts
        train_window: Quarters for training
        test_offset: Offset to test quarter
        results_dir: Directory for saving models/results
        epochs_per_window: Training epochs per window
    
    Returns:
        (results_df, models_dir, results_dir)
    """
    
    print("=" * 100)
    print(f"TEMPORAL LINK PREDICTION: SLIDING WINDOW EVALUATION")
    print(f"Train window: {train_window} quarters | Test offset: {test_offset} quarter(s)")
    print("=" * 100)
    
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, 'temporal_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize data layer and model
    data_layer = HoldingsDataLayer(data, n_funds, n_stocks, device=device)
    model = BipartiteGraphSAGE(n_funds, n_stocks).to(device)
    
    quarters = data_layer.quarters()
    print(f"\nAvailable quarters: {quarters}\n")
    
    results_per_window = []
    
    for window_idx, (train_quarters, test_quarter) in enumerate(
        get_sliding_window_splits(quarters, train_window=train_window, test_offset=test_offset)
    ):
        test_year, test_q = test_quarter
        train_label = ' → '.join([f"{y}Q{q}" for y, q in train_quarters])
        
        print(f"\n{'─' * 100}")
        print(f"WINDOW {window_idx + 1} | TEST: {test_year}Q{test_q}")
        print(f"TRAIN: {train_label}")
        print(f"{'─' * 100}")
        
        try:
            # Build training graph
            train_graph = data_layer.window_graph(train_quarters)
            if train_graph is None:
                print("  ⚠ Skipped: empty training window")
                continue
            
            # Training parameters (lower LR and epochs for transfer learning after first window)
            lr = 0.01 if window_idx == 0 else 0.001
            epochs = epochs_per_window
            
            print(f"  → Training ({epochs} epochs, lr={lr})...")
            z, loss = train_window(model, train_graph, device, epochs=epochs, lr=lr)
            print(f"    Loss: {loss:.6f}")
            
            # Evaluation
            metrics = evaluate_window(model, z, data_layer, train_graph, test_quarter, device)
            if metrics is None:
                print(f"  ⚠ Skipped: no new test links")
                continue
            
            print(f"  ✓ AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"    New links evaluated: {metrics['n_new_links']:,}")
            
            # Save model
            model_name = f"window_{window_idx+1:02d}_{test_year}Q{test_q}.pkl"
            model_path = os.path.join(models_dir, model_name)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model.cpu(),
                    'embeddings': z.cpu().numpy(),
                    'train_quarters': train_quarters,
                    'test_quarter': test_quarter,
                    'metrics': metrics,
                    'n_funds': n_funds,
                    'n_stocks': n_stocks,
                }, f)
            model = model.to(device)  # Move back to device
            print(f"  ✓ Model saved: {model_name}")
            
            # Record results
            results_per_window.append({
                'window': window_idx + 1,
                'train_quarters': train_label,
                'test_year': test_year,
                'test_quarter': test_q,
                'auc': metrics['auc'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'n_new_links': metrics['n_new_links'],
            })
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    # Summary
    print(f"\n\n{'=' * 100}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 100}")
    
    if results_per_window:
        results_df = pd.DataFrame(results_per_window)
        
        print(f"\nResults across {len(results_df)} windows:\n")
        print(results_df.to_string(index=False))
        
        print(f"\n\nAggregate Statistics:")
        print(f"  Average AUC:       {results_df['auc'].mean():.4f} (±{results_df['auc'].std():.4f})")
        print(f"  Average Precision: {results_df['precision'].mean():.4f} (±{results_df['precision'].std():.4f})")
        print(f"  Average Recall:    {results_df['recall'].mean():.4f} (±{results_df['recall'].std():.4f})")
        
        print(f"\n\nBy Year:")
        by_year = results_df.groupby('test_year')[['auc', 'precision', 'recall']].mean()
        print(by_year)
        
        # Save results
        csv_path = os.path.join(results_dir, 'temporal_link_prediction_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        print(f"✓ Models saved to {models_dir}/ ({len(results_df)} model files)")
    else:
        print("No results generated.")
        results_df = None
    
    return results_df, models_dir, results_dir


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("Stock Market Social Network - Temporal Link Prediction (Optimized)")
    print("=" * 100)
    
    # Load data
    print("\n[STEP 1] Loading data...")
    data, n_funds, n_stocks = load_data()
    
    if data is None:
        print("ERROR: Failed to load data.")
        exit(1)
    
    print(f"\n✓ Data loaded: {len(data) if hasattr(data, '__len__') else 'Polars DataFrame'} records")
    
    # Run temporal pipeline
    print("\n[STEP 2] Running temporal link prediction...")
    results_df, models_dir, results_dir = run_temporal_link_prediction(
        data, n_funds, n_stocks,
        train_window=3,
        test_offset=1,
        results_dir='results_optimized/',
        epochs_per_window=30
    )
    
    print("\n[STEP 3] Pipeline complete!")
    print(f"  Models: {models_dir}")
    print(f"  Results: {results_dir}")
