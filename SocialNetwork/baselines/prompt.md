# Stock Market Social Network - Project Documentation

## Project Overview

**Title**: Stock Market Social Network - Temporal Link Prediction  
**Purpose**: Predict which stocks an investment fund is likely to buy in the next quarter using advanced graph-based machine learning and temporal analysis.  
**Data Source**: SEC 13F Reports (official quarterly holdings reports from U.S. investment funds with assets >$100M)  
**Time Period**: 2021-2024 (quarterly data)  
**Architecture**: Production-grade Python pipeline with strict temporal causality and no data leakage

---

## Business Objective

Enable investors and fund managers to:
- Predict fund behavior in equity markets
- Identify investment trends and fund interconnections
- Gain competitive advantage through early identification of stock movements
- Analyze network effects in institutional investing

---

## Technical Architecture

### High-Level Pipeline Flow

```
SEC 13F Data
    ↓
[ETL Pipeline] → Extract, Transform, Load
    ↓
[Data Cleaning] → Normalize, Filter 2021-2024
    ↓
[Quarterly Preparation] → Build per-quarter datasets
    ↓
[Sliding Window Loop] ← Iterate through quarters
    ├─ [Graph Construction] → Bipartite fund-stock graphs
    ├─ [Feature Engineering] → Topological features
    ├─ [GraphSAGE] → Generate node embeddings (parallel)
    ├─ [Negative Sampling] → Create training pairs
    ├─ [LightGBM Training] → Binary classification model
    └─ [Evaluation] → Test on hold-out quarter
    ↓
[Output Reports] → AUC/Precision/Recall metrics & predictions
```

### Component Breakdown

#### 1. **ETL Pipeline** (`ETL/`)
Modular, extensible Extract-Transform-Load system:
- **Extractors**: Multiple data source support (CSV, XML, Database)
- **Data Handlers**: CSV/XML file handlers, Database handlers (PostgreSQL support)
- **Manipulation Layer**: Data transformation and filtering logic
- **Load Layer**: Multiple output destinations (Files, PostgreSQL, Graph DB)
- **DAL (Data Access Layer)**: Abstraction for database operations
- **Logging**: Comprehensive ETL logging and error tracking

**Key Features**:
- Extensible architecture using Strategy pattern
- Support for SEC 13F XML/CSV parsing
- Context-based extractor selection
- Error handling and validation at each stage

#### 2. **Data Pipeline** (`Asaf/network-pipeline.py`)
Quarterly holdings data processing:

**Input**:
- Parquet files containing fund-stock holdings (CIK, CUSIP, VALUE, SSHPRNAMT, PERIOD_DATE, QUARTER)
- Reference data: ticker-to-CUSIP mapping, historical prices

**Processing**:
- Load and normalize column names to uppercase
- Filter data to 2021-2024 timeframe
- Create quarterly reference datasets
- Handle missing values and data validation

**Output**:
- Clean quarterly DataFrames ready for graph construction

#### 3. **Graph Construction**
Builds multiple graph representations per quarter:

**Bipartite Graph** `G=(F, S, E)`:
- **F**: Fund nodes (CIK identifiers)
- **S**: Stock nodes (CUSIP identifiers)  
- **E**: Edges representing fund-stock holdings
- **Edge attributes**: VALUE, SSHPRNAMT, PERIOD_DATE, weight calculations

**Fund-Fund Projected Graph**:
- Projects bipartite to unipartite (fund similarity network)
- Similarity metric: Shared stock holdings and co-investment patterns
- Directionality: Determined by temporal ordering (first mover → follower)
- Use case: Identify investment trends and fund influence networks

**Key Algorithm**:
```
For each pair (fund1, fund2):
    shared_stocks = common_holdings(fund1, fund2)
    if |shared_stocks| > threshold:
        weight = similarity_score(fund1, fund2)
        time_order = determine_leader(fund1, fund2)
        add_directed_edge(leader → follower, weight)
```

#### 4. **Feature Engineering**
Extracts 5 topological features per node per quarter:

1. **Degree Centrality** `deg(v)`
   - Count of edges connected to node v
   - Interpretation: Fund diversification or stock popularity

2. **PageRank** `PR(v)` (100 iterations, damping factor 0.85)
   - Recursive importance: Node is important if connected to important nodes
   - Formula: `PR(v) = (1-d)/N + d * Σ(PR(u)/L(u) for u→v)`
   - Interpretation: Fund influence in the investment ecosystem

3. **HITS Algorithm** (Hubs & Authorities, 100 iterations)
   - Authority Score: Receives edges from strong hubs
   - Hub Score: Points to strong authorities
   - Interpretation: Authority funds vs. pioneering/trend-following funds

4. **Closeness Centrality** `C(v) = (n-1) / Σ d(u,v)`
   - Average distance to all other nodes
   - Interpretation: How quickly information/trends spread from this fund

5. **Community Detection** (Leiden algorithm via igraph)
   - Assigns community labels to each node
   - Interpretation: Fund clusters or sector groupings
   - Output: community_id per node

**Feature Processing**:
- Compute per node per quarter
- Normalize using MinMaxScaler (0-1 range)
- Store as feature vectors for ML input

#### 5. **Node Embeddings (GraphSAGE)**
Graph Neural Network for representation learning:

**Architecture**:
- Input: Node features + graph topology
- Layers: Multiple aggregation layers with neighborhood sampling
- Output: Low-dimensional embeddings (e.g., 64-128 dims)

**Algorithm** (per layer k):
```
h_v^(k) = σ(W^(k) · AGGREGATE^(k)({h_v^(k-1)} ∪ {h_u^(k-1), ∀u ∈ N(v)}))
```
Where:
- `h_v^(k)`: Embedding of node v at layer k
- `N(v)`: Neighbors of v
- `W^(k)`: Learnable weight matrix
- `σ`: Nonlinearity (ReLU, etc.)

**Key Properties**:
- Inductive: Generalizes to unseen nodes
- Neighborhood aggregation: Captures local structure
- Parallel training on GPU (CUDA supported)

**Output**: Embedding matrix `embeddings[fund/stock] = [64-128 dimensional vector]`

#### 6. **Negative Sampling**
Creates balanced training dataset:

**Positive Samples**: Actual fund-stock holdings in next quarter
**Negative Samples**: Non-existent fund-stock pairs (sampled randomly)

**Balancing Strategy**:
- Ratio: 1:1 or adjustable based on data imbalance
- Prevents model bias toward negative class
- Ensures realistic prediction thresholds

#### 7. **LightGBM Classification**
Gradient boosting model for temporal link prediction:

**Input Features** (concatenated):
- Fund features (degree, pagerank, hits_hub, hits_auth, closeness, community)
- Stock features (same 6 features)
- Fund embedding (64-128 dims)
- Stock embedding (64-128 dims)
- **Total**: ~6+6+64 = 76+ dimensions per sample

**Model Configuration**:
- Objective: Binary classification (will/won't buy)
- Metric: AUC-ROC (primary), Precision, Recall, F1
- Hyperparameters: Tunable (num_leaves, learning_rate, etc.)
- Output: Probability prediction ∈ [0,1]

**Training Data**:
- Quarters Q1-Q8 of 2-year window (training)
- Evaluated on Quarter Q9 (test)

---

## Sliding Window Approach

**Purpose**: Simulate real-world sequential prediction with strict temporal causality

**Mechanism**:
```
Window 1: Q1 2021 - Q2 2023 (train) → Q3 2023 (predict)
Window 2: Q2 2021 - Q3 2023 (train) → Q4 2023 (predict)
Window 3: Q3 2021 - Q4 2023 (train) → Q1 2024 (predict)
...
Window N: Q1 2023 - Q4 2023 (train) → Q1 2024 (predict)
```

**Guarantees**:
- ✅ No time leakage: Future quarters never used in feature computation
- ✅ No data leakage: Test quarter strictly separate from training window
- ✅ No label leakage: Model never sees target quarter during training
- ✅ Out-of-sample evaluation: All test data is temporally future

---

## Data Flow & Quality Assurance

### Input Data Schema

Each quarterly CSV/Parquet contains:
```
CIK (string)           # Fund identifier (SEC Central Index Key)
CUSIP (string)         # Stock identifier (Committee on Uniform Securities ID Procedures)
VALUE (float)          # Dollar amount of holding
SSHPRNAMT (float)      # Share amount
PERIOD_DATE (datetime) # Report date (end of quarter)
QUARTER (string)       # Quarter label (e.g., "Q1 2021")
```

### Data Cleaning Steps
1. Remove NULL/NaN values
2. Validate CIK and CUSIP formats
3. Convert dates to datetime format
4. Normalize column names to uppercase
5. Filter to 2021-2024 timeframe only
6. Remove duplicate entries (CIK-CUSIP pairs)
7. Aggregate by quarter to get latest value

---

## Evaluation Metrics (Per Sliding Window)

### Primary Metrics
- **AUC-ROC**: Area Under Receiver Operating Characteristic curve (0-1, higher better)
  - Measures: Ability to rank positives vs. negatives
  - Interpretation: 0.5 = random, 1.0 = perfect, ≥0.7 = good

- **Precision**: TP / (TP + FP)
  - Interpretation: Of predicted holdings, what fraction are correct?

- **Recall**: TP / (TP + FN)
  - Interpretation: Of actual holdings, what fraction were identified?

- **F1-Score**: Harmonic mean of Precision and Recall
  - Balanced metric for imbalanced datasets

### Output Reports
- **CSV Report**: `final_scores_report.csv`
  - Columns: Window, AUC, Precision, Recall, F1, Timestamp
  - Usage: Track model performance over time
  - Actionability: Identify quarters with degraded performance

---

## Production Inference

### Loading a Trained Model
```python
from network_pipeline import NodeConnectionPredictor

# Load saved model
predictor = NodeConnectionPredictor.load_model('artifacts/model_Q1_2024.pkl')

# Predict for specific fund
recommendations = predictor.predict(
    fund_cik='1325091',
    top_k=10
)
# Output: List of top 10 stocks recommended for purchase
```

### Prediction Confidence
- Model outputs probability for each fund-stock pair
- Ranking by probability provides investment recommendation ordering
- Confidence threshold: Adjustable based on risk tolerance

---

## Key Technologies & Libraries

### Graph & Network
- **NetworkX**: Bipartite graph construction, centrality algorithms
- **igraph + leidenalg**: Community detection (Leiden algorithm)
- **PyTorch Geometric**: Graph Neural Networks (GraphSAGE)

### ML & Data Science
- **LightGBM**: Gradient boosting for classification
- **scikit-learn**: Metrics, preprocessing (MinMaxScaler)
- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical operations

### Deep Learning
- **PyTorch**: Neural network framework with CUDA support
- **CUDA**: GPU acceleration (if available)

### Infrastructure
- **PostgreSQL**: Optional database storage
- **Pickle/Joblib**: Model serialization
- **Logging**: Comprehensive audit trail

---

## Directory Structure

```
Social-Network-Stock-Market/
├── ETL/                           # Extract-Transform-Load pipeline
│   ├── etl_pipeline.py           # Main ETL orchestration
│   ├── Extractors/               # Data source extractors
│   ├── data_handlers/            # File/Database handlers
│   ├── manipulation/             # Data transformation logic
│   ├── load/                     # Data loaders (files, DB)
│   ├── logger/                   # Logging utilities
│   ├── dal/                      # Data Access Layer
│   ├── utils/                    # CSV-to-Parquet converters
│   └── tests/                    # Test suites
│
├── SocialNetwork/
│   └── src/
│       ├── Asaf/
│       │   ├── network-pipeline.py       # Main prediction pipeline
│       │   ├── network-pipeline.ipynb    # Jupyter notebook version
│       │   ├── open_parquet.py          # Utility for loading parquet
│       │   ├── html_report_creators/    # Report generation
│       │   ├── Bash_scripts/            # Automation scripts
│       │   └── artifacts/               # Saved models & embeddings
│       ├── Notebooks/
│       │   └── LightGCN/                # Alternative GNN approaches
│       └── ...
│
├── Data/
│   ├── quarterly_datasets.json   # Quarter configuration
│   ├── Indexes/                  # Index data (Russell 3000, SPY)
│   └── Holdings_details_*.csv    # Reference holdings
│
├── RawToFilteredtable/           # Data filtering pipeline
│   ├── filterholdings/           # Holdings filtering logic
│   └── indices_parser/           # Index parsing utilities
│
└── requirements.txt              # Python dependencies
```

---

## Execution Flow

### Development/Research
1. **Data Preparation**
   ```bash
   python etl_pipeline.py  # Extract from SEC, load to DB/CSV
   ```

2. **Exploratory Analysis**
   - Open `network-pipeline.ipynb` in Jupyter
   - Inspect quarterly graphs and features
   - Validate data quality

3. **Model Training & Evaluation**
   ```bash
   python network-pipeline.py  # Run full pipeline with sliding windows
   ```
   - Generates `final_scores_report.csv`
   - Saves models to `artifacts/`

4. **Inference & Recommendations**
   ```python
   predictor = NodeConnectionPredictor.load_model(...)
   recommendations = predictor.predict(fund_cik='1325091')
   ```

### Production Automation
- **Bash Scripts** in `Bash_scripts/`:
  - `run_network_pipeline.sh`: Full pipeline execution
  - `run_network_pipeline_only_cpu.sh`: CPU-only version
  - `cloud_script_template.sh`: Cloud deployment template

---

## Key Design Decisions

### 1. **Bipartite vs. Unipartite Graphs**
- **Bipartite** (fund-stock): Preserves original relationships
- **Projected** (fund-fund): Enables influence modeling
- Both used together for comprehensive feature set

### 2. **Sliding Window Over Single Train-Test Split**
- Rationale: More robust evaluation across time
- Multiple windows simulate rolling production deployment
- Captures temporal dynamics and model degradation

### 3. **GraphSAGE for Embeddings**
- Advantages: Inductive learning, scalable, proven on heterogeneous networks
- Alternative considered: GAT (Graph Attention), GCN (Graph Convolution)
- GPU acceleration for large graphs

### 4. **LightGBM for Classification**
- Advantages: Fast, interpretable, handles high-dimensional features
- Alternative considered: Neural networks, SVM
- Tuned for imbalanced dataset (many non-holdings vs. few holdings)

### 5. **Temporal Causality Enforcement**
- All features computed from historical data only
- Strict date filtering to prevent leakage
- Evaluation on future quarters only

---

## Success Criteria & Metrics

- **Model Performance**: AUC-ROC ≥ 0.7 across sliding windows
- **Robustness**: Consistent metrics across different quarters
- **Scalability**: Process 1000+ funds × 5000+ stocks in <10 min/quarter
- **Interpretability**: Feature importance analysis available
- **Reproducibility**: Saved models enable deterministic predictions

---

## Deployment Considerations

### Scaling
- Partition processing by fund subset (10-100 funds/process)
- Distributed graph construction using Dask or Spark
- Model serving via REST API (Flask/FastAPI)

### Monitoring
- Track AUC/Precision per new quarter
- Alert on metric degradation (drift detection)
- Log prediction confidence distribution

### Retraining
- Retrain model monthly or quarterly
- Incorporate new fund-stock relationships
- Archive historical models for comparison

---

## Dependencies

### Core Requirements
- Python ≥ 3.8
- pandas (data manipulation)
- numpy (numerical computing)
- networkx (graph algorithms)
- igraph + leidenalg (community detection)
- torch + torch_geometric (GNNs)
- lightgbm (gradient boosting)
- scikit-learn (ML utilities)
- psycopg2-binary (PostgreSQL, optional)

### Development
- Jupyter (notebooks)
- matplotlib, seaborn (visualization)
- pytest (testing)

---

## References & Further Reading

- **Graph Neural Networks**: Kipf & Welling (GCN), Hamilton et al. (GraphSAGE)
- **Temporal Link Prediction**: Zuo et al. (survey), Dunlavy et al. (temporal networks)
- **Community Detection**: Traag et al. (Leiden algorithm)
- **LightGBM**: Microsoft LightGBM documentation
- **SEC 13F Forms**: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F&dateb=&owner=exclude&count=100

---

## Authors & Contributions

- **Primary Architect**: Data Science Team
- **ETL Development**: Backend Engineers
- **Network Analysis**: Asaf
- **ML Model Development**: ML Engineers
- **Testing & Validation**: QA Team

**Contact**: For questions or collaboration, please reach out to the development team.

---

**Last Updated**: 2026-05-04  
**Version**: 1.0 (Production Ready)
