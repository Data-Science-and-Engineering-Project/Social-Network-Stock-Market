# Stock Market Social Network

A graph-based machine learning system that predicts which stocks institutional investment funds will buy in the next quarter, trained on SEC 13F filings from 2013 to present.

## Core Idea

Institutional funds that hold overlapping positions form an implicit social network. This project models that network as a bipartite graph (funds ↔ stocks), extracts graph features and embeddings (GraphSAGE), and trains a link-prediction model (LightGBM) to forecast next-quarter purchases — with strict prevention of data/time leakage.

---

## Repository Structure

```
Social-Network-Stock-Market/
│
├── Data/                        # Shared reference data
│   └── Indexes/
│       ├── RUSSELL3000 HISTORY/ # Russell 3000 index files (PDF 2013–2018, XML 2019–2025)
│       ├── Holdings_details_Russell_3000_ETF.csv
│       └── SPY.csv
│
├── ETL/                         # Stage 1 — Raw 13F ingestion
│   └── ...                      # Reads SEC EDGAR filings → Social_13F PostgreSQL DB
│
├── preprocess/                  # Stage 2 — Data preprocessing pipeline
│   ├── step1_russell3000_filtering/   # Filter holdings to valid Russell 3000 securities
│   ├── step2_delta_graphs/            # Normalized weights + quarter-over-quarter deltas
│   ├── step3_fundamental_data/        # Enrich with EODHD financial ratios
│   └── README.md                      # Full preprocessing docs
│
├── SocialNetwork/               # Stage 3 — Graph ML
│   ├── model/                   # LightGCN / GraphSAGE + LightGBM link prediction
│   └── baselines/               # Jaccard, AdamicAdar, Preferential Attachment baselines
│
├── protfolio/                   # Stage 4 — Portfolio backtesting & reporting
│
├── requirements.txt             # All project dependencies
└── .gitignore
```

---

## End-to-End Pipeline

```mermaid
graph LR
    classDef source   fill:#2E86C1,stroke:#1A5276,color:#fff
    classDef etl      fill:#1E8449,stroke:#145A32,color:#fff
    classDef db       fill:#D35400,stroke:#A04000,color:#fff
    classDef prep     fill:#2E86C1,stroke:#1A5276,color:#fff
    classDef gnn      fill:#7D3C98,stroke:#5B2C6F,color:#fff
    classDef portfolio fill:#B7950B,stroke:#7D6608,color:#fff

    SEC[(SEC EDGAR)]:::source
    EODHD[(EODHD API)]:::source
    ETL[ETL Pipeline]:::etl
    DB1[(Social_13F)]:::db
    DB2[(13FGNN)]:::db

    subgraph PRE [Preprocessing]
        s1[Russell 3000 Filtering]:::prep --> s2[Delta Graphs]:::prep --> s3[Fundamentals]:::prep
    end

    subgraph ML [GNN Model]
        g1[GraphSAGE Embeddings]:::gnn --> g2[LightGCN Classifier]:::gnn
    end

    ALGO[Portfolio Algorithm]:::portfolio
    OUT[Portfolio Suggestion]:::portfolio

    SEC --> ETL --> DB1 --> PRE --> DB2 --> ML
    ALGO --> OUT
    EODHD --> DB2
    DB2 -->|financial data| ALGO
    ML -->|model results| ALGO

    style PRE fill:#D6EAF8,stroke:#2E86C1
    style ML  fill:#E8DAEF,stroke:#7D3C98
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL with `Social_13F` DB populated from ETL
- EODHD API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the repo root:

```
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_user
DB_PASSWORD=your_password
EDOHD_API=your_eodhd_api_key
```

### 4. Run preprocessing

```bash
# Step 1 – Russell 3000 filtering
python preprocess/step1_russell3000_filtering/run_full_pipeline.py

# Step 2 – Delta graphs
python preprocess/step2_delta_graphs/run_pipeline.py

# Step 3 – Fundamentals (Jupyter notebook)
# Open preprocess/step3_fundamental_data/Insert_Fundamentaldata.ipynb and run all cells
```

### 5. Train the model

Open and run the notebooks in `SocialNetwork/model/`.

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| SEC EDGAR 13F | Quarterly institutional holdings (funds, CUSIPs, values) |
| Russell 3000 Index | Universe of valid US equities (2013–2025) |
| EODHD API | Historical prices, trading dates, and company fundamentals |

## Key Identifiers

| Identifier | Description | Example |
|-----------|-------------|---------|
| `CIK` | SEC fund identifier | `1325091` |
| `CUSIP` | 9-char security identifier | `037833100` |
| `TICKER` | Stock symbol | `AAPL` |

## Temporal Design

Data is partitioned into strict non-overlapping windows to prevent leakage:

| Split | Role |
|-------|------|
| Q1 + Q2 | Training (graph construction, model training) |
| Q3 | Validation (out-of-sample evaluation) |
| Q4 | Prediction (final inference target) |

No future data is ever visible during training or feature construction.
