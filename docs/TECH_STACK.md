# Technology Stack

Full inventory of every language, library, database, API, and tool used across the project.

---

## Languages

| Language | Used In | Purpose |
|----------|---------|---------|
| **Python 3.10+** | ETL, Preprocessing, Models, Portfolio | Primary implementation language |
| **SQL** | ETL, Preprocessing | Schema DDL, embedded queries (psycopg2) |
| **HTML** | Portfolio | Static report pages (portfolio_report.html, portfolio_report_mobile.html) |
| **YAML** | CI/CD | GitHub Actions workflow |

---

## Databases & Storage

| Technology | Type | Used In | Purpose |
|------------|------|---------|---------|
| **PostgreSQL** | Relational DB | ETL, Preprocessing, Models | Main data warehouse; two databases: `Social_13F` (raw holdings) and `13FGNN` (preprocessed for GNN) |
| **Parquet (Apache Arrow)** | Columnar file format | Preprocessing, Models, Portfolio | Data interchange between pipeline stages |

### PostgreSQL Databases

**`Social_13F`** — populated by the ETL stage
- `holdings` — range-partitioned quarterly table of SEC 13F filings
- `ticker_prices`, `ticker_to_cusip`, `holdings_filtered_new`

**`13FGNN`** — populated by the Preprocessing stage
- `stocks_return`, `normalized_holdings`, `cik_aum`, `changed_holdings`, `changed_stas`, `ticker_prices`, `ticker_to_cusip`

---

## Python Libraries

### Data Manipulation & Storage

| Library | Purpose |
|---------|---------|
| **pandas** | DataFrames, CSV/Parquet I/O, aggregations |
| **numpy** | Numerical computing and array operations |
| **pyarrow** | Parquet file format support |
| **polars** | High-performance dataframe operations (Preprocessing) |
| **joblib** | Model serialization and parallel job execution |

### Database Connectivity

| Library | Purpose |
|---------|---------|
| **psycopg2-binary** | PostgreSQL driver; used for bulk `COPY` inserts and all DB queries |

### Data Ingestion & Web Scraping

| Library | Purpose |
|---------|---------|
| **requests** | HTTP client for SEC EDGAR downloads and EODHD API calls |
| **beautifulsoup4** | HTML/XML parsing for web scraping (ticker mapping) |
| **lxml** | XML/HTML parsing for Russell 3000 index files |
| **pdfplumber** | PDF parsing for historical Russell 3000 index PDFs |
| **yfinance** | Yahoo Finance stock prices and Russell 3000 benchmark data |
| **pandas-market-calendars** | Trading calendar and market date utilities |

### Graph & Network Analysis

| Library | Purpose |
|---------|---------|
| **networkx** | Bipartite graph construction, centrality measures, classic link prediction algorithms (Jaccard, Adamic-Adar, Preferential Attachment) |
| **python-igraph** | Alternative graph library used in baseline experiments |
| **leidenalg** | Community detection via the Leiden algorithm |

### Machine Learning

| Library | Purpose |
|---------|---------|
| **scikit-learn** | Metrics (AUC, F1, precision, recall, AP), preprocessing, baseline classifiers |
| **lightgbm** | Gradient boosting classifier for link prediction ranking |

### Deep Learning & Graph Neural Networks

| Library | Purpose |
|---------|---------|
| **torch (PyTorch)** | Deep learning framework for GNN training |
| **torch-geometric** | GNN layer implementations: `GCNConv`, `SAGEConv`, `GATConv`, LightGCN |

### Visualization

| Library | Purpose |
|---------|---------|
| **matplotlib** | Static plots and charts (analysis notebooks) |
| **seaborn** | Statistical visualizations |

### Utilities

| Library | Purpose |
|---------|---------|
| **python-dotenv** | Loads environment variables from `.env` files |
| **tqdm** | Progress bars for long-running operations |

---

## External APIs & Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| **SEC EDGAR** | REST API | Quarterly 13F institutional holdings ZIPs (CIK, CUSIP, value, share counts) |
| **EODHD** | REST API | End-of-day stock prices, trading dates, company fundamentals |
| **Yahoo Finance** | yfinance library | Stock prices and Russell 3000 (^RUA) benchmark for backtesting |
| **QuantumOnline** | Web scraping | Supplementary CUSIP-to-ticker mapping |

---

## CI/CD & Deployment

| Tool | Purpose |
|------|---------|
| **GitHub Actions** | CI/CD pipeline — triggered on push to `main` |
| **GitHub Pages** | Hosts the mobile portfolio report (`portfolio_report_mobile.html`) as a static site |

### Deployment Flow
Push to `main` → GitHub Actions checkout → copy report → upload artifact → deploy to GitHub Pages.

---

## Development Tools

| Tool | Purpose |
|------|---------|
| **Git** | Version control |
| **Jupyter Notebooks** | Interactive analysis and model experimentation (30+ notebooks) |
| **VS Code** | IDE (`.vscode/` settings present) |
| **pip** | Python package management |

---

## Component × Technology Matrix

| Component | Languages | DB / Storage | Key Libraries | APIs |
|-----------|-----------|-------------|---------------|------|
| **ETL** | Python, SQL | PostgreSQL | requests, pandas, psycopg2 | SEC EDGAR |
| **Preprocessing** | Python | PostgreSQL, Parquet | pandas, polars, pdfplumber, lxml, beautifulsoup4, yfinance | EODHD, QuantumOnline |
| **Graph Models** | Python | PostgreSQL, Parquet | torch, torch-geometric, scikit-learn, lightgbm, networkx | — |
| **Baselines** | Python | Parquet | networkx, python-igraph, leidenalg, scikit-learn | — |
| **Portfolio** | Python, HTML | Parquet | pandas, yfinance, matplotlib | Yahoo Finance |
| **Deployment** | YAML | — | — | GitHub Pages |

---

## Environment Variables

```
# PostgreSQL
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# EODHD
EDOHD_API, EODHD_API_URL, EODHD_MAX_WORKERS, EODHD_RATE_LIMIT_SLEEP

# Web scraping
QUANTUMONLINE_URL, QUANTUMONLINE_WORKERS

# Pipeline paths
PIPELINE_INPUT_DIR, PIPELINE_OUTPUT_CSV
```
