# ETL Pipeline — High-Level Overview

This pipeline ingests quarterly **SEC 13F portfolio holdings** filings, cleans and normalizes the data, and loads it into a partitioned PostgreSQL database.

---

## Directory Structure

```
ETL/
├── etl_pipeline.py          # Main entry point — orchestrates all three stages
├── dal/
│   └── dal.py               # Data Access Layer (thin wrapper over loader)
├── Extractors/
│   ├── base_strategy.py     # Abstract base class for all extraction strategies
│   ├── extractor_context.py # Strategy factory — maps type strings to classes
│   └── External/
│       └── sec_extraction_strategy.py  # SEC 13F extractor (main extractor)
├── manipulation/
│   └── manipulation.py      # 8-step data cleaning & transformation pipeline
├── load/
│   ├── load.py              # DataLoader wrapper
│   └── postgres_loader.py   # PostgreSQL-specific loader
├── data_handlers/
│   ├── web_data_fetcher.py  # Streaming HTTP download utility
│   └── db_data_handler/
│       ├── db_abstract.py       # Abstract DB handler interface
│       ├── postgres_handler.py  # PostgreSQL implementation (bulk COPY, partitioning)
│       ├── sql_db_handler.py    # SQLite implementation (legacy, unused)
│       └── create_table.sql     # Schema DDL with partition definitions
├── logger/
│   └── logger.py            # Singleton logger (file + console, auto-rotation)
├── utils/
│   ├── utils.py             # JSON loading, quarter-string parsing
│   └── csv_to_parquet.py    # Offline CSV → Parquet conversion utility
└── Data/
    ├── quarterly_datasets.json  # Maps quarters to SEC ZIP filenames (2017–2025)
    └── run.json                 # Specifies which quarters to process in a run
```

---

## Pipeline Stages

### Stage 1 — Extract

**Entry:** `etl_pipeline.py` reads the list of quarters from `Data/run.json`.

**What happens:**
- The `ExtractorContext` factory instantiates the `SecExtractionStrategy`.
- Up to **4 worker threads** download and process quarters in parallel.
- For each quarter:
  1. The quarterly ZIP is downloaded from the SEC website (cached locally in `13f_outputs/` to avoid re-downloading).
  2. The ZIP is extracted to a temporary directory.
  3. Two TSV files are parsed: `infotable.tsv` (individual holdings) and `submission.tsv` (filer metadata).
  4. The two tables are **inner-joined on `ACCESSION_NUMBER`**.
  5. An optional CIK filter is applied to restrict to specific firms.
- All quarter results are concatenated into a single DataFrame.

**Output:** A raw DataFrame containing holdings + filer metadata for all requested quarters.

---

### Stage 2 — Manipulation (Transform)

**Entry:** `manipulation/manipulation.py`

The transformation runs 8 steps in a fixed sequence:

| Step | Action |
|------|--------|
| 1 | Lowercase all column names |
| 2 | Strip underscores from column names |
| 3 | Drop 12 irrelevant columns (e.g., `titleofclass`, `figi`, voting/discretion columns) |
| 4 | Clean data: remove put/call rows, strip CUSIP whitespace, drop duplicates, drop null `cusip`/`value` rows |
| 5 | Filter: keep only records from **2013 Q2 onwards** |
| 6 | Add computed fields: `value_per_share = value / sshprnamt`, `is_complete` flag (≥80% non-null columns) |
| 7 | Parse `periodofreport` → split into integer `year` and `quarter` columns |
| 8 | Replace `value_per_share` with the **median per (year, quarter)** group to normalize outliers; recalculate `value = value_per_share × sshprnamt` |

**Output:** A clean, normalized DataFrame ready for loading.

---

### Stage 3 — Load

**Entry:** `load/load.py` → `load/postgres_loader.py` → `data_handlers/db_data_handler/postgres_handler.py`

**What happens:**
1. Connect to PostgreSQL using environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).
2. Ensure the parent table `holdings` exists with **RANGE partitioning** on the `period_start` DATE column.
3. For each `(year, quarter)` group in the DataFrame:
   - Dynamically create the quarterly partition if it doesn't exist (e.g., `holdings_2024_q1`).
   - Bulk-insert data using the PostgreSQL **`COPY` command** (via an in-memory CSV buffer) for maximum throughput.
4. Commit the transaction.

**Output:** Data is persisted in quarterly partitioned tables in PostgreSQL.

---

## Database Schema

```
holdings  (parent, RANGE partitioned by period_start)
├── accessionnumber   TEXT
├── infotablesk       TEXT
├── nameofissuer      TEXT
├── cusip             TEXT      -- indexed
├── value             NUMERIC
├── sshprnamt         NUMERIC
├── filingdate        DATE      -- indexed
├── cik               TEXT      -- indexed
├── value_per_share   NUMERIC
├── year              INTEGER
├── quarter           INTEGER
└── period_start      DATE      -- partition key

Partitions: holdings_2014_q1, holdings_2014_q2, ... (created automatically)
```

---

## Configuration

| File | Purpose |
|------|---------|
| `Data/quarterly_datasets.json` | Maps each quarter (e.g., `2024_Q1`) to the corresponding SEC ZIP filename |
| `Data/run.json` | List of quarters to process in the current run |
| `.env` | PostgreSQL connection credentials (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) |

---

## Key Design Decisions

- **Strategy Pattern** — The extractor layer is designed for extensibility. The context maps type keys (`"sec"`, `"csv"`, `"xml"`) to strategy classes, making it straightforward to add new data sources.
- **Parallel extraction** — Multi-threaded download and parsing (4 workers) with thread-safe result collection via `Lock`.
- **Bulk loading** — PostgreSQL `COPY` via an in-memory CSV buffer is significantly faster than row-by-row `INSERT`.
- **Automatic partitioning** — Quarterly partitions are created on demand; no manual schema maintenance is needed when adding new quarters.
- **Median normalization** — `value_per_share` is replaced with the quarterly group median to reduce the effect of data entry errors and outliers in the SEC source data.
- **Local ZIP cache** — Downloaded ZIPs are stored in `13f_outputs/` so re-runs of the same quarters skip the download step.
- **Singleton logger** — A single logger instance is shared across all modules, with automatic log file rotation (keeps the 2 most recent log files).
