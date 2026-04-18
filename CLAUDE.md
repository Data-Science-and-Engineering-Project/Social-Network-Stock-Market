# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Social Network Stock Market**: A comprehensive system for analyzing stock market portfolios through a social network lens, combining SEC 13F portfolio data with network analysis and machine learning models.

### Core Components

1. **ETL Pipeline** (`ETL/`): Extracts portfolio data from SEC Edgar, transforms it, and loads into PostgreSQL with automatic partition management.
2. **Data Science Models** (`DS Models/`): Feature engineering, model training, and testing for portfolio analysis.
3. **Social Network** (`Social Network/`): Network analysis using Russell 3000 ETF holdings and sector relationships.
4. **Notebooks** (`Notebooks/`): Jupyter notebooks for exploratory analysis and model development (LightGCN recommendation models, data analysis).

## Setup & Development

### Environment Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (`.env` file):
   - Database: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
   - APIs: `EDOHD_API`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
   - Database env vars also available as: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`

### Key Commands

**Run ETL pipeline**:
```bash
python ETL/etl_pipeline.py
```
Reads from `data/run.json` with quarters configuration, executes extract → manipulate → load pipeline, logs to `logs/` directory.

**Run Jupyter notebooks**:
```bash
jupyter notebook
```
Key notebooks: `Notebooks/LightGCN/LightGCN.ipynb`, `Notebooks/data_analysis/data_analysis.ipynb`

**Code quality**:
```bash
black .                    # Format code
pylint ETL/               # Lint check
isort .                   # Sort imports
ruff check .              # Fast linting
```

**Testing**:
```bash
pytest ETL/tests/         # Run ETL tests
```

## Architecture

### ETL Pipeline Architecture (`ETL/`)

**Data Flow**: Extract → Manipulate → Load

- **Extractors** (`ETL/Extractors/`): Strategy pattern implementation
  - `ExtractorContext`: Routes extraction based on type (e.g., "sec" for SEC Edgar)
  - Loads quarterly 13F portfolio data from SEC
  
- **Data Handlers** (`ETL/data_handlers/`)
  - `PostgresHandler`: Manages PostgreSQL connections with partition support
  - `db_abstract.py`: Base class defining interface for database handlers
  - Handles credential loading from environment

- **DAL** (`ETL/dal/dal.py`): Data access layer
  - `DAL.load_data()`: Main entry point for loading transformed data
  - Handles automatic partition creation and median calculations

- **Manipulation** (`ETL/manipulation/manipulation.py`): Data transformation
  - `DataManipulation` class orchestrates cleaning/transformation
  - Customizable transformation rules

- **Load** (`ETL/load/load.py`, `postgres_loader.py`): Database insertion
  - Batch operations with `psycopg2.execute_batch`
  - Partition-aware loading

- **Logging** (`ETL/logger/logger.py`): Custom ETL logger
  - Console and file output
  - Structured logging with timestamps
  - Log files saved to `logs/` directory

### Data Science Models (`DS Models/`)

- **Features** (`features/`): Feature engineering and extraction
- **Training** (`training/`): Model training pipelines
- **Testing** (`test/`): Model evaluation and testing

### Social Network Analysis (`Social Network/`)

- **Network pipeline** (`src/network-pipeline.ipynb`): Main network analysis workflow
- **Russell 3000 filtering** (`russell/`): ETF holdings analysis and CUSIP selection
- References: `Holdings_details_Russell_3000_ETF.csv`, `SPY.csv`

### Notebooks (`Notebooks/`)

- **LightGCN models**: 
  - `LightGCN.ipynb`: Graph Convolutional Network for portfolio recommendations
  - `RobustLightGCN.ipynb`: Robustness improvements
- **Data analysis**: `data_analysis/data_analysis.ipynb` - Exploratory data analysis
- **Test data**: CSV files with quarterly portfolio data (2018 Q1-Q4)

## Key Dependencies

- **Data Processing**: pandas, numpy, pyarrow
- **Database**: psycopg2 (PostgreSQL)
- **SEC Data**: edgar, edgartools
- **ML/DL**: torch, scikit-learn (in DS Models)
- **Visualization**: matplotlib, seaborn
- **APIs**: openai, anthropic libraries
- **Code Quality**: black, pylint, isort, ruff
- **Testing**: pytest
- **Utilities**: python-dotenv, beautifulsoup4, requests

## Database Schema & Partitioning

PostgreSQL database (`Social_13F`) contains portfolio data with:
- **Automatic partitioning**: Tables are partitioned by year
- **Median calculations**: Automatic statistics computed during load
- Configured via environment variables
- Connection pooling and batch operations for performance

## Git Workflow

- Main branch: `main`
- Current development branch: `russell3000_filtering`
- Always commit with descriptive messages
- Keep requirements.txt updated when adding dependencies

## Common Development Tasks

### Adding a new data extractor
1. Create strategy class in `ETL/Extractors/`
2. Implement extraction logic
3. Register in `ExtractorContext`
4. Add unit test in `ETL/tests/`

### Modifying database schema
1. Update `DAL` and `PostgresHandler` if needed
2. Test with debug mode enabled in `etl_pipeline.py`
3. Verify partition handling

### Running single ETL job for testing
Set `debug_mode = True` in `ETL/etl_pipeline.py` to limit to first 20 records and first quarter in `data/run.json`.

## Important Notes

- SEC Edgar data fetching respects rate limits; avoid rapid re-requests
- All credentials and sensitive API keys are in `.env` - never commit these
- Database partitions are created automatically; verify in PostgreSQL admin
- ETL logs are detailed and timestamped; check `logs/` for troubleshooting
- Notebook outputs (especially LightGCN training) can be memory-intensive on large datasets
