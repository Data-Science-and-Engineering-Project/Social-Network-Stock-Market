"""
Configuration module for RawToFilteredtable pipeline.
Loads all configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment - try multiple locations
for env_path in [
    Path(__file__).parent.parent / '.env',  # Project root
    Path(__file__).parent / '.env',  # RawToFilteredtable
    Path.cwd() / '.env',  # Current directory
]:
    if env_path.exists():
        load_dotenv(env_path)
        break

# ==================== DATABASE CONFIGURATION ====================

# Database connection (uses same as ETL pipeline)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Database table names
TABLE_HOLDINGS_NEW = os.getenv("TABLE_HOLDINGS_NEW", "holdings")
TABLE_TICKERPRICES = os.getenv("TABLE_TICKERPRICES", "ticker_prices")
TABLE_TICKER_TO_CUSIP = os.getenv("TABLE_TICKER_TO_CUSIP", "ticker_to_cusip")
TABLE_HOLDINGS_FILTERED = os.getenv("TABLE_HOLDINGS_FILTERED", "holdings_filtered_new")

# ==================== API CONFIGURATION ====================

# EODHD API
EODHD_API_KEY = os.getenv("EDOHD_API")
EODHD_API_URL = os.getenv("EODHD_API_URL", "https://eodhd.com/api/eod/{ticker}.US")
EODHD_MAX_WORKERS = int(os.getenv("EODHD_MAX_WORKERS", "10"))
EODHD_RATE_LIMIT_SLEEP = float(os.getenv("EODHD_RATE_LIMIT_SLEEP", "0.06"))
EODHD_TIMEOUT = int(os.getenv("EODHD_TIMEOUT", "30"))

# QuantumOnline API
QUANTUMONLINE_URL = os.getenv("QUANTUMONLINE_URL", "https://www.quantumonline.com")
QUANTUMONLINE_WORKERS = int(os.getenv("QUANTUMONLINE_WORKERS", "12"))
QUANTUMONLINE_SLEEP = float(os.getenv("QUANTUMONLINE_SLEEP", "0.2"))
QUANTUMONLINE_TIMEOUT = int(os.getenv("QUANTUMONLINE_TIMEOUT", "30000"))

# ==================== FILE PATHS ====================

# Default input/output paths
DEFAULT_INPUT_DIR = os.getenv("PIPELINE_INPUT_DIR", "Data/Indexes/RUSSELL3000 HISTORY")
DEFAULT_OUTPUT_CSV = os.getenv("PIPELINE_OUTPUT_CSV", "Data/Indexes/RUSSELL3000 HISTORY/russell3000_final.csv")
DEFAULT_FILTERED_CSV = os.getenv("PIPELINE_FILTERED_CSV", "Data/Indexes/RUSSELL3000 HISTORY/russell3000_filtered.csv")
DEFAULT_REFERENCE_FILE = os.getenv("PIPELINE_REFERENCE_FILE", "Data/Indexes/Holdings_details_Russell_3000_ETF.csv")
DEFAULT_LOG_FILE = os.getenv("PIPELINE_LOG_FILE", "RawToFilteredtable/pipeline_log.txt")

# ==================== QUARTER CONFIGURATION ====================

DEFAULT_PRICE_START_QUARTER = os.getenv("PIPELINE_PRICE_START", "Q2_2013")
DEFAULT_PRICE_END_QUARTER = os.getenv("PIPELINE_PRICE_END", "Q2_2025")
DEFAULT_FILTER_START_QUARTER = os.getenv("PIPELINE_FILTER_START", "Q2_2013")
DEFAULT_FILTER_END_QUARTER = os.getenv("PIPELINE_FILTER_END", "Q2_2025")

# ==================== VALIDATION ====================

def validate_config():
    """Validate that required configuration is present."""
    errors = []
    
    if not DB_NAME:
        errors.append("DB_NAME environment variable is required")
    if not DB_USER:
        errors.append("DB_USER environment variable is required")
    if not DB_PASSWORD:
        errors.append("DB_PASSWORD environment variable is required")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

