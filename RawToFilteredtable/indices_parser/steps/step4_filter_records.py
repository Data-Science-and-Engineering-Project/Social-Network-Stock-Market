"""Step 4: Filter out records with empty CUSIP or TICKER."""

from pathlib import Path
from typing import Dict

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from indices_parser.utils import is_empty_cusip


def filter_invalid_records(input_file: Path, valid_file: Path, 
                           filtered_file: Path) -> Dict:
    """
    Filter records: keep only those with valid CUSIP and TICKER.
    Save invalid records to separate file.
    """
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    cusip_col = _find_column(df, 'cusip')
    ticker_col = _find_column(df, 'ticker')
    
    if not cusip_col or not ticker_col:
        raise ValueError("Missing required columns: cusip, ticker")
    
    # Valid: both CUSIP and TICKER are filled
    valid_mask = (
        ~df[cusip_col].apply(is_empty_cusip) &
        df[ticker_col].notna() &
        (df[ticker_col] != '')
    )
    
    df_valid = df[valid_mask].copy()
    df_filtered = df[~valid_mask].copy()
    
    # Save files
    df_valid.to_csv(valid_file, index=False)
    if len(df_filtered) > 0:
        df_filtered.to_csv(filtered_file, index=False)
    
    return {
        'initial': len(df),
        'valid': len(df_valid),
        'filtered': len(df_filtered),
        'empty_cusip': df[cusip_col].apply(is_empty_cusip).sum(),
        'empty_ticker': (df[ticker_col].isna() | (df[ticker_col] == '')).sum()
    }


def _find_column(df: pd.DataFrame, name: str) -> str:
    """Find column case-insensitively."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None

