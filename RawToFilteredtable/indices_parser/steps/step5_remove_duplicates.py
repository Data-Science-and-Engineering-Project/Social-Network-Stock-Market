"""Step 5: Remove duplicate records (keep oldest year)."""

from pathlib import Path
from typing import Dict

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from indices_parser.utils import is_empty_cusip


def remove_duplicate_records(input_file: Path, output_file: Path) -> Dict:
    """
    Remove duplicates by CUSIP and TICKER, keeping the oldest year.
    """
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    cusip_col = _find_column(df, 'cusip')
    ticker_col = _find_column(df, 'ticker')
    year_col = _find_column(df, 'year')
    
    if not year_col:
        raise ValueError("'year' column required for deduplication")
    
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    
    to_remove = set()
    
    # Remove by CUSIP (keep oldest)
    if cusip_col:
        cusip_dupes = _find_duplicates(df, cusip_col, year_col, is_empty_cusip)
        to_remove.update(cusip_dupes)
    
    # Remove by TICKER (keep oldest)
    if ticker_col:
        ticker_dupes = _find_duplicates(df, ticker_col, year_col, 
                                        lambda x: pd.isna(x) or x == '')
        to_remove.update(ticker_dupes)
    
    df_clean = df[~df.index.isin(to_remove)].copy()
    df_clean.to_csv(output_file, index=False)
    
    return {
        'initial': len(df),
        'final': len(df_clean),
        'removed': len(to_remove)
    }


def _find_column(df: pd.DataFrame, name: str) -> str:
    """Find column case-insensitively."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None


def _find_duplicates(df: pd.DataFrame, group_col: str, year_col: str,
                     is_empty_fn) -> set:
    """Find duplicate indices to remove (keep oldest year)."""
    to_remove = set()
    
    df_valid = df[~df[group_col].apply(is_empty_fn)]
    
    for _, group in df_valid.groupby(group_col):
        if len(group) > 1:
            sorted_group = group.sort_values(year_col, ascending=True)
            to_remove.update(sorted_group.index[1:].tolist())
    
    return to_remove

