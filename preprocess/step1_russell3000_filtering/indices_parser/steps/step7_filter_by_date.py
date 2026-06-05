"""Step 7: Filter out tickers with trading_end_date before minimum year."""

from pathlib import Path
from typing import Dict

import pandas as pd


def filter_by_trading_end_date(input_file: Path, output_file: Path, 
                                min_year: int = 2013) -> Dict:
    """
    Remove tickers that stopped trading before min_year.
    Keep tickers with no end date (still active) or end date >= min_year.
    """
    df = pd.read_csv(input_file)
    
    if 'trading_end_date' not in df.columns:
        df.to_csv(output_file, index=False)
        return {'skipped': True, 'reason': 'No trading_end_date column'}
    
    # Parse dates
    df['_end_parsed'] = pd.to_datetime(df['trading_end_date'], errors='coerce')
    min_date = pd.Timestamp(f'{min_year}-01-01')
    
    # Keep if: no end date OR end date >= min_year
    keep_mask = (
        df['_end_parsed'].isna() |
        (df['trading_end_date'] == '') |
        (df['_end_parsed'] >= min_date)
    )
    
    df_filtered = df[keep_mask].drop(columns=['_end_parsed'])
    df_filtered.to_csv(output_file, index=False)
    
    return {
        'initial': len(df),
        'final': len(df_filtered),
        'removed': len(df) - len(df_filtered),
        'min_year': min_year
    }

