"""Step 6: Determine trading periods using EODHD API."""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# Import configuration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from RawToFilteredtable.config import (
    EODHD_API_KEY, EODHD_API_URL, EODHD_MAX_WORKERS, 
    EODHD_RATE_LIMIT_SLEEP, EODHD_TIMEOUT
)

# Use config values
EODHD_URL = EODHD_API_URL
MAX_WORKERS = EODHD_MAX_WORKERS
RATE_LIMIT_SLEEP = EODHD_RATE_LIMIT_SLEEP


def determine_trading_periods(input_file: Path, api_key: str = None) -> Dict:
    """
    Query EODHD API to determine trading start/end dates for each ticker.
    Updates file in place with trading_status, trading_start_date, trading_end_date.
    """
    api_key = api_key or EODHD_API_KEY
    if not api_key:
        return {'skipped': True, 'reason': 'No API key'}
    
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    # Check if already processed
    if all(col in df.columns for col in ['trading_status', 'trading_start_date', 'trading_end_date']):
        return {'skipped': True, 'reason': 'Already processed'}
    
    ticker_col = _find_column(df, 'ticker')
    if not ticker_col:
        raise ValueError("'ticker' column not found")
    
    # Get unique tickers
    tickers = df[ticker_col].dropna().unique().tolist()
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    
    print(f"  Querying {len(tickers)} tickers...")
    results = _query_tickers_parallel(tickers, api_key)
    
    # Add columns to dataframe
    df['trading_status'] = df[ticker_col].apply(
        lambda x: results.get(str(x).strip().upper(), {}).get('status', 'unknown')
    )
    df['trading_start_date'] = df[ticker_col].apply(
        lambda x: results.get(str(x).strip().upper(), {}).get('start', '')
    )
    df['trading_end_date'] = df[ticker_col].apply(
        lambda x: results.get(str(x).strip().upper(), {}).get('end', '')
    )
    
    df.to_csv(input_file, index=False)
    
    active = (df['trading_status'] == 'active').sum()
    inactive = (df['trading_status'] == 'inactive').sum()
    
    return {'active': active, 'inactive': inactive, 'error': len(tickers) - active - inactive}


def _find_column(df: pd.DataFrame, name: str) -> Optional[str]:
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None


def _query_tickers_parallel(tickers: list, api_key: str) -> Dict[str, Dict]:
    """Query EODHD API in parallel for trading periods."""
    results = {}
    lock = threading.Lock()
    total = len(tickers)
    stats = {'completed': 0, 'active': 0, 'inactive': 0}
    
    def query_one(ticker: str):
        result = _query_ticker(ticker, api_key)
        with lock:
            results[ticker] = result
            stats['completed'] += 1
            if result['status'] == 'active':
                stats['active'] += 1
            elif result['status'] == 'inactive':
                stats['inactive'] += 1
            
            # Print progress every 25 items
            if stats['completed'] % 25 == 0 or stats['completed'] == total:
                pct = (stats['completed'] / total) * 100
                print(f"\r    Progress: {stats['completed']}/{total} ({pct:.1f}%) | Active: {stats['active']} | Inactive: {stats['inactive']}", 
                      end='', flush=True)
        return result
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(query_one, t) for t in tickers]
        for future in as_completed(futures):
            future.result()
    
    print()  # New line after progress
    return results


def _query_ticker(ticker: str, api_key: str) -> Dict:
    """Query single ticker for historical data."""
    try:
        url = EODHD_URL.format(ticker=ticker)
        response = requests.get(url, params={'api_token': api_key, 'fmt': 'json'}, timeout=EODHD_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return {'status': 'inactive', 'start': '', 'end': ''}
        
        dates = [datetime.strptime(r['date'], '%Y-%m-%d') for r in data if 'date' in r]
        if not dates:
            return {'status': 'inactive', 'start': '', 'end': ''}
        
        time.sleep(RATE_LIMIT_SLEEP)
        return {
            'status': 'active',
            'start': min(dates).strftime('%Y-%m-%d'),
            'end': max(dates).strftime('%Y-%m-%d')
        }
    except Exception:
        return {'status': 'error', 'start': '', 'end': ''}

