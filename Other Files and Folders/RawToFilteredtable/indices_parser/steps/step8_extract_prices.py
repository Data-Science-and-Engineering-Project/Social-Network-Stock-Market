"""Step 8: Extract quarter-end closing prices using EODHD API."""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import pandas_market_calendars as mcal
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from RawToFilteredtable.config import (
    EODHD_API_KEY, EODHD_API_URL, EODHD_MAX_WORKERS, 
    EODHD_RATE_LIMIT_SLEEP, EODHD_TIMEOUT, TABLE_TICKERPRICES,
    DEFAULT_PRICE_START_QUARTER, DEFAULT_PRICE_END_QUARTER
)

# Use config values
EODHD_URL = EODHD_API_URL
MAX_WORKERS = EODHD_MAX_WORKERS
RATE_LIMIT_SLEEP = EODHD_RATE_LIMIT_SLEEP


def extract_quarter_end_prices(input_file: Path, 
                                table_name: str = None,
                                start_quarter: str = None, 
                                end_quarter: str = None,
                                api_key: str = None) -> Dict:
    """
    Extract quarter-end prices for all tickers and insert into database.
    """
    table_name = table_name or TABLE_TICKERPRICES
    start_quarter = start_quarter or DEFAULT_PRICE_START_QUARTER
    end_quarter = end_quarter or DEFAULT_PRICE_END_QUARTER
    
    api_key = api_key or EODHD_API_KEY
    if not api_key:
        return {'skipped': True, 'reason': 'No API key'}
    
    # Build quarter-to-trading-day mapping (once)
    quarter_mapping = _build_quarter_mapping(start_quarter, end_quarter)
    dates_with_quarters = sorted(
        [(td, ps) for qe, (td, ps) in quarter_mapping.items()],
        key=lambda x: x[0]
    )
    
    # Load tickers
    df = pd.read_csv(input_file)
    ticker_col = next((c for c in df.columns if c.lower() == 'ticker'), None)
    if not ticker_col:
        raise ValueError("'ticker' column not found")
    
    tickers_info = _extract_tickers_info(df, ticker_col)
    print(f"  Processing {len(tickers_info)} tickers...")
    
    # Prepare the table ONCE before parallel processing
    print(f"  Preparing table '{table_name}'...")
    try:
        handler = PostgresHandler()
        handler.connect()
        handler.drop_table(table_name)
        # Create table with sample schema
        sample_df = pd.DataFrame([{
            'ticker': 'SAMPLE', 
            'date': datetime.now().date(), 
            'price': 0.0, 
            'period_start': datetime.now().date()
        }])
        handler.create_table(table_name, sample_df)
        # Delete the sample row
        cursor = handler.connection.cursor()
        cursor.execute(f'DELETE FROM "{table_name}" WHERE ticker = %s', ('SAMPLE',))
        handler.connection.commit()
        cursor.close()
        handler.disconnect()
        print(f"  Table '{table_name}' ready")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not prepare table: {e}")
    
    # Process in parallel
    stats = {'processed': 0, 'inserted': 0}
    no_prices = set()
    lock = threading.Lock()
    
    def process_ticker(info: Dict) -> int:
        ticker = info['ticker']
        
        # Filter dates for this ticker's trading period
        relevant_dates = _filter_dates(dates_with_quarters, info['start'], info['end'])
        if not relevant_dates:
            with lock:
                no_prices.add(ticker)
                stats['processed'] += 1
            return 0
        
        # Fetch prices (single API call)
        prices = _fetch_prices(ticker, relevant_dates, api_key)
        if not prices:
            with lock:
                no_prices.add(ticker)
                stats['processed'] += 1
            return 0
        
        # Insert to database (append mode - table already exists)
        try:
            handler = PostgresHandler()
            handler.connect()
            inserted = handler.insert_dataframe_regular(pd.DataFrame(prices), table_name, if_exists='append')
            handler.disconnect()
        except Exception as e:
            inserted = 0
        
        with lock:
            stats['processed'] += 1
            stats['inserted'] += inserted
            
            # Print progress every 25 items
            if stats['processed'] % 25 == 0 or stats['processed'] == len(tickers_info):
                pct = (stats['processed'] / len(tickers_info)) * 100
                print(f"\r    Progress: {stats['processed']}/{len(tickers_info)} ({pct:.1f}%) | Inserted: {stats['inserted']} prices", 
                      end='', flush=True)
        
        return inserted
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, info) for info in tickers_info]
        for future in as_completed(futures):
            future.result()
    
    print()
    
    # Save tickers with no prices
    if no_prices:
        output_file = input_file.parent / 'tickers_no_prices.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(sorted(no_prices)))
        print(f"  Saved {len(no_prices)} tickers with no prices to {output_file}")
    
    return {
        'processed': stats['processed'],
        'inserted': stats['inserted'],
        'no_prices': len(no_prices)
    }


def _build_quarter_mapping(start: str, end: str) -> Dict[datetime, Tuple[datetime, datetime]]:
    """Build quarter-end -> (trading day, period_start) mapping."""
    quarter_ends = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}
    quarter_starts = {1: '01-01', 2: '04-01', 3: '07-01', 4: '10-01'}
    
    start_q, start_y = int(start[1]), int(start.split('_')[1])
    end_q, end_y = int(end[1]), int(end.split('_')[1])
    
    # Generate quarter dates (end date and start date)
    dates = []
    year, q = start_y, start_q
    while (year, q) <= (end_y, end_q):
        qe = datetime.strptime(f"{year}-{quarter_ends[q]}", '%Y-%m-%d')
        qs = datetime.strptime(f"{year}-{quarter_starts[q]}", '%Y-%m-%d')
        dates.append((qe, qs))
        q += 1
        if q > 4:
            q, year = 1, year + 1
    
    # Get NYSE trading days
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=min(d[0] for d in dates), 
                              end_date=max(d[0] for d in dates) + pd.Timedelta(days=10))
    trading_days = set(schedule.index)
    
    # Map to trading days (keep period_start)
    mapping = {}
    for qdate, period_start in dates:
        ts = pd.Timestamp(qdate)
        if ts in trading_days:
            mapping[qdate] = (qdate, period_start)
        else:
            future = sorted(d for d in trading_days if d > ts)
            if future:
                mapping[qdate] = (future[0].to_pydatetime(), period_start)
    
    return mapping


def _extract_tickers_info(df: pd.DataFrame, ticker_col: str) -> List[Dict]:
    """Extract unique tickers with their trading periods."""
    info = []
    for _, row in df.drop_duplicates(subset=[ticker_col]).iterrows():
        ticker = str(row[ticker_col]).strip().upper()
        if ticker and ticker != 'NAN':
            info.append({
                'ticker': ticker,
                'start': row.get('trading_start_date') if pd.notna(row.get('trading_start_date')) else None,
                'end': row.get('trading_end_date') if pd.notna(row.get('trading_end_date')) else None
            })
    return info


def _filter_dates(dates: List[Tuple], start: str, end: str) -> List[Tuple]:
    """Filter dates to ticker's trading period."""
    if not start and not end:
        return dates
    
    start_dt = datetime.strptime(start, '%Y-%m-%d') if start else datetime(1900, 1, 1)
    end_dt = datetime.strptime(end, '%Y-%m-%d') if end else datetime(2100, 1, 1)
    
    return [(td, qe) for td, qe in dates if start_dt <= td <= end_dt]


def _fetch_prices(ticker: str, dates: List[Tuple], api_key: str) -> List[Dict]:
    """Fetch prices for ticker on specific dates."""
    trading_dates = [d[0] for d in dates]
    min_date = min(trading_dates) - pd.Timedelta(days=5)
    max_date = max(trading_dates) + pd.Timedelta(days=5)
    
    try:
        response = requests.get(
            EODHD_URL.format(ticker=ticker),
            params={
                'api_token': api_key,
                'fmt': 'json',
                'period': 'd',
                'from': min_date.strftime('%Y-%m-%d'),
                'to': max_date.strftime('%Y-%m-%d')
            },
            timeout=EODHD_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    
    if not data:
        return []
    
    # Build lookup
    lookup = {r['date']: float(r.get('adjusted_close') or r.get('close'))
              for r in data if 'date' in r and (r.get('adjusted_close') or r.get('close'))}
    
    time.sleep(RATE_LIMIT_SLEEP)
    
    return [
        {'ticker': ticker, 'date': td.date(), 'price': lookup[td.strftime('%Y-%m-%d')], 'period_start': ps.date()}
        for td, ps in dates if td.strftime('%Y-%m-%d') in lookup
    ]

