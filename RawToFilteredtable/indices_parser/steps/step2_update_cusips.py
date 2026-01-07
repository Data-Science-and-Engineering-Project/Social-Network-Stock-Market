"""Step 2: Update empty CUSIPs based on company name from database."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler
from indices_parser.utils import is_empty_cusip, get_first_n_words, year_to_quarter_end


def update_cusips_by_name(input_file: Path, table_name: str) -> Dict:
    """
    Update empty CUSIPs by searching company name in database.
    
    Returns dict with statistics.
    """
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    cusip_col = _find_column(df, 'cusip')
    name_col = _find_column(df, 'name')
    year_col = _find_column(df, 'year')
    
    if not cusip_col or not name_col:
        raise ValueError("Missing required columns: cusip, name")
    
    # Filter rows with empty CUSIPs
    empty_mask = df[cusip_col].apply(is_empty_cusip)
    indices_to_update = df[empty_mask].index.tolist()
    
    if not indices_to_update:
        return {'updated': 0, 'not_found': 0, 'empty_before': 0, 'empty_after': 0}
    
    empty_before = empty_mask.sum()
    
    # Connect and query
    handler = PostgresHandler()
    if not handler.connect():
        raise Exception("Failed to connect to database")
    
    try:
        # Prepare search params
        search_params = _prepare_search_params(df, indices_to_update, name_col, year_col)
        
        # Batch query
        results = _batch_find_cusips(handler, search_params, table_name)
        
        # Apply updates
        updated = 0
        for idx, (name_search, period_start) in zip(indices_to_update, search_params):
            cusip = results.get((name_search, period_start), '')
            if cusip:
                df.at[idx, cusip_col] = cusip
                updated += 1
        
        df.to_csv(input_file, index=False)
        empty_after = df[cusip_col].apply(is_empty_cusip).sum()
        
        return {
            'updated': updated,
            'not_found': len(indices_to_update) - updated,
            'empty_before': empty_before,
            'empty_after': empty_after
        }
    finally:
        handler.disconnect()


def _find_column(df: pd.DataFrame, name: str) -> str:
    """Find column case-insensitively."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None



def _prepare_search_params(df: pd.DataFrame, indices: List[int], 
                           name_col: str, year_col: str) -> List[Tuple[str, datetime.date, datetime.date]]:
    """Prepare (name_search, period_start) tuples for querying with partition pruning."""
    params = []
    for idx in indices:
        name = get_first_n_words(df.at[idx, name_col], 3)
        year = int(df.at[idx, year_col]) if year_col else 2025
        if not year == 2025:
            year = year + 1 
        # Get quarter from dataframe if available, otherwise default to Q2
        # Calculate period_start and period_end for partition pruning
        start_month = 7
        period_start = datetime(year, start_month, 1).date()
        
        
        params.append((name, period_start))
    return params


def _batch_find_cusips(handler: PostgresHandler, 
                       search_params: List[Tuple[str, datetime.date, datetime.date]], 
                       table_name: str) -> Dict[Tuple[str, datetime.date, datetime.date], str]:
    """Query database for CUSIPs matching company name patterns using partition pruning."""
    results = {}
    total = len(search_params)
    found = 0
    
    # Use period_start for partition pruning instead of quarter_end
    query = f"""
        SELECT cusip, COUNT(*) as cnt
        FROM {table_name}
        WHERE nameofissuer LIKE %s AND period_start = %s 
        GROUP BY cusip ORDER BY cnt DESC LIMIT 1
    """
    
    print(f"Searching {total} company names in database...")
    
    with handler.connection.cursor() as cursor:
        for i, (name_search, period_start) in enumerate(search_params, 1):
            if not name_search:
                continue
            try:
                pattern = f"%{name_search.replace(chr(39), chr(39)*2)}%"
                cursor.execute(query, (pattern, period_start))
                row = cursor.fetchone()
                if row and row[0]:
                    results[(name_search, period_start)] = str(row[0]).strip()
                    found += 1
            except Exception:
                pass
            
            # Print progress every 100 items or at milestones
            if i % 100 == 0 or i == total:
                pct = (i / total) * 100
                print(f"\r    Progress: {i}/{total} ({pct:.1f}%) | Found: {found}", end='', flush=True)
    
    print()  # New line after progress
    return results

