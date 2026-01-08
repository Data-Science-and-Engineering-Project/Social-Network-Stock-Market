#!/usr/bin/env python
"""
Filter Holdings Pipeline

This pipeline:
1. Loads a CSV file with ticker-to-CUSIP mappings into a mapping table
2. For each quarter (configurable via env vars):
   - Loads holdings from source table (default: holdings)
   - Filters to keep only CUSIPs that exist in mapping table
   - Inserts filtered data into target table (default: holdings_filtered)
   - Uses period_start for partition pruning (optimized queries)

All table names and quarter ranges are configurable via environment variables.
See RawToFilteredtable/config.py for details.

Note: Queries use period_start with date ranges for optimal partition pruning.

Usage:
    python filterholdings/filter_holdings_pipeline.py --input <cusip_csv>
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler

# Import config - use absolute import to avoid circular import
import sys
from pathlib import Path
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RawToFilteredtable.config import (
    TABLE_HOLDINGS_NEW, TABLE_TICKER_TO_CUSIP, TABLE_HOLDINGS_FILTERED,
    DEFAULT_FILTER_START_QUARTER, DEFAULT_FILTER_END_QUARTER
)


# Quarter configuration - for period_start (beginning of quarter)
QUARTER_STARTS = {
    1: 1,   # Q1 starts in January
    2: 4,   # Q2 starts in April
    3: 7,   # Q3 starts in July
    4: 10   # Q4 starts in October
}


def generate_quarters(start: str = 'Q2_2013', end: str = 'Q2_2025') -> list:
    """Generate list of quarters with period_start and period_end for partition pruning."""
    start_q, start_y = int(start[1]), int(start.split('_')[1])
    end_q, end_y = int(end[1]), int(end.split('_')[1])
    
    quarters = []
    year, q = start_y, start_q
    while (year, q) <= (end_y, end_q):
        # Calculate period_start (beginning of quarter)
        start_month = QUARTER_STARTS[q]
        period_start = datetime(year, start_month, 1).date()
        
        # Calculate period_end (beginning of next quarter, exclusive)
        if q == 4:
            period_end = datetime(year + 1, 1, 1).date()
        else:
            period_end = datetime(year, start_month + 3, 1).date()
        
        quarters.append({
            'quarter': f"Q{q}_{year}",
            'period_start': period_start,
            'period_end': period_end,
            'year': year,
            'quarter_num': q
        })
        q += 1
        if q > 4:
            q = 1
            year += 1
    
    return quarters


def step1_load_ticker_to_cusip(input_file: Path, handler: PostgresHandler, 
                                table_name: str = None) -> dict:
    """
    Step 1: Load CSV file into TickerToCusip table.
    
    Args:
        input_file: Path to CSV with ticker-CUSIP mappings
        handler: Database handler
        table_name: Table name (defaults to config)
    
    Returns:
        Stats dict
    """
    table_name = table_name or TABLE_TICKER_TO_CUSIP
    
    print("\n" + "=" * 80)
    print("STEP 1: LOADING TICKER-TO-CUSIP MAPPING")
    print("=" * 80)
    
    # Load CSV
    print(f"\n📂 Loading CSV: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows")
    
    # Find CUSIP column
    cusip_col = next((c for c in df.columns if c.lower() == 'cusip'), None)
    ticker_col = next((c for c in df.columns if c.lower() == 'ticker'), None)
    
    if not cusip_col:
        print("❌ Error: 'cusip' column not found")
        return {'success': False, 'error': 'cusip column not found'}
    
    print(f"  CUSIP column: {cusip_col}")
    print(f"  TICKER column: {ticker_col or 'not found'}")
    
    # Create table
    print(f"\n📊 Creating {table_name} table...")
    
    create_table_sql = f"""
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        name VARCHAR(255) NOT NULL,
        cusip VARCHAR(20) NOT NULL,
        ticker VARCHAR(10) NOT NULL,
        trading_start_date DATE,
        trading_end_date DATE,
        PRIMARY KEY (cusip, ticker)
    );
    """
    
    try:
        cursor = handler.connection.cursor()
        cursor.execute(create_table_sql)
        handler.connection.commit()
        cursor.close()
        print("  ✅ Table created")
    except Exception as e:
        print(f"  ❌ Error creating table: {e}")
        return {'success': False, 'error': str(e)}
    
    # Find additional columns
    name_col = next((c for c in df.columns if c.lower() in ['name', 'nameofissuer', 'company']), None)
    
    # Prepare data for insert
    df_insert = pd.DataFrame()
    df_insert['name'] = df[name_col] if name_col else ''
    df_insert['cusip'] = df[cusip_col]
    df_insert['ticker'] = df[ticker_col] if ticker_col else ''
    
    # Add trading dates if available
    if 'trading_start_date' in df.columns:
        df_insert['trading_start_date'] = pd.to_datetime(df['trading_start_date'], errors='coerce').dt.date
    else:
        df_insert['trading_start_date'] = None
        
    if 'trading_end_date' in df.columns:
        df_insert['trading_end_date'] = pd.to_datetime(df['trading_end_date'], errors='coerce').dt.date
    else:
        df_insert['trading_end_date'] = None
    
    # Remove duplicates by (cusip, ticker)
    df_insert = df_insert.drop_duplicates(subset=['cusip', 'ticker'])
    print(f"  Unique records: {len(df_insert):,}")
    
    # Insert data
    print("\n📥 Inserting data...")
    try:
        inserted = handler.insert_dataframe_regular(df_insert, table_name, if_exists='replace')
        print(f"  ✅ Inserted {inserted:,} rows")
    except Exception as e:
        print(f"  ❌ Error inserting data: {e}")
        return {'success': False, 'error': str(e)}
    
    return {
        'success': True,
        'rows_loaded': len(df),
        'unique_records': len(df_insert),
        'rows_inserted': inserted
    }


def step2_filter_holdings_by_quarter(handler: PostgresHandler, 
                                      start_quarter: str = None,
                                      end_quarter: str = None,
                                      source_table: str = None,
                                      mapping_table: str = None,
                                      target_table: str = None) -> dict:
    """
    Step 2: Filter holdings for each quarter.
    
    For each quarter:
    - Load data from source_table
    - Keep only rows where CUSIP exists in mapping_table
    - Insert into target_table
    
    Args:
        handler: Database handler
        start_quarter: Start quarter (defaults to config)
        end_quarter: End quarter (defaults to config)
        source_table: Source table name (defaults to config)
        mapping_table: Mapping table name (defaults to config)
        target_table: Target table name (defaults to config)
    
    Returns:
        Stats dict
    """
    start_quarter = start_quarter or DEFAULT_FILTER_START_QUARTER
    end_quarter = end_quarter or DEFAULT_FILTER_END_QUARTER
    source_table = source_table or TABLE_HOLDINGS_NEW
    mapping_table = mapping_table or TABLE_TICKER_TO_CUSIP
    target_table = target_table or TABLE_HOLDINGS_FILTERED
    
    print("\n" + "=" * 80)
    print("STEP 2: FILTERING HOLDINGS BY QUARTER")
    print("=" * 80)
    
    quarters = generate_quarters(start_quarter, end_quarter)
    print(f"\n📅 Processing {len(quarters)} quarters ({start_quarter} to {end_quarter})")
    
    # Ensure target table exists (without partitions for simplicity)
    print(f"\n📊 Preparing {target_table} table...")
    
    create_table_sql = f"""
    DROP TABLE IF EXISTS {target_table};
    CREATE TABLE {target_table} (
        nameofissuer TEXT,
        cusip TEXT,
        sshprnamt DOUBLE PRECISION,
        cik TEXT,
        year INT,
        quarter INT,
        period_start DATE NOT NULL
    );
    CREATE INDEX idx_{target_table}_period_start ON {target_table}(period_start);
    CREATE INDEX idx_{target_table}_cusip ON {target_table}(cusip);
    """
    
    try:
        cursor = handler.connection.cursor()
        cursor.execute(create_table_sql)
        handler.connection.commit()
        cursor.close()
        print("  ✅ Table created with indexes")
    except Exception as e:
        print(f"  ❌ Error creating table: {e}")
        return {'success': False, 'error': str(e)}
    
    # Process each quarter
    print("\n🔍 Filtering holdings per quarter...")
    print("-" * 80)
    
    total_original = 0
    total_filtered = 0
    total_inserted = 0
    
    for i, q in enumerate(quarters):
        quarter_str = q['quarter']
        period_start = q['period_start']
        period_end = q['period_end']
        year = q['year']
        quarter_num = q['quarter_num']
        
        # Filter and insert in one SQL query (efficient)
        # Use period_start for partition pruning
        filter_insert_sql = f"""
        INSERT INTO {target_table} (nameofissuer, cusip, sshprnamt, cik, year, quarter, period_start)
        SELECT 
            h.nameofissuer,
            h.cusip,
            h.sshprnamt,
            h.cik,
            h.year,
            h.quarter,
            h.period_start
        FROM {source_table} h
        INNER JOIN {mapping_table} t ON h.cusip = t.cusip
        WHERE h.period_start >= %s
          AND h.period_start < %s;
        """
        
        # Count original records - use period_start for partition pruning
        count_original_sql = f"""
        SELECT COUNT(*) 
        FROM {source_table} 
        WHERE period_start >= %s AND period_start < %s;
        """
        
        # Count filtered records - use period_start for partition pruning
        count_filtered_sql = f"""
        SELECT COUNT(*) 
        FROM {source_table} h
        INNER JOIN {mapping_table} t ON h.cusip = t.cusip
        WHERE h.period_start >= %s
          AND h.period_start < %s;
        """
        
        try:
            cursor = handler.connection.cursor()
            
            # Get original count
            cursor.execute(count_original_sql, (period_start, period_end))
            original_count = cursor.fetchone()[0]
            
            # Get filtered count
            cursor.execute(count_filtered_sql, (period_start, period_end))
            filtered_count = cursor.fetchone()[0]
            
            # Insert filtered data
            cursor.execute(filter_insert_sql, (period_start, period_end))
            inserted_count = cursor.rowcount
            
            handler.connection.commit()
            cursor.close()
            
            total_original += original_count
            total_filtered += filtered_count
            total_inserted += inserted_count
            
            pct = (i + 1) / len(quarters) * 100
            filter_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
            
            print(f"  [{i+1}/{len(quarters)}] {quarter_str}: "
                  f"{original_count:,} → {filtered_count:,} ({filter_rate:.1f}% kept) | "
                  f"Inserted: {inserted_count:,}")
            
        except Exception as e:
            print(f"  ❌ Error processing {quarter_str}: {e}")
            continue
    
    print("-" * 80)
    
    # Summary
    overall_filter_rate = (total_filtered / total_original * 100) if total_original > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"  Total original records:  {total_original:,}")
    print(f"  Total filtered records:  {total_filtered:,}")
    print(f"  Total inserted:          {total_inserted:,}")
    print(f"  Overall filter rate:     {overall_filter_rate:.1f}% kept")
    print(f"  Records removed:         {total_original - total_filtered:,}")
    
    return {
        'success': True,
        'quarters_processed': len(quarters),
        'total_original': total_original,
        'total_filtered': total_filtered,
        'total_inserted': total_inserted,
        'filter_rate': overall_filter_rate
    }


def run_pipeline(input_file: Path, 
                 start_quarter: str = None,
                 end_quarter: str = None) -> dict:
    """
    Run the complete filter holdings pipeline.
    
    Args:
        input_file: Path to CSV with ticker-CUSIP mappings
        start_quarter: Start quarter
        end_quarter: End quarter
    
    Returns:
        Pipeline stats
    """
    start_quarter = start_quarter or DEFAULT_FILTER_START_QUARTER
    end_quarter = end_quarter or DEFAULT_FILTER_END_QUARTER
    
    print("=" * 80)
    print("FILTER HOLDINGS PIPELINE")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Quarter range: {start_quarter} to {end_quarter}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to database
    print("\n🔌 Connecting to database...")
    handler = PostgresHandler()
    
    if not handler.connect():
        print(f"  ❌ Connection failed")
        return {'success': False, 'error': 'Failed to connect to database'}
    
    print(f"  ✅ Connected to {handler.database}")
    
    try:
        # Step 1: Load ticker-to-CUSIP mapping
        step1_stats = step1_load_ticker_to_cusip(input_file, handler)
        if not step1_stats.get('success'):
            return step1_stats
        
        # Step 2: Filter holdings by quarter
        step2_stats = step2_filter_holdings_by_quarter(handler, start_quarter, end_quarter)
        if not step2_stats.get('success'):
            return step2_stats
        
    finally:
        handler.disconnect()
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'success': True,
        'step1': step1_stats,
        'step2': step2_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Filter holdings by relevant CUSIPs")
    parser.add_argument('--input', '-i', required=True, help='Path to CSV with ticker-CUSIP mappings')
    parser.add_argument('--start', default=DEFAULT_FILTER_START_QUARTER, 
                       help=f'Start quarter (default: {DEFAULT_FILTER_START_QUARTER})')
    parser.add_argument('--end', default=DEFAULT_FILTER_END_QUARTER, 
                       help=f'End quarter (default: {DEFAULT_FILTER_END_QUARTER})')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    run_pipeline(input_file, args.start, args.end)


if __name__ == '__main__':
    main()

