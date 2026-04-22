#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to export a database table to parquet file(s).

This script:
1. Connects to the PostgreSQL database
2. Reads the specified table
3. Exports to parquet file(s)
4. If the table is large, splits into multiple files (max 250 MB per file)
"""

import argparse
import os
import sys
import io
from pathlib import Path
from typing import Optional

# Add project root to Python path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler


# Maximum file size in MB
MAX_FILE_SIZE_MB = 250
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def get_table_row_count(handler: PostgresHandler, table_name: str) -> int:
    """
    Get the total number of rows in the table.
    
    Args:
        handler: Database handler instance
        table_name: Name of the table
        
    Returns:
        Total row count
    """
    cursor = handler.connection.cursor()
    quoted_table_name = handler._quote_table_name(table_name)
    cursor.execute(f"SELECT COUNT(*) FROM {quoted_table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def estimate_chunk_size(handler: PostgresHandler, table_name: str, output_dir: Path, sample_size: int = 10000) -> int:
    """
    Estimate the number of rows that will fit in a 250 MB parquet file.
    
    Args:
        handler: Database handler instance
        table_name: Name of the table
        output_dir: Directory where we can write a temporary test file
        sample_size: Number of rows to sample for size estimation
        
    Returns:
        Estimated chunk size (number of rows)
    """
    cursor = handler.connection.cursor()
    quoted_table_name = handler._quote_table_name(table_name)
    
    # Get a sample of data
    query = f"SELECT * FROM {quoted_table_name} LIMIT {sample_size}"
    df_sample = pd.read_sql_query(query, handler.connection)
    cursor.close()
    
    if df_sample.empty:
        return 100000  # Default chunk size
    
    # Write sample to parquet to estimate size (use output directory for temp file)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_file_path = output_dir / f"_temp_size_estimate_{os.getpid()}.parquet"
    try:
        df_sample.to_parquet(tmp_file_path, index=False, compression='snappy')
        sample_size_bytes = os.path.getsize(tmp_file_path)
        
        # Clean up temp file
        if tmp_file_path.exists():
            tmp_file_path.unlink()
    except Exception as e:
        # If we can't write, use a conservative default
        print(f"Warning: Could not estimate chunk size ({str(e)}), using default")
        return 100000
    
    if sample_size_bytes == 0:
        return 100000  # Default chunk size
    
    # Calculate rows per MB, then estimate for 250 MB
    bytes_per_row = sample_size_bytes / len(df_sample)
    rows_per_mb = (1024 * 1024) / bytes_per_row
    estimated_chunk_size = int(rows_per_mb * MAX_FILE_SIZE_MB * 0.9)  # Use 90% to be safe
    
    # Ensure minimum chunk size
    return max(estimated_chunk_size, 1000)


def write_parquet_chunk(df: pd.DataFrame, output_path: Path, chunk_num: int) -> tuple[Path, float]:
    """
    Write a DataFrame chunk to a parquet file.
    
    Args:
        df: DataFrame to write
        output_path: Base output path (without extension)
        chunk_num: Chunk number for naming
        
    Returns:
        Tuple of (file_path, file_size_mb)
    """
    if df.empty:
        return None, 0.0
    
    # Create filename with chunk number
    if chunk_num == 0:
        file_path = output_path.with_suffix('.parquet')
    else:
        file_path = output_path.parent / f"{output_path.stem}_part{chunk_num:03d}.parquet"
    
    # Write to parquet
    df.to_parquet(file_path, index=False, compression='snappy')
    
    # Get file size
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return file_path, file_size_mb


def split_large_chunk(df: pd.DataFrame, output_path: Path, chunk_num: int) -> list[tuple[Path, float]]:
    """
    Split a large DataFrame into smaller chunks that fit within size limit.
    
    Args:
        df: DataFrame to split
        output_path: Base output path
        chunk_num: Base chunk number
        
    Returns:
        List of tuples (file_path, file_size_mb)
    """
    files_created = []
    
    # Estimate rows per file based on current DataFrame size
    if len(df) == 0:
        return files_created
    
    # Try writing the whole chunk first to see if it fits
    test_path = output_path.parent / f"{output_path.stem}_test.parquet"
    df.to_parquet(test_path, index=False, compression='snappy')
    test_size = os.path.getsize(test_path)
    os.remove(test_path)
    
    if test_size <= MAX_FILE_SIZE_BYTES:
        # It fits, write it normally
        file_path, file_size_mb = write_parquet_chunk(df, output_path, chunk_num)
        if file_path:
            files_created.append((file_path, file_size_mb))
        return files_created
    
    # Need to split further
    # Calculate how many rows per sub-chunk
    bytes_per_row = test_size / len(df)
    rows_per_chunk = int((MAX_FILE_SIZE_BYTES * 0.9) / bytes_per_row)  # Use 90% to be safe
    rows_per_chunk = max(rows_per_chunk, 100)  # Minimum 100 rows
    
    # Split into sub-chunks
    num_sub_chunks = (len(df) + rows_per_chunk - 1) // rows_per_chunk
    
    for sub_chunk_num in range(num_sub_chunks):
        start_idx = sub_chunk_num * rows_per_chunk
        end_idx = min((sub_chunk_num + 1) * rows_per_chunk, len(df))
        sub_df = df.iloc[start_idx:end_idx]
        
        sub_chunk_name = chunk_num * 1000 + sub_chunk_num  # Ensure unique numbering
        file_path, file_size_mb = write_parquet_chunk(sub_df, output_path, sub_chunk_name)
        
        if file_path:
            files_created.append((file_path, file_size_mb))
            print(f"  Created sub-chunk: {file_path.name} ({file_size_mb:.2f} MB)")
    
    return files_created


def export_table_to_parquet(
    table_name: str,
    output_path: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> list[tuple[Path, float]]:
    """
    Export a database table to parquet file(s).
    
    Args:
        table_name: Name of the table to export
        output_path: Output file path (optional, defaults to table_name.parquet)
        chunk_size: Number of rows per chunk (optional, auto-estimated if not provided)
        
    Returns:
        List of tuples (file_path, file_size_mb) for created files
    """
    # Initialize database handler
    handler = PostgresHandler()
    
    try:
        # Connect to database
        print(f"Connecting to database...")
        if not handler.connect():
            raise ConnectionError("Failed to connect to database")
        
        # Check if table exists
        if not handler.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Get table row count
        print(f"Reading table '{table_name}'...")
        total_rows = get_table_row_count(handler, table_name)
        print(f"Total rows: {total_rows:,}")
        
        if total_rows == 0:
            print("Table is empty. No files created.")
            return []
        
        # Set output path
        if output_path is None:
            output_path = Path.cwd() / f"{table_name}.parquet"
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Estimate chunk size if not provided
        if chunk_size is None:
            print("Estimating optimal chunk size...")
            chunk_size = estimate_chunk_size(handler, table_name, output_path.parent)
            print(f"Estimated chunk size: {chunk_size:,} rows per file")
        
        # Calculate number of chunks needed
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        print(f"\nExporting to parquet file(s)...")
        print(f"Will create approximately {num_chunks} file(s)")
        
        files_created = []
        quoted_table_name = handler._quote_table_name(table_name)
        
        # Use streaming with chunksize (much faster than OFFSET for large tables)
        # This reads data in chunks without using slow OFFSET
        print(f"\nUsing optimized streaming export...")
        
        # Read in larger chunks from database for efficiency
        read_chunk_size = min(chunk_size * 3, 1000000)  # Read up to 1M rows at a time from DB
        
        print(f"Reading from database in chunks of {read_chunk_size:,} rows...")
        
        chunk_num = 0
        current_file_rows = []
        
        # Use chunksize parameter for efficient streaming (no OFFSET!)
        query = f"SELECT * FROM {quoted_table_name}"
        for df_chunk in pd.read_sql_query(query, handler.connection, chunksize=read_chunk_size):
            if df_chunk.empty:
                break
            
            # Process rows from this database chunk
            rows_processed = 0
            while rows_processed < len(df_chunk):
                remaining_in_chunk = len(df_chunk) - rows_processed
                remaining_for_file = chunk_size - len(current_file_rows) if current_file_rows else chunk_size
                rows_to_take = min(remaining_for_file, remaining_in_chunk)
                
                # Add rows to current file buffer
                chunk_slice = df_chunk.iloc[rows_processed:rows_processed + rows_to_take]
                if current_file_rows:
                    current_file_rows.append(chunk_slice)
                else:
                    current_file_rows = [chunk_slice]
                
                rows_processed += rows_to_take
                
                # Check if we've accumulated enough for a file
                total_rows_in_buffer = sum(len(df) for df in current_file_rows)
                if total_rows_in_buffer >= chunk_size:
                    # Combine and write
                    df_to_write = pd.concat(current_file_rows, ignore_index=True)
                    file_path, file_size_mb = write_parquet_chunk(df_to_write, output_path, chunk_num)
                    
                    # Check if file is too large
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        print(f"  File too large ({file_size_mb:.2f} MB), splitting...")
                        os.remove(file_path)
                        sub_files = split_large_chunk(df_to_write, output_path, chunk_num)
                        files_created.extend(sub_files)
                    else:
                        if file_path:
                            files_created.append((file_path, file_size_mb))
                            print(f"  Created: {file_path.name} ({file_size_mb:.2f} MB, {len(df_to_write):,} rows)")
                    
                    chunk_num += 1
                    current_file_rows = []
                    
                    # Progress update every 10 files
                    if len(files_created) % 10 == 0:
                        print(f"  Progress: {len(files_created)} file(s) created...")
        
        # Write any remaining rows
        if current_file_rows:
            df_to_write = pd.concat(current_file_rows, ignore_index=True)
            file_path, file_size_mb = write_parquet_chunk(df_to_write, output_path, chunk_num)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                print(f"  File too large ({file_size_mb:.2f} MB), splitting...")
                os.remove(file_path)
                sub_files = split_large_chunk(df_to_write, output_path, chunk_num)
                files_created.extend(sub_files)
            else:
                if file_path:
                    files_created.append((file_path, file_size_mb))
                    print(f"  Created: {file_path.name} ({file_size_mb:.2f} MB, {len(df_to_write):,} rows)")
        
        # Summary
        total_size_mb = sum(size for _, size in files_created)
        print(f"\n{'='*80}")
        print(f"EXPORT COMPLETE")
        print(f"{'='*80}")
        print(f"Total files created: {len(files_created)}")
        print(f"Total size: {total_size_mb:.2f} MB")
        print(f"Files:")
        for file_path, file_size_mb in files_created:
            print(f"  - {file_path.name} ({file_size_mb:.2f} MB)")
        print(f"{'='*80}\n")
        
        return files_created
        
    except Exception as e:
        print(f"\nError exporting table: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        handler.disconnect()


def main():
    """Main function to execute the script."""
    parser = argparse.ArgumentParser(
        description="Export a database table to parquet file(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export table 'holdings' to default location
  python from_database_to_parquet.py holdings
  
  # Export table 'ticker_prices' to specific location
  python from_database_to_parquet.py ticker_prices -o Data/ticker_prices.parquet
  
  # Export with custom chunk size
  python from_database_to_parquet.py holdings -c 50000
        """
    )
    
    parser.add_argument(
        'table_name',
        type=str,
        help='Name of the table to export'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: ./<table_name>.parquet)'
    )
    
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=None,
        help=f'Number of rows per chunk (default: auto-estimated, target: ~{MAX_FILE_SIZE_MB} MB per file)'
    )
    
    args = parser.parse_args()
    
    try:
        export_table_to_parquet(
            table_name=args.table_name,
            output_path=args.output,
            chunk_size=args.chunk_size
        )
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
