#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for converting large CSV files to Parquet format.

This module provides utilities to efficiently convert large CSV files to Parquet
format with support for chunked reading to handle files larger than available RAM.
"""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def csv_to_parquet(
    csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 50000,
    compression: str = "snappy",
    verbose: bool = True
) -> Union[str, Path]:
    """
    Convert a large CSV file to Parquet format.
    
    This function reads the CSV file in chunks and writes to Parquet,
    which is more memory-efficient than reading the entire file at once.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Path to save the output Parquet file. 
                    If None, saves in the same directory as CSV with .parquet extension
        chunk_size: Number of rows per chunk (default: 50000)
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd', 'none')
        verbose: Print progress information
        
    Returns:
        Path to the created Parquet file
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.with_suffix('.parquet')
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"Converting {csv_path.name} ({file_size_mb:.2f} MB) to Parquet...")
    
    # Read CSV in chunks and collect tables
    parquet_writer = None
    rows_processed = 0
    
    try:
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            if chunk.empty:
                continue
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(chunk)
            
            # Initialize writer with the first chunk's schema
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression=compression
                )
            
            # Write chunk to Parquet
            parquet_writer.write_table(table)
            rows_processed += len(chunk)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {rows_processed:,} rows...")
        
        if parquet_writer is not None:
            parquet_writer.close()
    
    except Exception as e:
        # Clean up on error
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Error converting CSV to Parquet: {e}")
    
    if rows_processed == 0:
        raise ValueError("CSV file appears to be empty")
    
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"✓ Conversion complete! {rows_processed:,} rows written to {output_path.name} ({output_size_mb:.2f} MB)")
    
    return output_path


def csv_to_parquet_with_dtypes(
    csv_path: Union[str, Path],
    dtype_mapping: dict,
    output_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 50000,
    compression: str = "snappy",
    verbose: bool = True
) -> Union[str, Path]:
    """
    Convert CSV to Parquet with specific data types.
    
    Args:
        csv_path: Path to the input CSV file
        dtype_mapping: Dictionary mapping column names to pandas dtypes
                      Example: {'date': 'datetime64[ns]', 'price': 'float32'}
        output_path: Path to save the output Parquet file
        chunk_size: Number of rows per chunk
        compression: Compression algorithm
        verbose: Print progress information
        
    Returns:
        Path to the created Parquet file
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if output_path is None:
        output_path = csv_path.with_suffix('.parquet')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"Converting {csv_path.name} ({file_size_mb:.2f} MB) to Parquet with specific dtypes...")
    
    parquet_writer = None
    rows_processed = 0
    
    try:
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            if chunk.empty:
                continue
            
            # Apply dtype conversions
            for col, dtype in dtype_mapping.items():
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(dtype)
            
            table = pa.Table.from_pandas(chunk)
            
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression=compression
                )
            
            parquet_writer.write_table(table)
            rows_processed += len(chunk)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {rows_processed:,} rows...")
        
        if parquet_writer is not None:
            parquet_writer.close()
    
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Error converting CSV to Parquet: {e}")
    
    if rows_processed == 0:
        raise ValueError("CSV file appears to be empty")
    
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"✓ Conversion complete! {rows_processed:,} rows written to {output_path.name} ({output_size_mb:.2f} MB)")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python csv_to_parquet.py <csv_file> [output_file] [chunk_size]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    
    try:
        result = csv_to_parquet(csv_file, output_file, chunk_size)
        print(f"Successfully saved to: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
