#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the `holdings_filtered_new` table into separate parquet files,
one file per distinct `period_start` value.

Usage examples:

  # Default: read holdings_filtered_new and write under Data/parquuet_files/holdings_filtered_new
  python export_holdings_filtered_by_period.py

  # Custom output directory
  python export_holdings_filtered_by_period.py -o Data/parquuet_files/custom_dir
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Any

import pandas as pd

# Add project root to Python path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler


def get_distinct_periods(
    handler: PostgresHandler,
    table_name: str,
    period_column: str = "period_start",
) -> List[Any]:
    """
    Fetch all distinct period values from the given table.
    """
    cursor = handler.connection.cursor()
    quoted_table_name = handler._quote_table_name(table_name)
    query = f"SELECT DISTINCT {period_column} FROM {quoted_table_name} ORDER BY {period_column}"
    cursor.execute(query)
    periods = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return periods


def safe_period_str(period_value: Any) -> str:
    """
    Convert a period value (date/datetime/other) into a filesystem-safe string.
    """
    if period_value is None:
        return "null"

    # Let pandas / datetime-like objects stringify first, then sanitize
    s = str(period_value)
    # Remove characters that are problematic in filenames
    for ch in [":", "/", "\\", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "")
    s = s.replace(" ", "_")
    return s


def export_holdings_filtered_by_period(
    table_name: str = "holdings_filtered_new",
    period_column: str = "period_start",
    output_dir: Optional[str] = None,
) -> List[Tuple[Path, float]]:
    """
    Export a table into separate parquet files, one per distinct period_start.

    Args:
        table_name: Source table name (default: holdings_filtered_new)
        period_column: Column used to split data (default: period_start)
        output_dir: Base directory for parquet files.
                    Defaults to <project_root>/Data/parquuet_files/holdings_filtered_new_by_period

    Returns:
        List of (file_path, file_size_mb) for created files.
    """
    handler = PostgresHandler()
    files_created: List[Tuple[Path, float]] = []

    try:
        print("Connecting to database...")
        if not handler.connect():
            raise ConnectionError("Failed to connect to database")

        if not handler.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist in the database")

        # Resolve output directory
        if output_dir is None:
            output_base = project_root / "Data" / "parquuet_files" / "holdings_filtered_new_by_period"
        else:
            output_base = Path(output_dir)

        output_base.mkdir(parents=True, exist_ok=True)

        print(f"Exporting table '{table_name}' by distinct '{period_column}' values...")

        periods = get_distinct_periods(handler, table_name, period_column)
        if not periods:
            print("No distinct periods found. Nothing to export.")
            return []

        print(f"Found {len(periods)} distinct period(s).")

        quoted_table_name = handler._quote_table_name(table_name)

        for idx, period_value in enumerate(periods, start=1):
            period_str = safe_period_str(period_value)
            file_name = f"{table_name}_{period_column}_{period_str}.parquet"
            file_path = output_base / file_name

            print(f"[{idx}/{len(periods)}] Exporting period '{period_value}' -> {file_name}")

            # Read all rows for this period into memory.
            # This is intentionally per-period, as user requested one file per period.
            query = f"SELECT * FROM {quoted_table_name} WHERE {period_column} = %s"
            df = pd.read_sql_query(query, handler.connection, params=[period_value])

            if df.empty:
                print(f"  Period '{period_value}' has no rows. Skipping.")
                continue

            df.to_parquet(file_path, index=False, compression="snappy")

            size_mb = file_path.stat().st_size / (1024 * 1024)
            files_created.append((file_path, size_mb))
            print(f"  Created: {file_path} ({size_mb:.2f} MB, {len(df):,} rows)")

        print("\n" + "=" * 80)
        print("EXPORT BY PERIOD COMPLETE")
        print("=" * 80)
        print(f"Total files created: {len(files_created)}")
        total_mb = sum(size for _, size in files_created)
        print(f"Total size: {total_mb:.2f} MB")
        print("Files:")
        for path, size in files_created:
            print(f"  - {path} ({size:.2f} MB)")
        print("=" * 80 + "\n")

        return files_created

    except Exception as e:
        print(f"\nError exporting table by period: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        handler.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the 'holdings_filtered_new' table into separate parquet files, "
            "one per distinct period_start."
        )
    )

    parser.add_argument(
        "-t",
        "--table-name",
        type=str,
        default="holdings_filtered_new",
        help="Source table name (default: holdings_filtered_new)",
    )

    parser.add_argument(
        "-p",
        "--period-column",
        type=str,
        default="period_start",
        help="Column used to split the data (default: period_start)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for parquet files. "
            "Default: <project_root>/Data/parquuet_files/holdings_filtered_new_by_period"
        ),
    )

    args = parser.parse_args()

    try:
        export_holdings_filtered_by_period(
            table_name=args.table_name,
            period_column=args.period_column,
            output_dir=args.output_dir,
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()

