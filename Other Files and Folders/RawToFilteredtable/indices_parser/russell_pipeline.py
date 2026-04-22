#!/usr/bin/env python
"""
Russell 3000 Index Processing Pipeline

Processes Russell 3000 index files through 8 steps:
1. Parse PDF/XML files into CSV
2. Update CUSIPs by company name
3. Map CUSIPs to TICKERs
4. Filter invalid records
5. Remove duplicates (keep oldest)
6. Determine trading periods
7. Filter by trading end date
8. Extract quarter-end prices

Usage:
    python indices_parser/russell_pipeline.py --input-dir DIR [options]
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indices_parser.utils import format_duration

# Import config - use absolute import to avoid circular import
import sys
from pathlib import Path
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RawToFilteredtable.config import (
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_CSV, DEFAULT_FILTERED_CSV,
    DEFAULT_REFERENCE_FILE, TABLE_HOLDINGS_NEW, TABLE_TICKERPRICES,
    DEFAULT_PRICE_START_QUARTER, DEFAULT_PRICE_END_QUARTER
)
from indices_parser.steps import (
    parse_all_index_files,
    update_cusips_by_name,
    map_cusips_to_tickers,
    filter_invalid_records,
    remove_duplicate_records,
    determine_trading_periods,
    filter_by_trading_end_date,
    extract_quarter_end_prices,
)


def run_pipeline(args) -> None:
    """Execute the full pipeline."""
    input_dir = Path(args.input_dir)
    intermediate_file = input_dir / 'russell3000_all_years.csv'
    output_file = Path(args.output)
    filtered_file = Path(args.filtered)
    
    print("=" * 80)
    print("RUSSELL 3000 INDEX PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_file}")
    
    pipeline_start = time.time()
    stats = {}
    
    # Step 1: Parse files
    stats[1] = _run_step(1, "Parse Index Files", args.skip_step,
        lambda: parse_all_index_files(input_dir, intermediate_file))
    
    # Step 2: Update CUSIPs
    stats[2] = _run_step(2, "Update CUSIPs by Name", args.skip_step,
        lambda: update_cusips_by_name(intermediate_file, args.table))
    
    # Step 3: Map TICKERs
    stats[3] = _run_step(3, "Map CUSIPs to TICKERs", args.skip_step,
        lambda: map_cusips_to_tickers(intermediate_file, Path(args.reference), 
                                       args.use_quantumonline))
    
    # Step 4: Filter records
    stats[4] = _run_step(4, "Filter Invalid Records", args.skip_step,
        lambda: filter_invalid_records(intermediate_file, output_file, filtered_file))
    
    # Step 5: Remove duplicates
    stats[5] = _run_step(5, "Remove Duplicates", args.skip_step,
        lambda: remove_duplicate_records(output_file, output_file))
    
    # Step 6: Trading periods
    stats[6] = _run_step(6, "Determine Trading Periods", args.skip_step,
        lambda: determine_trading_periods(output_file))
    
    # Step 7: Filter by end date
    stats[7] = _run_step(7, "Filter by Trading End Date", args.skip_step,
        lambda: filter_by_trading_end_date(output_file, output_file, 2013))
    
    # Step 8: Extract prices
    stats[8] = _run_step(8, "Extract Quarter-End Prices", args.skip_step,
        lambda: extract_quarter_end_prices(output_file, args.price_table,
                                           args.price_start, args.price_end))
    
    # Final report
    _print_report(stats, output_file, filtered_file, time.time() - pipeline_start)


def _run_step(num: int, name: str, skip_steps: list, func) -> Dict:
    """Run a single pipeline step with timing and error handling."""
    print(f"\n{'='*80}")
    print(f"STEP {num}: {name.upper()}")
    print("=" * 80)
    
    if num in skip_steps:
        print("  ⏭️  Skipped")
        return {'skipped': True, 'duration': 0}
    
    start = time.time()
    try:
        result = func()
        result['duration'] = time.time() - start
        result['success'] = True
        print(f"\n  ✅ Completed in {format_duration(result['duration'])}")
        for key, value in result.items():
            if key not in ('duration', 'success', 'skipped'):
                print(f"     {key}: {value}")
        return result
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'duration': time.time() - start}


def _print_report(stats: Dict, output_file: Path, filtered_file: Path, 
                  total_duration: float) -> None:
    """Print final pipeline report."""
    print("\n" + "=" * 80)
    print("📊 PIPELINE SUMMARY")
    print("=" * 80)
    
    total_time = sum(s.get('duration', 0) for s in stats.values())
    
    for num in sorted(stats.keys()):
        s = stats[num]
        status = "⏭️" if s.get('skipped') else ("✅" if s.get('success') else "❌")
        duration = format_duration(s.get('duration', 0))
        print(f"  Step {num}: {status} ({duration})")
    
    print(f"\n  Total Duration: {format_duration(total_duration)}")
    print(f"  Output File: {output_file}")
    print(f"  Filtered File: {filtered_file}")
    print(f"\n✅ Pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Russell 3000 Index Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', '-i', type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Input directory with index files (default: {DEFAULT_INPUT_DIR})')
    
    parser.add_argument('--output', '-o', type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f'Output CSV file (default: {DEFAULT_OUTPUT_CSV})')
    
    parser.add_argument('--filtered', '-f', type=str,
        default=DEFAULT_FILTERED_CSV,
        help=f'Filtered records file (default: {DEFAULT_FILTERED_CSV})')
    
    parser.add_argument('--table', '-t', type=str, default=TABLE_HOLDINGS_NEW,
        help=f'Database table for CUSIP lookup (default: {TABLE_HOLDINGS_NEW})')
    
    parser.add_argument('--reference', '-r', type=str,
        default=DEFAULT_REFERENCE_FILE,
        help=f'Reference file for TICKER mapping (default: {DEFAULT_REFERENCE_FILE})')
    
    parser.add_argument('--no-quantumonline', action='store_false',
        dest='use_quantumonline',
        help='Disable web fallback for TICKER mapping')
    
    parser.add_argument('--skip-step', type=int, nargs='+', default=[],
        help='Steps to skip (1-8)')
    
    parser.add_argument('--price-table', type=str, default=TABLE_TICKERPRICES,
        help=f'Database table for prices (default: {TABLE_TICKERPRICES})')
    
    parser.add_argument('--price-start', type=str, default=DEFAULT_PRICE_START_QUARTER,
        help=f'Start quarter for prices (default: {DEFAULT_PRICE_START_QUARTER})')
    
    parser.add_argument('--price-end', type=str, default=DEFAULT_PRICE_END_QUARTER,
        help=f'End quarter for prices (default: {DEFAULT_PRICE_END_QUARTER})')
    
    args = parser.parse_args()
    
    # Validate
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌ Error: Directory not found: {input_dir}")
        sys.exit(1)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.filtered).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
