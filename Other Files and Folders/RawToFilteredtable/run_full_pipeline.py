#!/usr/bin/env python
"""
Full Pipeline: Russell Index Parser → Filter Holdings

This script runs both pipelines sequentially:
1. indices_parser: Parse index files, update CUSIPs, map TICKERs, determine trading periods, extract prices
2. filterholdings: Load ticker-CUSIP mapping, filter holdings by relevant CUSIPs

All output is logged to pipeline_log.txt

Usage:
    python RawToFilteredtable/run_full_pipeline.py [options]
    
Examples:
    python RawToFilteredtable/run_full_pipeline.py
    python RawToFilteredtable/run_full_pipeline.py --skip-indices-step 6 7 8
    python RawToFilteredtable/run_full_pipeline.py --log-file my_log.txt
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict
import io

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

# Import pipelines
from indices_parser.russell_pipeline import run_pipeline as run_indices_pipeline
from filterholdings.filter_holdings_pipeline import run_pipeline as run_filter_pipeline

# Import config - use absolute import to avoid circular import
# Import after setting up paths to ensure RawToFilteredtable is in sys.path
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from RawToFilteredtable.config import (
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_CSV, DEFAULT_FILTERED_CSV, DEFAULT_LOG_FILE,
    DEFAULT_REFERENCE_FILE, TABLE_HOLDINGS_NEW, TABLE_TICKERPRICES,
    DEFAULT_PRICE_START_QUARTER, DEFAULT_PRICE_END_QUARTER,
    DEFAULT_FILTER_START_QUARTER, DEFAULT_FILTER_END_QUARTER
)


class TeeOutput:
    """Write to both stdout and a file simultaneously."""
    
    def __init__(self, file_path: Path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def create_args_namespace(args_dict: Dict):
    """Create argparse Namespace from dictionary."""
    return argparse.Namespace(**args_dict)


def run_full_pipeline(
    input_dir: Path,
    output_csv: Path,
    filtered_csv: Path,
    log_file: Path,
    skip_indices_steps: list = None,
    skip_filter: bool = False,
    table: str = None,
    reference: str = None,
    use_quantumonline: bool = True,
    price_table: str = None,
    price_start: str = None,
    price_end: str = None,
    filter_start: str = None,
    filter_end: str = None
) -> Dict:
    """
    Run the complete pipeline: indices_parser → filterholdings.
    
    Args:
        input_dir: Directory with Russell index files (PDF/XML)
        output_csv: Output CSV from indices_parser (input to filterholdings)
        filtered_csv: Filtered records from indices_parser
        log_file: Path to log file
        skip_indices_steps: Steps to skip in indices_parser (1-8)
        skip_filter: Skip filterholdings pipeline entirely
        table: Database table for CUSIP lookup
        reference: Reference file for TICKER mapping
        use_quantumonline: Use web fallback for TICKER mapping
        price_table: Database table for prices
        price_start: Start quarter for prices
        price_end: End quarter for prices
        filter_start: Start quarter for filtering
        filter_end: End quarter for filtering
    
    Returns:
        Dict with pipeline results
    """
    skip_indices_steps = skip_indices_steps or []
    
    # Use defaults from config if not provided
    table = table or TABLE_HOLDINGS_NEW
    reference = reference or DEFAULT_REFERENCE_FILE
    price_table = price_table or TABLE_TICKERPRICES
    price_start = price_start or DEFAULT_PRICE_START_QUARTER
    price_end = price_end or DEFAULT_PRICE_END_QUARTER
    filter_start = filter_start or DEFAULT_FILTER_START_QUARTER
    filter_end = filter_end or DEFAULT_FILTER_END_QUARTER
    
    # Setup logging to both console and file
    tee = TeeOutput(log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        pipeline_start = time.time()
        
        print("=" * 80)
        print("🚀 FULL PIPELINE: RUSSELL INDEX → FILTERED HOLDINGS")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file}")
        print(f"Input directory: {input_dir}")
        print(f"Output CSV: {output_csv}")
        print("-" * 80)
        
        results = {
            'indices_parser': None,
            'filterholdings': None
        }
        
        # =====================================================================
        # PHASE 1: INDICES PARSER PIPELINE
        # =====================================================================
        print("\n" + "=" * 80)
        print("📦 PHASE 1: INDICES PARSER PIPELINE")
        print("=" * 80)
        
        phase1_start = time.time()
        
        # Create args namespace for indices_parser
        indices_args = create_args_namespace({
            'input_dir': str(input_dir),
            'output': str(output_csv),
            'filtered': str(filtered_csv),
            'table': table,
            'reference': reference,
            'use_quantumonline': use_quantumonline,
            'skip_step': skip_indices_steps,
            'price_table': price_table,
            'price_start': price_start,
            'price_end': price_end
        })
        
        try:
            run_indices_pipeline(indices_args)
            results['indices_parser'] = {
                'success': True,
                'duration': time.time() - phase1_start
            }
            print(f"\n✅ Phase 1 completed in {format_duration(time.time() - phase1_start)}")
        except Exception as e:
            print(f"\n❌ Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()
            results['indices_parser'] = {
                'success': False,
                'error': str(e),
                'duration': time.time() - phase1_start
            }
            # Don't continue if indices_parser failed
            return results
        
        # =====================================================================
        # PHASE 2: FILTER HOLDINGS PIPELINE
        # =====================================================================
        if skip_filter:
            print("\n" + "=" * 80)
            print("⏭️  PHASE 2: FILTER HOLDINGS (SKIPPED)")
            print("=" * 80)
            results['filterholdings'] = {'skipped': True}
        else:
            print("\n" + "=" * 80)
            print("📦 PHASE 2: FILTER HOLDINGS PIPELINE")
            print("=" * 80)
            
            phase2_start = time.time()
            
            # Check if output CSV exists
            if not output_csv.exists():
                print(f"❌ Error: Output CSV from Phase 1 not found: {output_csv}")
                results['filterholdings'] = {
                    'success': False,
                    'error': 'Output CSV not found'
                }
            else:
                try:
                    filter_result = run_filter_pipeline(
                        output_csv, 
                        filter_start, 
                        filter_end
                    )
                    results['filterholdings'] = {
                        'success': filter_result.get('success', False),
                        'duration': time.time() - phase2_start,
                        **filter_result
                    }
                    print(f"\n✅ Phase 2 completed in {format_duration(time.time() - phase2_start)}")
                except Exception as e:
                    print(f"\n❌ Phase 2 failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results['filterholdings'] = {
                        'success': False,
                        'error': str(e),
                        'duration': time.time() - phase2_start
                    }
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        total_duration = time.time() - pipeline_start
        
        print("\n" + "=" * 80)
        print("📊 FULL PIPELINE SUMMARY")
        print("=" * 80)
        
        # Phase 1 status
        p1 = results['indices_parser']
        p1_status = "✅" if p1 and p1.get('success') else "❌"
        p1_duration = format_duration(p1.get('duration', 0)) if p1 else "N/A"
        print(f"  Phase 1 (Indices Parser): {p1_status} ({p1_duration})")
        
        # Phase 2 status
        p2 = results['filterholdings']
        if p2 and p2.get('skipped'):
            p2_status = "⏭️ Skipped"
            p2_duration = ""
        else:
            p2_status = "✅" if p2 and p2.get('success') else "❌"
            p2_duration = f" ({format_duration(p2.get('duration', 0))})" if p2 else ""
        print(f"  Phase 2 (Filter Holdings): {p2_status}{p2_duration}")
        
        print(f"\n  Total Duration: {format_duration(total_duration)}")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n  Log saved to: {log_file}")
        print("=" * 80)
        
        return results
        
    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.close()
        
        # Print final message to console
        print(f"\n✅ Pipeline complete. Full log saved to: {log_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Russell Index Parser → Filter Holdings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument('--input-dir', '-i', type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Input directory with Russell index files (default: {DEFAULT_INPUT_DIR})')
    
    parser.add_argument('--output', '-o', type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f'Output CSV file (input to filterholdings) (default: {DEFAULT_OUTPUT_CSV})')
    
    parser.add_argument('--filtered', type=str,
        default=DEFAULT_FILTERED_CSV,
        help=f'Filtered records file (default: {DEFAULT_FILTERED_CSV})')
    
    parser.add_argument('--log-file', '-l', type=str,
        default=DEFAULT_LOG_FILE,
        help=f'Log file path (default: {DEFAULT_LOG_FILE})')
    
    # Indices parser options
    parser.add_argument('--skip-indices-step', type=int, nargs='+', default=[],
        help='Steps to skip in indices_parser (1-8)')
    
    parser.add_argument('--table', '-t', type=str, default=TABLE_HOLDINGS_NEW,
        help=f'Database table for CUSIP lookup (default: {TABLE_HOLDINGS_NEW})')
    
    parser.add_argument('--reference', '-r', type=str,
        default=DEFAULT_REFERENCE_FILE,
        help=f'Reference file for TICKER mapping (default: {DEFAULT_REFERENCE_FILE})')
    
    parser.add_argument('--no-quantumonline', action='store_false',
        dest='use_quantumonline',
        help='Disable web fallback for TICKER mapping')
    
    parser.add_argument('--price-table', type=str, default=TABLE_TICKERPRICES,
        help=f'Database table for prices (default: {TABLE_TICKERPRICES})')
    
    parser.add_argument('--price-start', type=str, default=DEFAULT_PRICE_START_QUARTER,
        help=f'Start quarter for prices (default: {DEFAULT_PRICE_START_QUARTER})')
    
    parser.add_argument('--price-end', type=str, default=DEFAULT_PRICE_END_QUARTER,
        help=f'End quarter for prices (default: {DEFAULT_PRICE_END_QUARTER})')
    
    # Filter holdings options
    parser.add_argument('--skip-filter', action='store_true',
        help='Skip the filterholdings pipeline')
    
    parser.add_argument('--filter-start', type=str, default=DEFAULT_FILTER_START_QUARTER,
        help=f'Start quarter for filtering (default: {DEFAULT_FILTER_START_QUARTER})')
    
    parser.add_argument('--filter-end', type=str, default=DEFAULT_FILTER_END_QUARTER,
        help=f'End quarter for filtering (default: {DEFAULT_FILTER_END_QUARTER})')
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_csv = Path(args.output)
    filtered_csv = Path(args.filtered)
    log_file = Path(args.log_file)
    
    # Create directories if needed
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    try:
        results = run_full_pipeline(
            input_dir=input_dir,
            output_csv=output_csv,
            filtered_csv=filtered_csv,
            log_file=log_file,
            skip_indices_steps=args.skip_indices_step,
            skip_filter=args.skip_filter,
            table=args.table,
            reference=args.reference,
            use_quantumonline=args.use_quantumonline,
            price_table=args.price_table,
            price_start=args.price_start,
            price_end=args.price_end,
            filter_start=args.filter_start,
            filter_end=args.filter_end
        )
        
        # Exit with error code if any phase failed
        if not results.get('indices_parser', {}).get('success'):
            sys.exit(1)
        if not results.get('filterholdings', {}).get('skipped') and \
           not results.get('filterholdings', {}).get('success'):
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



