"""Step 3: Map CUSIPs to TICKERs using reference file and web fallback."""

import re
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from indices_parser.utils import pad_cusip
from RawToFilteredtable.config import (
    QUANTUMONLINE_URL, QUANTUMONLINE_WORKERS, 
    QUANTUMONLINE_SLEEP, QUANTUMONLINE_TIMEOUT
)

_lock = threading.Lock()


def map_cusips_to_tickers(input_file: Path, reference_file: Path, 
                          use_web_fallback: bool = True) -> Dict:
    """Map CUSIPs to TICKERs using reference file, optionally with web fallback."""
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    cusip_col = _find_col(df, 'cusip')
    ticker_col = _find_col(df, 'ticker') or 'TICKER'
    if ticker_col not in df.columns:
        df[ticker_col] = ''
    
    if not cusip_col:
        raise ValueError("'cusip' column not found")
    
    # Load reference and map
    ref = _load_reference(reference_file)
    print(f"  Loaded {len(ref)} CUSIP→TICKER mappings from reference file")
    
    stats = {'reference': 0, 'web': 0, 'not_found': 0}
    missing = []
    already = 0
    
    print(f"  Searching {len(df)} records in reference file...")
    for idx, row in df.iterrows():
        if pd.notna(row[ticker_col]) and str(row[ticker_col]).strip():
            already += 1
            continue
        
        cusip = str(row[cusip_col]).strip().upper() if pd.notna(row[cusip_col]) else ''
        if not cusip:
            continue
        
        ticker = ref.get(cusip) or ref.get(pad_cusip(cusip))
        if ticker:
            df.at[idx, ticker_col] = ticker
            stats['reference'] += 1
        else:
            missing.append((idx, cusip))
    
    print(f"  ⏭️  Already had TICKER: {already}")
    print(f"  ✅ Found {stats['reference']} NEW tickers from reference file")
    print(f"  ⏳ {len(missing)} CUSIPs still need lookup")
    
    df.to_csv(input_file, index=False)
    print(f"  💾 Saved intermediate results to {input_file.name}")
    
    # Web fallback
    if use_web_fallback and missing:
        print(f"  Looking up {len(missing)} CUSIPs via web...")
        web_results = _lookup_web(missing)
        for idx, _ in missing:
            if idx in web_results:
                df.at[idx, ticker_col] = web_results[idx]
                stats['web'] += 1
            else:
                stats['not_found'] += 1
    else:
        stats['not_found'] = len(missing)
    
    df.to_csv(input_file, index=False)
    return stats


def _find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find column case-insensitively."""
    return next((c for c in df.columns if c.lower() == name.lower()), None)


def _load_reference(path: Path) -> Dict[str, str]:
    """Load CUSIP->TICKER mapping from reference CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    cusip_col, ticker_col = _find_col(df, 'cusip'), _find_col(df, 'ticker')
    if not cusip_col or not ticker_col:
        return {}
    
    return {
        str(row[cusip_col]).strip().upper(): str(row[ticker_col]).strip()
        for _, row in df.iterrows()
        if pd.notna(row[cusip_col]) and pd.notna(row[ticker_col])
        and str(row[cusip_col]).strip() and str(row[ticker_col]).strip()
    }


def _lookup_web(tasks: List[Tuple[int, str]]) -> Dict[int, str]:
    """Lookup tickers via QuantumOnline using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  ⚠️ Playwright not installed, skipping web lookup")
        return {}
    
    results = {}
    total = len(tasks)
    progress = {'done': 0, 'found': 0}
    browsers = []  # Track all browsers for cleanup
    browsers_lock = threading.Lock()
    local = threading.local()
    
    def get_page():
        if not hasattr(local, 'pw'):
            local.pw = sync_playwright().start()
            local.browser = local.pw.chromium.launch(headless=True,
                args=['--disable-blink-features=AutomationControlled'])
            local.page = local.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ).new_page()
            with browsers_lock:
                browsers.append((local.browser, local.pw))
        return local.page
    
    def process(task):
        idx, cusip = task
        ticker = _search_quantumonline(get_page(), cusip)
        with _lock:
            progress['done'] += 1
            if ticker:
                results[idx] = ticker
                progress['found'] += 1
            if progress['done'] % 10 == 0 or ticker:
                pct = progress['done'] / total * 100
                print(f"\r    Progress: {progress['done']}/{total} ({pct:.1f}%) | Found: {progress['found']}", 
                      end='', flush=True)
        time.sleep(QUANTUMONLINE_SLEEP)
    
    workers = min(QUANTUMONLINE_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(process, t) for t in tasks]))
    
    # Cleanup all browsers
    for browser, pw in browsers:
        try:
            browser.close()
            pw.stop()
        except Exception:
            pass
    
    print(f"\r    Progress: {total}/{total} (100.0%) | Found: {progress['found']}")
    return results


def _search_quantumonline(page, cusip: str) -> Optional[str]:
    """Search QuantumOnline for ticker by CUSIP."""
    try:
        cusip = pad_cusip(cusip)
        
        # Navigate if needed
        if QUANTUMONLINE_URL not in page.url:
            page.goto(QUANTUMONLINE_URL, wait_until="domcontentloaded", timeout=QUANTUMONLINE_TIMEOUT)
        
        # Set dropdown to CUSIP
        page.wait_for_selector('select[name="sopt"]', state="visible", timeout=3000)
        page.select_option('select[name="sopt"]', value="cusip")
        
        # Fill and submit
        page.fill('input[name="tickersymbol"]', cusip)
        page.press('input[name="tickersymbol"]', "Enter")
        page.wait_for_load_state("domcontentloaded", timeout=QUANTUMONLINE_TIMEOUT)
        
        # Parse result
        soup = BeautifulSoup(page.content(), 'html.parser')
        
        # Search in center elements
        for el in soup.find_all('center'):
            match = re.search(r'Ticker\s+Symbol[:\s]+([A-Za-z]{1,5})\*?', el.get_text())
            if match:
                return match.group(1).upper()
        
        # Fallback: entire page
        match = re.search(r'Ticker\s+Symbol[:\s]+([A-Za-z]{1,5})\*?', soup.get_text())
        return match.group(1).upper() if match else None
        
    except Exception:
        return None


def main():
    """Run step 3 as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 3: Map CUSIPs to TICKERs")
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    from RawToFilteredtable.config import DEFAULT_REFERENCE_FILE
    parser.add_argument('--reference', '-r', 
                        default=DEFAULT_REFERENCE_FILE,
                        help=f'Reference CSV file (default: {DEFAULT_REFERENCE_FILE})')
    parser.add_argument('--no-web', action='store_true', help='Disable web fallback')
    parser.add_argument('--use-web', action='store_true', help='Enable web fallback (default)')
    
    args = parser.parse_args()
    input_file, ref_file = Path(args.input), Path(args.reference)
    
    if not input_file.exists():
        sys.exit(f"❌ Input file not found: {input_file}")
    if not ref_file.exists():
        sys.exit(f"❌ Reference file not found: {ref_file}")
    
    print("=" * 80)
    print("STEP 3: MAP CUSIPS TO TICKERS")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Reference: {ref_file}")
    print(f"Web fallback: {'disabled' if args.no_web else 'enabled'}\n")
    
    try:
        stats = map_cusips_to_tickers(input_file, ref_file, not args.no_web)
        print(f"\n{'=' * 80}\n📊 RESULTS: {stats}\n{'=' * 80}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
