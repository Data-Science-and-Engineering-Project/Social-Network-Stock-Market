"""Step 1: Parse all PDF and XML index files into a single CSV."""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from indices_parser.parsers import IndexParser
from indices_parser.utils import extract_year_from_filename, pad_cusip


def parse_all_index_files(input_dir: Path, output_file: Path) -> Dict:
    """
    Parse all index files and combine into single CSV.
    
    PDF (2013-2018): Keep oldest year for each company
    XML (2019-2025): Update CUSIP if company exists, keep original year
    """
    companies: Dict[str, Dict] = {}
    
    # Process PDF files first (sorted by year)
    pdf_stats = _process_pdf_files(input_dir, companies)
    
    # Process XML files (update CUSIPs)
    xml_stats = _process_xml_files(input_dir, companies)
    
    # Save results
    if not companies:
        raise ValueError("No companies found in any files")
    
    df = _create_dataframe(companies)
    df.to_csv(output_file, index=False)
    
    return {
        'total_rows': len(df),
        'pdf_added': pdf_stats['added'],
        'xml_added': xml_stats['added'],
        'xml_updated': xml_stats['updated'],
        'year_range': f"{df['year'].min()}-{df['year'].max()}"
    }


def _process_pdf_files(input_dir: Path, companies: Dict) -> Dict:
    """Process PDF files (2013-2018), keep oldest year per company."""
    stats = {'added': 0}
    
    pdf_files = _get_sorted_files(input_dir, '*.pdf', 2013, 2018)
    total = len(pdf_files)
    
    print(f"  Found {total} PDF files to process")
    
    for i, (year, pdf_file) in enumerate(pdf_files, 1):
        print(f"  [{i}/{total}] Processing PDF: {pdf_file.name} ({year})", end='', flush=True)
        try:
            df = IndexParser(pdf_file).to_dataframe()
            added_this_file = 0
            for _, row in df.iterrows():
                name = str(row.get('name', '')).strip()
                if name and name not in companies:
                    companies[name] = {'cusip': '', 'year': year}
                    stats['added'] += 1
                    added_this_file += 1
            print(f" → {added_this_file} new companies")
        except Exception as e:
            print(f" → ⚠️ Error: {e}")
    
    return stats


def _process_xml_files(input_dir: Path, companies: Dict) -> Dict:
    """Process XML files (2019-2025), update CUSIP for existing companies."""
    stats = {'added': 0, 'updated': 0}
    
    xml_files = _get_sorted_files(input_dir, '*.xml', 2019, 2025)
    total = len(xml_files)
    
    print(f"  Found {total} XML files to process")
    
    for i, (year, xml_file) in enumerate(xml_files, 1):
        print(f"  [{i}/{total}] Processing XML: {xml_file.name} ({year})", end='', flush=True)
        try:
            df = IndexParser(xml_file).to_dataframe()
            added_this_file = 0
            updated_this_file = 0
            for _, row in df.iterrows():
                name = str(row.get('name', '')).strip()
                cusip = str(row.get('cusip', '')).strip()
                
                if not name:
                    continue
                
                if name in companies:
                    if cusip:
                        companies[name]['cusip'] = cusip
                        stats['updated'] += 1
                        updated_this_file += 1
                else:
                    companies[name] = {'cusip': cusip, 'year': year}
                    stats['added'] += 1
                    added_this_file += 1
            print(f" → +{added_this_file} new, {updated_this_file} updated")
        except Exception as e:
            print(f" → ⚠️ Error: {e}")
    
    return stats


def _get_sorted_files(directory: Path, pattern: str, min_year: int, max_year: int) -> list:
    """Get files matching pattern, sorted by year, within year range."""
    files_with_years = []
    for f in directory.glob(f"Russell{pattern}"):
        year = extract_year_from_filename(f.name)
        if year and min_year <= year <= max_year:
            files_with_years.append((year, f))
    return sorted(files_with_years, key=lambda x: x[0])


def _create_dataframe(companies: Dict) -> pd.DataFrame:
    """Create DataFrame from companies dict with padded CUSIPs."""
    data = [
        {'name': name, 'cusip': pad_cusip(info['cusip']), 'year': info['year']}
        for name, info in companies.items()
    ]
    return pd.DataFrame(data).sort_values(['year', 'name']).reset_index(drop=True)

