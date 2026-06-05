"""Common utility functions for indices parser pipeline."""

import re
from typing import Optional
import pandas as pd


def is_empty_cusip(cusip: str) -> bool:
    """Check if CUSIP is empty or placeholder."""
    if pd.isna(cusip):
        return True
    cusip_str = str(cusip).strip()
    return cusip_str in ("", "0", "000000000")


def pad_cusip(cusip: str, length: int = 9) -> str:
    """Pad CUSIP to specified length with leading zeros."""
    if not cusip:
        return "0" * length
    cusip_clean = str(cusip).strip().upper()
    return cusip_clean.zfill(length)[:length]


def get_first_n_words(text: str, n: int = 3) -> str:
    """Extract first N words from text."""
    if not text or pd.isna(text):
        return ""
    words = str(text).strip().split()
    return " ".join(words[:n])


def year_to_quarter_end(year: int) -> str:
    """Convert year to Q2 quarter end date (June 30 of next year)."""
    if year == 2025:
        return f"{year}-06-30"
    return f"{year + 1}-06-30"


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract 4-digit year from filename."""
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

