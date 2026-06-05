"""PDF parser for SEC Form N-Q filings."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


class PDFIndexParser:
    """Parser for PDF format index files (SEC Form N-Q)."""

    # Pattern: Company Name -> Shares -> Value (handles multi-column layouts)
    COMPANY_PATTERN = re.compile(
        r'(?P<company>[A-Z][A-Za-z0-9\.\-&(),\'/ ]+?)\s+'
        r'(?P<shares>\d{1,3}(?:,\d{3})*)\s+'
        r'\$?\s*(?P<value>\d{1,3}(?:,\d{3})*)',
        re.MULTILINE
    )
    
    SKIP_WORDS = frozenset([
        'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
        'common stocks', 'schedule', 'security', 'shares', 'value',
        'form n-q', 'ishares', 'russell'
    ])

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def parse(self) -> List[Dict[str, str]]:
        """Parse PDF and extract company names."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber required: pip install pdfplumber")
        
        companies = []
        with pdfplumber.open(self.file_path) as pdf:
            # Try table extraction first
            for page in pdf.pages:
                for table in (page.extract_tables() or []):
                    companies.extend(self._parse_table(table))
            
            # Fallback to text extraction
            if not companies:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                companies.extend(self._parse_text(text))
        
        return self._deduplicate(companies)

    def _parse_table(self, table: List) -> List[Dict[str, str]]:
        """Extract company names from table structure."""
        if not table or len(table) < 2:
            return []
        
        results = []
        for row in table[1:]:  # Skip header
            if row and row[0]:
                name = self._clean_name(str(row[0]))
                if self._is_valid_name(name):
                    results.append({'name': name})
        return results

    def _parse_text(self, text: str) -> List[Dict[str, str]]:
        """Extract company names from text using pattern matching."""
        results = []
        for match in self.COMPANY_PATTERN.finditer(text):
            name = self._clean_name(match.group("company"))
            if self._is_valid_name(name):
                results.append({'name': name})
        return results

    def _clean_name(self, name: str) -> str:
        """Clean company name: remove suffixes and footnotes."""
        if not name:
            return name
        # Remove everything after last period
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        # Remove footnote markers like (a), (b)
        name = re.sub(r'\([a-z,]+\)', '', name)
        return name.strip()

    def _is_valid_name(self, name: str) -> bool:
        """Validate company name."""
        return (
            2 <= len(name) <= 150 and
            not name.isdigit() and
            not re.match(r'^\d+', name) and
            name.lower() not in self.SKIP_WORDS
        )

    def _deduplicate(self, companies: List[Dict]) -> List[Dict]:
        """Remove duplicate company names."""
        seen = set()
        unique = []
        for c in companies:
            key = c['name'].lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def to_dataframe(self) -> pd.DataFrame:
        """Parse file and return as DataFrame."""
        companies = self.parse()
        return pd.DataFrame(companies) if companies else pd.DataFrame(columns=['name'])

