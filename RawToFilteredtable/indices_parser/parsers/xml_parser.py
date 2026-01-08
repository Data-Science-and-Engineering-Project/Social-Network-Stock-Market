"""XML parser for SEC Form N-Q NPORT-P format index files."""

from pathlib import Path
from typing import Dict, List, Optional, Union
from xml.etree import ElementTree as ET
import pandas as pd


class XMLIndexParser:
    """Parser for XML format index files (SEC Form N-Q NPORT-P)."""

    NAMESPACES = {
        'nport': 'http://www.sec.gov/edgar/nport',
        'com': 'http://www.sec.gov/edgar/common',
        'ncom': 'http://www.sec.gov/edgar/nportcommon'
    }

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"XML file not found: {file_path}")

    def parse(self) -> List[Dict[str, Union[str, float]]]:
        """Parse XML file and extract all investment securities."""
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        
        securities = (
            root.findall('.//nport:invstOrSec', self.NAMESPACES) or
            root.findall('.//invstOrSec')
        )
        
        return [s for s in (self._parse_security(sec) for sec in securities) if s]

    def _parse_security(self, elem: ET.Element) -> Optional[Dict]:
        """Parse single security element."""
        try:
            data = {
                'name': self._get_text(elem, 'name'),
                'cusip': self._get_text(elem, 'cusip'),
                'ticker': self._get_ticker(elem),
            }
            return data if data['name'] else None
        except Exception:
            return None

    def _get_text(self, elem: ET.Element, tag: str) -> str:
        """Get text from element, trying with and without namespace."""
        for ns_uri in [''] + list(self.NAMESPACES.values()):
            path = f'.//{{{ns_uri}}}{tag}' if ns_uri else f'.//{tag}'
            found = elem.find(path)
            if found is not None and found.text:
                return found.text.strip()
        return ''

    def _get_ticker(self, elem: ET.Element) -> str:
        """Extract ticker from identifiers element."""
        for ns_uri in [''] + list(self.NAMESPACES.values()):
            path = f'.//{{{ns_uri}}}identifiers' if ns_uri else './/identifiers'
            ids = elem.find(path)
            if ids is not None:
                ticker = ids.find('.//ticker')
                if ticker is not None:
                    return ticker.get('value', '')
        return ''

    def to_dataframe(self) -> pd.DataFrame:
        """Parse file and return as DataFrame."""
        securities = self.parse()
        return pd.DataFrame(securities) if securities else pd.DataFrame()

