"""Unified parser with automatic file type detection."""

from pathlib import Path
from typing import Union
import pandas as pd

from .xml_parser import XMLIndexParser
from .pdf_parser import PDFIndexParser


class IndexParser:
    """Unified parser that auto-detects file type (XML or PDF)."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        suffix = self.file_path.suffix.lower()
        
        if suffix == '.xml':
            self.parser = XMLIndexParser(file_path)
        elif suffix == '.pdf':
            self.parser = PDFIndexParser(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .xml or .pdf")

    def parse(self):
        """Parse file based on detected type."""
        return self.parser.parse()

    def to_dataframe(self) -> pd.DataFrame:
        """Parse file and return as DataFrame."""
        return self.parser.to_dataframe()


def parse_index_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """Convenience function to parse an index file."""
    return IndexParser(file_path).to_dataframe()

