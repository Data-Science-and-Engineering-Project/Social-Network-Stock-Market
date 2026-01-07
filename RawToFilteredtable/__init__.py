"""
RawToFilteredTable Pipeline Module

This module provides the complete pipeline to process Russell 3000 index data
from raw files to filtered holdings in the database.

Main entry point: run_full_pipeline.py

Phases:
1. indices_parser: Parse index files, update CUSIPs, map TICKERs, extract prices
2. filterholdings: Filter holdings by relevant CUSIPs for each quarter
"""

# Removed import to avoid circular import issues
# Import run_full_pipeline directly when needed: from RawToFilteredtable.run_full_pipeline import run_full_pipeline

__all__ = []



