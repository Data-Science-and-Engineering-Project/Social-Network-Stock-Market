#!/usr/bin/env python3
"""Entry point for GNN Database Pipeline."""

import sys
import argparse
from ETL.gnn_db_pipeline.pipeline import GNNDBPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Build 13FGNN database from Social_13F"
    )
    parser.add_argument(
        "--test-quarter",
        type=str,
        help="Run test on single quarter (e.g., 2017_Q3)",
    )
    args = parser.parse_args()

    try:
        pipeline = GNNDBPipeline()
        pipeline.run(quarter_filter=args.test_quarter)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
