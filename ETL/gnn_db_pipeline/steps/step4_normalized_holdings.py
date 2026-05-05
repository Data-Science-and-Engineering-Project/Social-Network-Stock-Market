"""Step 4: Create normalized_holdings table using pandas.

Schema:
    cik TEXT, cusip TEXT, year SMALLINT, quarter SMALLINT,
    shares BIGINT, price DOUBLE PRECISION, weight DOUBLE PRECISION
"""

import pandas as pd
from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    SOURCE_DB,
    TARGET_DB,
    SRC_HOLDINGS_FILTERED,
    SRC_TICKER_PRICES,
    SRC_TICKER_TO_CUSIP,
    TGT_NORMALIZED_HOLDINGS,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler


def run(quarter_filter: str = None):
    """Create normalized_holdings table using pandas computation.

    Args:
        quarter_filter: Optional filter like '2017_Q3' for testing. None = all data.
    """
    logger = ETLLogger(name="Step4_NormalizedHoldings", console_output=True)
    logger.info("=" * 80)
    logger.info("STEP 4: CREATE NORMALIZED_HOLDINGS (PANDAS)")
    logger.info("=" * 80)

    src_handler = None
    tgt_handler = None

    try:
        # Initialize handlers
        src_handler = ConfigurablePostgresHandler(SOURCE_DB)
        tgt_handler = ConfigurablePostgresHandler(TARGET_DB)

        src_handler.connect()
        tgt_handler.connect()

        # Build filter clause for loading data
        where_clause = ""
        if quarter_filter:
            logger.info(f"Filtering to quarter: {quarter_filter}")
            parts = quarter_filter.split("_Q")
            if len(parts) == 2:
                year = parts[0]
                quarter = parts[1]
                where_clause = f"WHERE year = {year} AND quarter = {quarter}"

        # Load data from source DB
        logger.info("Loading holdings data...")
        query_holdings = f"SELECT * FROM {SRC_HOLDINGS_FILTERED} {where_clause}"
        df_h = src_handler.query(query_holdings)
        logger.info(f"  Loaded {len(df_h)} holdings records")

        logger.info("Loading ticker_to_cusip mapping...")
        df_t2c = src_handler.query(f"SELECT * FROM {SRC_TICKER_TO_CUSIP}")
        logger.info(f"  Loaded {len(df_t2c)} ticker mappings")

        logger.info("Loading ticker_prices...")
        df_tp = src_handler.query(f"SELECT * FROM {SRC_TICKER_PRICES}")
        logger.info(f"  Loaded {len(df_tp)} price records")

        # Convert date columns to datetime for proper comparison
        logger.info("Converting date columns...")
        df_h["period_start"] = pd.to_datetime(df_h["period_start"])
        df_t2c["trading_start_date"] = pd.to_datetime(df_t2c["trading_start_date"])
        df_t2c["trading_end_date"] = pd.to_datetime(df_t2c["trading_end_date"])
        df_tp["period_start"] = pd.to_datetime(df_tp["period_start"])

        # Join holdings with ticker_to_cusip on CUSIP
        logger.info("Joining holdings with ticker_to_cusip...")
        df_join1 = df_h.merge(df_t2c, on="cusip", how="inner")
        logger.info(f"  After join: {len(df_join1)} records")

        # Filter by date range
        logger.info("Filtering by trading date ranges...")
        df_join1 = df_join1[
            (df_join1["period_start"] >= df_join1["trading_start_date"])
            & (
                (df_join1["trading_end_date"].isna())
                | (df_join1["period_start"] <= df_join1["trading_end_date"])
            )
        ]
        logger.info(f"  After date filter: {len(df_join1)} records")

        # Join with ticker_prices (INNER JOIN - drops holdings without matching prices)
        logger.info("Joining with ticker_prices (dropping holdings without prices)...")
        holdings_before = len(df_join1)
        df_join2 = df_join1.merge(
            df_tp[["ticker", "period_start", "price"]],
            on=["ticker", "period_start"],
            how="inner",
        )
        holdings_dropped = holdings_before - len(df_join2)
        logger.info(f"  After join: {len(df_join2)} records (dropped {holdings_dropped} without prices)")

        # Compute position values
        logger.info("Computing position values...")
        df_join2["position_value"] = df_join2["sshprnamt"] * df_join2["price"]

        # Remove holdings with near-zero position values BEFORE computing portfolio totals
        logger.info("Removing holdings with near-zero position values...")
        holdings_before = len(df_join2)
        df_join2 = df_join2[df_join2["position_value"] > 1e-6]
        holdings_removed = holdings_before - len(df_join2)
        if holdings_removed > 0:
            logger.info(f"  Removed {holdings_removed} holdings with near-zero position value")

        # Compute portfolio totals (only from remaining holdings)
        logger.info("Computing portfolio totals...")
        portfolio_totals = (
            df_join2.groupby(["cik", "year", "quarter"])["position_value"]
            .sum()
            .reset_index()
            .rename(columns={"position_value": "total_value"})
        )

        # Merge totals back and compute weights
        logger.info("Computing weights...")
        result = df_join2.merge(
            portfolio_totals, on=["cik", "year", "quarter"], how="inner"
        )
        result["weight"] = result["position_value"] / result["total_value"]

        # Renormalize weights to sum to exactly 1.0 per portfolio (handles floating-point precision)
        logger.info("Renormalizing weights to sum to 1.0...")
        result["weight"] = result.groupby(["cik", "year", "quarter"])["weight"].transform(
            lambda x: x / x.sum()
        )

        # Rename sshprnamt -> shares for output
        result = result.rename(columns={"sshprnamt": "shares"})

        # Select final columns (columns needed for GNN + raw inputs for delta computation)
        final_columns = [
            "cik",
            "cusip",
            "year",
            "quarter",
            "shares",
            "price",
            "weight",
        ]
        result = result[final_columns]

        logger.info(f"Final normalized_holdings: {len(result)} records")

        # Verify weight sums are ~1.0
        weight_sums = result.groupby(["cik", "year", "quarter"])["weight"].sum()
        invalid_weights = (weight_sums - 1.0).abs() > 1e-6
        if invalid_weights.any():
            logger.warning(f"  {invalid_weights.sum()} portfolios have weight sums deviating > 1e-6 from 1.0")
            for idx, row in weight_sums[invalid_weights].items():
                logger.warning(f"    {idx}: {row:.10f}")
        else:
            logger.info("  ✓ All portfolio weight sums are valid (== 1.0)")

        # Create table and insert. When running the full pipeline, the first quarter
        # drops the old table so the new schema applies; subsequent quarters append.
        logger.info(f"Creating table {TGT_NORMALIZED_HOLDINGS}...")
        if not tgt_handler.create_table(TGT_NORMALIZED_HOLDINGS, result):
            logger.error(f"Failed to create table {TGT_NORMALIZED_HOLDINGS}")
            raise Exception(f"Failed to create table {TGT_NORMALIZED_HOLDINGS}")

        logger.info(f"Inserting {len(result)} rows...")
        count = tgt_handler._copy_dataframe(TGT_NORMALIZED_HOLDINGS, result)
        logger.info(f"Inserted {count} rows")

        logger.info("Step 4 completed")

    except Exception as e:
        logger.error(f"Step 4 failed: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        if src_handler:
            src_handler.disconnect()
        if tgt_handler:
            tgt_handler.disconnect()
        logger.close()
