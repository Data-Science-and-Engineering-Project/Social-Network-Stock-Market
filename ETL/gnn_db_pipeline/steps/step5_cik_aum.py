"""Step 5: Create cik_aum table (total AUM per fund per quarter) using pandas."""

import pandas as pd
from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    SOURCE_DB,
    TARGET_DB,
    SRC_HOLDINGS_FILTERED,
    SRC_TICKER_PRICES,
    SRC_TICKER_TO_CUSIP,
    TGT_CIK_AUM,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler


def run(quarter_filter: str = None):
    """Create cik_aum table using pandas computation.

    Args:
        quarter_filter: Optional filter like '2017_Q3' for testing. None = all data.
    """
    logger = ETLLogger(name="Step5_CIK_AUM", console_output=True)
    logger.info("=" * 80)
    logger.info("STEP 5: CREATE CIK_AUM (PANDAS)")
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

        # Filter by date range (period_start >= trading_start_date AND period_start <= trading_end_date)
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

        # Compute AUM (sum of sshprnamt * price per cik, year, quarter)
        logger.info("Computing AUM totals...")
        df_join2["position_value"] = df_join2["sshprnamt"] * df_join2["price"]

        result = (
            df_join2.groupby(["cik", "year", "quarter"])["position_value"]
            .sum()
            .reset_index()
            .rename(columns={"position_value": "total"})
        )

        result = result.sort_values(["cik", "year", "quarter"]).reset_index(drop=True)

        logger.info(f"Final cik_aum: {len(result)} records")

        # Create table and insert data
        logger.info(f"Creating table {TGT_CIK_AUM}...")
        if not tgt_handler.create_table(TGT_CIK_AUM, result):
            logger.error(f"Failed to create table {TGT_CIK_AUM}")
            raise Exception(f"Failed to create table {TGT_CIK_AUM}")

        logger.info(f"Inserting {len(result)} rows into {TGT_CIK_AUM}...")
        count = tgt_handler._copy_dataframe(TGT_CIK_AUM, result)
        logger.info(f"Inserted {count} rows")

        logger.info("Step 5 completed")

    except Exception as e:
        logger.error(f"Step 5 failed: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        if src_handler:
            src_handler.disconnect()
        if tgt_handler:
            tgt_handler.disconnect()
        logger.close()
