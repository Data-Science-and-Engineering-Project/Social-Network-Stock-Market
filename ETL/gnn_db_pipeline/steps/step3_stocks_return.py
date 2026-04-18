"""Step 3: Compute stocks_return (price_t / price_{t-1}) per (cusip, year, quarter).

Schema:
    cusip TEXT, year SMALLINT, quarter SMALLINT,
    return DOUBLE PRECISION
"""

import pandas as pd
import polars as pl
from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    TARGET_DB,
    TGT_TICKER_PRICES,
    TGT_TICKER_TO_CUSIP,
    TGT_STOCKS_RETURN,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler


def run():
    """Build stocks_return: one row per (cusip, year, quarter) with return vs previous quarter."""
    logger = ETLLogger(name="Step3_StocksReturn", console_output=True)
    logger.info("=" * 80)
    logger.info("STEP 3: CREATE STOCKS_RETURN")
    logger.info("=" * 80)

    handler = None
    try:
        handler = ConfigurablePostgresHandler(TARGET_DB)
        handler.connect()

        # Load reference tables (already copied to target by step2)
        logger.info("Loading ticker_prices...")
        df_tp = handler.query(
            f"SELECT ticker, period_start, price FROM {TGT_TICKER_PRICES}"
        )
        logger.info(f"  Loaded {len(df_tp)} price rows")

        logger.info("Loading ticker_to_cusip...")
        df_t2c = handler.query(
            f"SELECT ticker, cusip, trading_start_date, trading_end_date "
            f"FROM {TGT_TICKER_TO_CUSIP}"
        )
        logger.info(f"  Loaded {len(df_t2c)} ticker mappings")

        # Normalize date types
        df_tp["period_start"] = pd.to_datetime(df_tp["period_start"])
        df_t2c["trading_start_date"] = pd.to_datetime(df_t2c["trading_start_date"])
        df_t2c["trading_end_date"] = pd.to_datetime(df_t2c["trading_end_date"])

        # Join prices with ticker_to_cusip
        logger.info("Joining ticker_prices with ticker_to_cusip...")
        df = df_tp.merge(df_t2c, on="ticker", how="inner")
        logger.info(f"  After join: {len(df)} rows")

        # Filter to valid ticker/cusip assignment windows
        df = df[
            (df["period_start"] >= df["trading_start_date"])
            & (
                df["trading_end_date"].isna()
                | (df["period_start"] <= df["trading_end_date"])
            )
        ]
        logger.info(f"  After date filter: {len(df)} rows")

        # Derive year, quarter from period_start
        df["year"] = df["period_start"].dt.year.astype("int16")
        df["quarter"] = df["period_start"].dt.quarter.astype("int16")

        # Collapse to one price per (cusip, year, quarter) — mean in case of any duplicates
        df_q = (
            df.groupby(["cusip", "year", "quarter"], as_index=False)["price"].mean()
        )
        logger.info(f"  Per (cusip, year, quarter): {len(df_q)} rows")

        # Compute return = price_t / price_{t-1}, ordered by (year, quarter) within each cusip
        logger.info("Computing quarterly returns...")
        pdf = pl.from_pandas(df_q).sort(["cusip", "year", "quarter"])
        pdf = pdf.with_columns(
            pl.col("price").shift(1).over("cusip").alias("price_prev")
        ).with_columns(
            (pl.col("price") / pl.col("price_prev")).alias("return")
        )
        # First quarter per cusip has null return — drop it
        pdf = (
            pdf.filter(pl.col("return").is_not_null())
            .select("cusip", "year", "quarter", "return")
        )
        result = pdf.to_pandas()
        logger.info(f"Final stocks_return: {len(result)} rows")

        # Drop & recreate with explicit schema
        logger.info(f"Creating table {TGT_STOCKS_RETURN}...")
        handler.execute(f"DROP TABLE IF EXISTS {TGT_STOCKS_RETURN}")
        handler.execute(f"""
            CREATE TABLE {TGT_STOCKS_RETURN} (
                cusip TEXT,
                year SMALLINT,
                quarter SMALLINT,
                "return" DOUBLE PRECISION
            )
        """)

        count = handler._copy_dataframe(TGT_STOCKS_RETURN, result)
        logger.info(f"Inserted {count} rows")
        logger.info("Step 3 completed")

    except Exception as e:
        logger.error(f"Step 3 failed: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        if handler:
            handler.disconnect()
        logger.close()
