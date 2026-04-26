"""Step 2: Copy reference tables from Social_13F to 13FGNN."""

from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    SOURCE_DB,
    TARGET_DB,
    SRC_TICKER_PRICES,
    SRC_TICKER_TO_CUSIP,
    TGT_TICKER_PRICES,
    TGT_TICKER_TO_CUSIP,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler


def run():
    """Copy ticker_prices and ticker_to_cusip tables from source to target."""
    logger = ETLLogger(name="Step2_CopyTables", console_output=True)
    logger.info("=" * 80)
    logger.info("STEP 2: COPY REFERENCE TABLES")
    logger.info("=" * 80)

    src_handler = None
    tgt_handler = None

    try:
        # Initialize handlers
        src_handler = ConfigurablePostgresHandler(SOURCE_DB)
        tgt_handler = ConfigurablePostgresHandler(TARGET_DB)

        src_handler.connect()
        tgt_handler.connect()

        # Copy ticker_prices
        logger.info(f"Copying {SRC_TICKER_PRICES}...")
        df_prices = src_handler.query(f"SELECT * FROM {SRC_TICKER_PRICES}")
        logger.info(f"  Read {len(df_prices)} rows from {SRC_TICKER_PRICES}")

        if not tgt_handler.create_table(TGT_TICKER_PRICES, df_prices):
            logger.error(f"Failed to create table {TGT_TICKER_PRICES}")
            raise Exception(f"Failed to create table {TGT_TICKER_PRICES}")

        count = tgt_handler._copy_dataframe(TGT_TICKER_PRICES, df_prices)
        logger.info(f"  Inserted {count} rows into {TGT_TICKER_PRICES}")

        # Copy ticker_to_cusip
        logger.info(f"Copying {SRC_TICKER_TO_CUSIP}...")
        df_cusip = src_handler.query(f"SELECT * FROM {SRC_TICKER_TO_CUSIP}")
        logger.info(f"  Read {len(df_cusip)} rows from {SRC_TICKER_TO_CUSIP}")

        if not tgt_handler.create_table(TGT_TICKER_TO_CUSIP, df_cusip):
            logger.error(f"Failed to create table {TGT_TICKER_TO_CUSIP}")
            raise Exception(f"Failed to create table {TGT_TICKER_TO_CUSIP}")

        count = tgt_handler._copy_dataframe(TGT_TICKER_TO_CUSIP, df_cusip)
        logger.info(f"  Inserted {count} rows into {TGT_TICKER_TO_CUSIP}")

        logger.info("Step 2 completed")

    except Exception as e:
        logger.error(f"Step 2 failed: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        if src_handler:
            src_handler.disconnect()
        if tgt_handler:
            tgt_handler.disconnect()
        logger.close()
