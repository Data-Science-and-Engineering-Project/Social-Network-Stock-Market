"""GNN Database Pipeline orchestrator."""

from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    SOURCE_DB,
    TARGET_DB,
    SRC_HOLDINGS_FILTERED,
    TGT_NORMALIZED_HOLDINGS,
    TGT_CIK_AUM,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler
from ETL.gnn_db_pipeline.steps import (
    step1_create_database,
    step2_copy_tables,
    step3_stocks_return,
    step4_normalized_holdings,
    step5_cik_aum,
    step6_changed_holdings,
    step8_changed_stas,
)


class GNNDBPipeline:
    """Orchestrates all steps to build the 13FGNN database from Social_13F."""

    def __init__(self):
        self.logger = ETLLogger(name="GNNDBPipeline", console_output=True)

    def _get_all_quarters(self):
        """Discover all (year, quarter) pairs in the source holdings table."""
        handler = ConfigurablePostgresHandler(SOURCE_DB)
        handler.connect()
        try:
            df = handler.query(
                f"SELECT DISTINCT year, quarter FROM {SRC_HOLDINGS_FILTERED} "
                f"ORDER BY year, quarter"
            )
            return [f"{int(r['year'])}_Q{int(r['quarter'])}" for _, r in df.iterrows()]
        finally:
            handler.disconnect()

    def _reset_per_quarter_tables(self):
        """Drop normalized_holdings and cik_aum so per-quarter appends build fresh schemas."""
        handler = ConfigurablePostgresHandler(TARGET_DB)
        handler.connect()
        try:
            handler.execute(f"DROP TABLE IF EXISTS {TGT_NORMALIZED_HOLDINGS}")
            handler.execute(f"DROP TABLE IF EXISTS {TGT_CIK_AUM}")
        finally:
            handler.disconnect()

    def run(self, quarter_filter: str = None):
        """Run the full pipeline.

        Args:
            quarter_filter: Optional quarter filter (e.g., '2017_Q3') for testing.
                           If None, iterate over all quarters found in the source.
        """
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("GNN DATABASE PIPELINE")
            self.logger.info("=" * 80)

            step1_create_database.run()
            step2_copy_tables.run()
            step3_stocks_return.run()

            if quarter_filter:
                quarters = [quarter_filter]
            else:
                quarters = self._get_all_quarters()
                self.logger.info(f"Processing {len(quarters)} quarters: "
                                 f"{quarters[0]} -> {quarters[-1]}")

            # Reset per-quarter tables so the new schema (shares/price) applies
            self._reset_per_quarter_tables()

            for i, q in enumerate(quarters, start=1):
                self.logger.info("\n" + "#" * 80)
                self.logger.info(f"QUARTER {i}/{len(quarters)}: {q}")
                self.logger.info("#" * 80)
                step4_normalized_holdings.run(quarter_filter=q)
                step5_cik_aum.run(quarter_filter=q)

            step6_changed_holdings.run()
            step8_changed_stas.run()

            self.logger.info("\n" + "=" * 80)
            self.logger.info("GNN PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            self.logger.close()
