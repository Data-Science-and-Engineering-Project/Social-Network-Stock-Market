"""Configurable PostgreSQL handler for GNN database pipeline."""

import os
from dotenv import load_dotenv
from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler

load_dotenv()


class ConfigurablePostgresHandler(PostgresHandler):
    """PostgreSQL handler that accepts explicit database name.

    Extends PostgresHandler to allow connecting to different databases
    on the same server using the same credentials.
    """

    def __init__(self, database: str = None, **kwargs):
        """Initialize handler with explicit database name.

        Args:
            database: Database name to connect to. Overrides DB_NAME env var.
            **kwargs: Additional parameters (currently unused, for future extensibility).
        """
        super().__init__()
        if database:
            self.database = database

    def _get_sql_type(self, dtype) -> str:
        """Override float mapping to DOUBLE PRECISION to preserve full precision.

        Why: parent maps float → NUMERIC(18,4), which truncates portfolio weights
        below 0.00005 to 0 for large portfolios, breaking the weight-sum invariant.
        """
        dtype_str = str(dtype).lower()
        if "float" in dtype_str:
            return "DOUBLE PRECISION"
        return super()._get_sql_type(dtype)
