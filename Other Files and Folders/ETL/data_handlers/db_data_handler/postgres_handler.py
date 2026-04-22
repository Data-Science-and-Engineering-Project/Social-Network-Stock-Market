import io
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch
from typing import Optional, Set, Tuple
import pandas as pd
from ETL.data_handlers.db_data_handler.db_abstract import AbstractDBHandler
from ETL.logger.logger import ETLLogger

load_dotenv()


class PostgresHandler(AbstractDBHandler):
    """PostgreSQL database handler with automatic partition management."""

    def __init__(self,):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.database = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.connection: Optional[psycopg2.extensions.connection] = None
    # ==================== CONNECTION ====================

    def connect(self) -> bool:
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=5,
            )
            ETLLogger().info(
                f"Connected to PostgreSQL: {self.user}@{self.host}:{self.port}/{self.database}"
            )
            return True
        except psycopg2.Error as e:
            ETLLogger().error(f"PostgreSQL connection failed: {str(e)}")
            self.connection = None
            return False

    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None
            ETLLogger().info("Disconnected from PostgreSQL")

    # ==================== PUBLIC API ====================

    # ==================== INSERT DATAFRAME ====================
    
    def create_table(self, table_name: str, df: pd.DataFrame) -> bool:
            """
            Create table based on DataFrame schema.
            Thread-safe: Uses IF NOT EXISTS to handle concurrent creation attempts.

            Args:
                table_name: Name of table to create
                df: DataFrame with schema to use

            Returns:
                True if successful, False otherwise
            """
            try:
                if not self.connection:
                    self.connect()

                cursor = self.connection.cursor()

                # Check if table already exists (thread-safe check)
                if self.table_exists(table_name):
                    cursor.close()
                    return True

                # Generate column definitions from DataFrame dtypes
                columns = []
                for col_name, dtype in zip(df.columns, df.dtypes):
                    sql_type = self._get_sql_type(dtype)
                    columns.append(f'"{col_name}" {sql_type}')

                quoted_table_name = self._quote_table_name(table_name)
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {quoted_table_name} ({', '.join(columns)})"

                cursor.execute(create_table_sql)
                self.connection.commit()
                ETLLogger().info(f"Table '{table_name}' created successfully")
                cursor.close()
                return True
            except psycopg2.Error as e:
                error_msg = str(e)
                self.connection.rollback()
                if 'cursor' in locals():
                    try:
                        cursor.close()
                    except:
                        pass
                # Check if it's a "table already exists" error (race condition)
                if 'already exists' in error_msg.lower():
                    ETLLogger().warning(f"Table '{table_name}' already exists (race), continuing")
                    return True
                ETLLogger().error(f"Failed to create table '{table_name}': {error_msg}")
                return False
            except Exception as e:
                ETLLogger().error(f"Failed to create table '{table_name}': {str(e)}")
                if self.connection:
                    self.connection.rollback()
                if 'cursor' in locals():
                    try:
                        cursor.close()
                    except:
                        pass
                return False
    
    def _get_sql_type(self, dtype) -> str:
        """Convert pandas dtype to PostgreSQL type."""
        dtype_str = str(dtype).lower()
        if 'int' in dtype_str:
            return 'BIGINT'
        elif 'float' in dtype_str:
            return 'NUMERIC(18,4)'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        elif 'date' in dtype_str:
            return 'DATE'
        else:
            return 'TEXT'


    def insert_dataframe_regular(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "append"
    ) -> int:
        """
        Insert DataFrame records into PostgreSQL table.

        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: 'fail', 'replace', or 'append'

        Returns:
            Number of records inserted
        """
        if not self.connection:
            self.connect()

        try:
            # Handle if_exists logic
            if if_exists == "replace":
                self.drop_table(table_name)

            # Create table if it doesn't exist
            if not self.table_exists(table_name):
                self.create_table(table_name, df)

            # Insert data
            insert_count = self._copy_dataframe(table_name, df)
            ETLLogger().info(f"Loaded {insert_count} records into '{table_name}' table")
            return insert_count

        except Exception as e:
            ETLLogger().error(f"Failed to insert DataFrame: {str(e)}")
            return 0


    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> int:
        if not self.connection and not self.connect():
            ETLLogger().error("Failed to establish database connection")
            return 0

        try:
            df = self._add_period_start(df)

            # schema setup (once)
            self._ensure_parent_table_exists(table_name)
            self._ensure_partitions_exist(table_name, df)
            # self._ensure_indexes_exist(table_name)

            total_inserted = 0

            # 🔥 split by business truth: report period
            for (year, quarter), chunk in df.groupby(["year", "quarter"]):
                ETLLogger().info(f"Loading {year} Q{quarter} ({len(chunk)} rows)")
                inserted = self._copy_dataframe(table_name, chunk)
                total_inserted += inserted

            ETLLogger().info(f"Loaded total {total_inserted} records into '{table_name}'")
            return total_inserted

        except Exception as e:
            ETLLogger().error(f"Insert failed: {str(e)}")
            self.connection.rollback()
            return 0


    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            if not self.connection:
                self.connect()
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT to_regclass('{table_name}');")
            result = cursor.fetchone()[0] is not None
            cursor.close()
            return result
        except psycopg2.Error as e:
            ETLLogger().error(f"Failed to check table existence: {str(e)}")
            return False

    def drop_table(self, table_name: str) -> bool:
        """Drop a table if it exists."""
        try:
            if not self.connection:
                self.connect()
            cursor = self.connection.cursor()
            quoted_table_name = self._quote_table_name(table_name)
            cursor.execute(f"DROP TABLE IF EXISTS {quoted_table_name}")
            self.connection.commit()
            cursor.close()
            ETLLogger().info(f"Table '{table_name}' dropped")
            return True
        except psycopg2.Error as e:
            ETLLogger().error(f"Failed to drop table '{table_name}': {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False

    def _quote_table_name(self, table_name: str) -> str:
        """Quote table name to handle special characters."""
        return f'"{table_name}"'

    # ==================== SCHEMA MANAGEMENT ====================

    def _ensure_parent_table_exists(self, table_name: str) -> None:
        """Create partitioned parent table if it does not exist."""
        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            );
            """,
            (table_name,),
        )

        exists = cursor.fetchone()[0]

        if not exists:
            ETLLogger().info(f"Creating parent table '{table_name}'")

            cursor.execute(
                f"""
                CREATE TABLE {table_name} (
                    accessionnumber TEXT,
                    infotablesk TEXT,
                    nameofissuer TEXT,
                    cusip TEXT,
                    value NUMERIC,
                    sshprnamt BIGINT,
                    filingdate DATE,
                    cik TEXT,
                    value_per_share NUMERIC,
                    year INT,
                    quarter INT,
                    period_start DATE NOT NULL
                )
                PARTITION BY RANGE (period_start);
                """
            )
            self.connection.commit()

        cursor.close()

    # def _ensure_indexes_exist(self, table_name: str) -> None:
    #     """Create indexes on parent table (propagated to partitions)."""
    #     cursor = self.connection.cursor()
    #
    #     indexes = {
    #         f"{table_name}_cusip_idx": "cusip",
    #         f"{table_name}_cik_idx": "cik",
    #     }
    #
    #     for index_name, column in indexes.items():
    #         cursor.execute(
    #             """
    #             SELECT EXISTS (
    #                 SELECT 1
    #                 FROM pg_indexes
    #                 WHERE schemaname = 'public'
    #                   AND indexname = %s
    #             );
    #              """,
    #             (index_name,),
    #         )
    #
    #         if not cursor.fetchone()[0]:
    #             ETLLogger().info(f"Creating index '{index_name}'")
    #             cursor.execute(
    #                 f'CREATE INDEX {index_name} ON {table_name} ("{column}");'
    #             )
    #
    #     self.connection.commit()
    #     cursor.close()

    def _ensure_partitions_exist(self, table_name: str, df: pd.DataFrame) -> None:
        """Create missing quarterly partitions based on DataFrame contents."""

        cursor = self.connection.cursor()

        partitions: Set[Tuple[int, int]] = set(
            zip(df["year"].astype(int), df["quarter"].astype(int))
        )

        for year, quarter in partitions:
            start_month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
            start_date = f"{year}-{start_month:02d}-01"

            if quarter == 4:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{start_month + 3:02d}-01"

            partition_name = f"{table_name}_{year}_q{quarter}"

            # Check if partition already exists
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = %s
                );
                """,
                (partition_name,),
            )

            if not cursor.fetchone()[0]:
                try:
                    ETLLogger().info(f"Creating partition '{partition_name}'")
                    cursor.execute(
                        f"""
                        CREATE TABLE {partition_name}
                        PARTITION OF {table_name}
                        FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                        """
                    )
                except psycopg2.Error as e:
                    ETLLogger().warning(f"Failed to create partition '{partition_name}': {str(e)}")
                    self.connection.rollback()
                    cursor = self.connection.cursor()
            else:
                ETLLogger().info(f"Partition '{partition_name}' already exists")

        self.connection.commit()
        cursor.close()

    # ==================== DATA INSERT ====================

    @staticmethod
    def _add_period_start(df: pd.DataFrame) -> pd.DataFrame:
        if "year" not in df.columns or "quarter" not in df.columns:
            raise ValueError("DataFrame must include 'year' and 'quarter'")

        quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}

        df["period_start"] = pd.to_datetime(
            df["year"].astype(str)
            + "-"
            + df["quarter"].map(quarter_to_month).astype(str)
            + "-01"
        )

        return df

    # def _insert_data(self, table_name: str, df: pd.DataFrame) -> int:
    #     cursor = self.connection.cursor()
    #
    #     columns = df.columns.tolist()
    #     placeholders = ", ".join(["%s"] * len(columns))
    #
    #     insert_sql = f"""
    #         INSERT INTO {table_name} (
    #             {", ".join(f'"{col}"' for col in columns)}
    #         ) VALUES ({placeholders});
    #     """
    #
    #     records = (
    #         df.replace({pd.NaT: None})
    #         .where(pd.notna(df), None)
    #         .values
    #         .tolist()
    #     )
    #
    #     execute_batch(cursor, insert_sql, records, page_size=5000)
    #
    #     self.connection.commit()
    #     cursor.close()
    #
    #     return len(records)

    def _copy_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        cursor = self.connection.cursor()

        buffer = io.StringIO()
        df.to_csv(
            buffer,
            index=False,
            header=False,
            na_rep="\\N"
        )
        buffer.seek(0)

        columns = ", ".join(f'"{c}"' for c in df.columns)

        copy_sql = f"""
            COPY {table_name} ({columns})
            FROM STDIN
            WITH (FORMAT CSV, NULL '\\N')
        """

        cursor.copy_expert(copy_sql, buffer)
        self.connection.commit()
        cursor.close()

        return len(df)

