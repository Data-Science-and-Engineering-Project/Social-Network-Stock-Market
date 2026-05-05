"""Step 8: Quarterly graph stats on changed_holdings + log-return tertiles.

Builds `changed_stas` one quarter per query: average in-degree per cusip, average out-degree per CIK, counts,
and mean deltas. Ensures `stocks_return.log_return` is populated, then writes
per-quarter 1/3 and 2/3 `log_return` thresholds into `changed_stas`.
"""

import pandas as pd
from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    TARGET_DB,
    TGT_CHANGED_HOLDINGS,
    TGT_CHANGED_STAS,
    TGT_STOCKS_RETURN,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler


def _sql_stats_one_quarter() -> str:
    return f"""
WITH base AS (
    SELECT cik, cusip, change_in_shares, change_in_weight, change_in_adjusted_weight
    FROM {TGT_CHANGED_HOLDINGS}
    WHERE year = %s AND quarter = %s
),
per_cusip AS (
    SELECT cusip, COUNT(DISTINCT cik)::double precision AS deg_in
    FROM base
    GROUP BY cusip
),
per_cik AS (
    SELECT cik, COUNT(DISTINCT cusip)::double precision AS deg_out
    FROM base
    GROUP BY cik
),
agg AS (
    SELECT
        COUNT(DISTINCT cusip)::bigint AS num_cusip,
        COUNT(DISTINCT cik)::bigint AS num_cik,
        AVG(change_in_shares::double precision) AS avg_change_in_shares,
        AVG(change_in_weight) AS avg_change_in_weight,
        AVG(change_in_adjusted_weight) AS avg_change_in_adjusted_weight
    FROM base
)
SELECT
    %s::smallint AS year,
    %s::smallint AS quarter,
    (SELECT AVG(deg_in) FROM per_cusip) AS avg_degree_in_cusip,
    (SELECT AVG(deg_out) FROM per_cik) AS avg_degree_out_cik,
    agg.num_cusip,
    agg.num_cik,
    agg.avg_change_in_shares,
    agg.avg_change_in_weight,
    agg.avg_change_in_adjusted_weight
FROM agg
"""


def _ensure_log_return(handler: ConfigurablePostgresHandler, logger: ETLLogger) -> None:
    handler.execute(
        f"ALTER TABLE {TGT_STOCKS_RETURN} ADD COLUMN IF NOT EXISTS log_return DOUBLE PRECISION"
    )
    handler.execute(
        f"UPDATE {TGT_STOCKS_RETURN} "
        f"SET log_return = CASE WHEN return IS NOT NULL AND return > 0 "
        f"THEN LN(return) ELSE NULL END"
    )
    n = handler.query(
        f"SELECT COUNT(*) AS n FROM {TGT_STOCKS_RETURN} WHERE log_return IS NOT NULL"
    )
    logger.info(
        f"{TGT_STOCKS_RETURN}.log_return populated for {int(n.iloc[0, 0])} rows."
    )


def _apply_tertiles(handler: ConfigurablePostgresHandler, logger: ETLLogger) -> None:
    handler.execute(
        f"ALTER TABLE {TGT_CHANGED_STAS} ADD COLUMN IF NOT EXISTS "
        f"log_return_tertile_1 DOUBLE PRECISION"
    )
    handler.execute(
        f"ALTER TABLE {TGT_CHANGED_STAS} ADD COLUMN IF NOT EXISTS "
        f"log_return_tertile_2 DOUBLE PRECISION"
    )
    sql = f"""
WITH tert AS (
    SELECT
        year,
        quarter,
        percentile_cont(1.0 / 3.0) WITHIN GROUP (ORDER BY log_return) AS t1,
        percentile_cont(2.0 / 3.0) WITHIN GROUP (ORDER BY log_return) AS t2
    FROM {TGT_STOCKS_RETURN}
    WHERE log_return IS NOT NULL
    GROUP BY year, quarter
)
UPDATE {TGT_CHANGED_STAS} AS c
SET
    log_return_tertile_1 = t.t1,
    log_return_tertile_2 = t.t2
FROM tert AS t
WHERE c.year = t.year AND c.quarter = t.quarter
"""
    handler.execute(sql)
    logger.info(
        f"Updated {TGT_CHANGED_STAS} with per-quarter log_return tertile thresholds "
        f"(1/3, 2/3)."
    )


def run():
    """Populate `changed_stas` and tertile columns; refresh `stocks_return.log_return`."""
    logger = ETLLogger(name="step8_changed_stas", console_output=True)
    handler = None
    sql_stats = _sql_stats_one_quarter()

    try:
        logger.info(f"Step 8: building {TGT_CHANGED_STAS} on {TARGET_DB}")
        handler = ConfigurablePostgresHandler(TARGET_DB)
        handler.connect()

        _ensure_log_return(handler, logger)

        quarters_df = handler.query(
            f"SELECT DISTINCT year, quarter FROM {TGT_CHANGED_HOLDINGS} "
            f"ORDER BY year, quarter"
        )
        if quarters_df.empty:
            logger.error(
                f"No rows in {TGT_CHANGED_HOLDINGS}; cannot build {TGT_CHANGED_STAS}."
            )
            raise ValueError(f"empty {TGT_CHANGED_HOLDINGS}")

        quarter_rows = []
        n_q = len(quarters_df)
        for idx, r in enumerate(quarters_df.itertuples(index=False), start=1):
            y, q = int(r.year), int(r.quarter)
            one = handler.query(sql_stats, (y, q, y, q))
            row = one.iloc[0]
            quarter_rows.append(row)
            ncu = int(row["num_cusip"]) if not pd.isna(row["num_cusip"]) else 0
            nci = int(row["num_cik"]) if not pd.isna(row["num_cik"]) else 0
            logger.info(
                f"  [{idx}/{n_q}] {y} Q{q} | num_cusip={ncu} num_cik={nci}"
            )

        df_stats = pd.DataFrame(quarter_rows).reset_index(drop=True)
        df_stats["year"] = pd.to_numeric(df_stats["year"], errors="coerce").astype(
            "int16"
        )
        df_stats["quarter"] = pd.to_numeric(df_stats["quarter"], errors="coerce").astype(
            "int8"
        )
        df_stats["num_cusip"] = pd.to_numeric(
            df_stats["num_cusip"], errors="coerce"
        ).astype("int64")
        df_stats["num_cik"] = pd.to_numeric(
            df_stats["num_cik"], errors="coerce"
        ).astype("int64")

        handler.insert_dataframe_regular(
            df_stats, TGT_CHANGED_STAS, if_exists="replace"
        )
        logger.info(f"Wrote {len(df_stats)} rows to {TARGET_DB}.{TGT_CHANGED_STAS}")

        _apply_tertiles(handler, logger)
        logger.info("Step 8 completed")

    except Exception as e:
        logger.error(f"Step 8 failed: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        if handler:
            handler.disconnect()
        logger.close()
