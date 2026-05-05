"""Step 6: Build changed_holdings with per-(cik, cusip, year, quarter) deltas.

Strategy: load two consecutive quarters into memory, loop per-CIK (aggregating any
duplicate (cik, cusip) rows before comparison), write the pair's results, then slide
the window forward — only the *next* quarter is fetched per iteration; the current
quarter stays in memory as the next pair's `prev`.

Columns & formulas (t = curr quarter, t-1 = prev quarter):
    change_in_shares          = shares_t - shares_{t-1}
    change_in_weight          = w_t      - w_{t-1}
    change_in_adjusted_weight = w_t * (AUM_t / AUM_{t-1}) - w_{t-1}

(year, quarter) = curr quarter, e.g. (2023, 2) indicates the Q1->Q2 2023 change.
"""

import pandas as pd
from ETL.logger.logger import ETLLogger
from ETL.gnn_db_pipeline.config import (
    TARGET_DB,
    TGT_NORMALIZED_HOLDINGS,
    TGT_CIK_AUM,
    TGT_CHANGED_HOLDINGS,
)
from ETL.gnn_db_pipeline.db_connector import ConfigurablePostgresHandler


_OUTPUT_COLS = [
    "cik", "cusip", "year", "quarter",
    "change_in_shares", "change_in_weight", "change_in_adjusted_weight",
]


def _load_quarter(handler, y, q):
    """Load raw (cik, cusip, shares, weight) rows for one quarter. No dedup here."""
    return handler.query(
        f"SELECT cik, cusip, shares, weight FROM {TGT_NORMALIZED_HOLDINGS} "
        f"WHERE year={y} AND quarter={q}"
    )


def _cik_delta(prev_rows, curr_rows, factor, y_curr, q_curr, cik):
    """Compute this CIK's deltas across prev/curr positions.

    `prev_rows` / `curr_rows` may be None (CIK absent that quarter) or have
    duplicate cusips; both cases are handled by aggregating cusip totals.
    """
    def _agg(df):
        if df is None or len(df) == 0:
            return None
        return df.groupby("cusip", as_index=False).agg(
            shares=("shares", "sum"), weight=("weight", "sum")
        )

    p = _agg(prev_rows)
    c = _agg(curr_rows)

    if p is not None and c is not None:
        merged = p.rename(columns={"shares": "s_prev", "weight": "w_prev"}).merge(
            c.rename(columns={"shares": "s_curr", "weight": "w_curr"}),
            on="cusip", how="outer",
        )
    elif p is not None:
        merged = p.rename(columns={"shares": "s_prev", "weight": "w_prev"}).assign(
            s_curr=0, w_curr=0.0
        )
    elif c is not None:
        merged = c.rename(columns={"shares": "s_curr", "weight": "w_curr"}).assign(
            s_prev=0, w_prev=0.0
        )
    else:
        return None

    merged = merged.fillna({"s_prev": 0, "s_curr": 0, "w_prev": 0.0, "w_curr": 0.0})
    merged["s_prev"] = merged["s_prev"].astype("int64")
    merged["s_curr"] = merged["s_curr"].astype("int64")

    merged["change_in_shares"] = merged["s_curr"] - merged["s_prev"]
    merged["change_in_weight"] = merged["w_curr"] - merged["w_prev"]
    merged["change_in_adjusted_weight"] = (
        merged["w_curr"] * factor - merged["w_prev"]
    )

    # Drop rows where every delta is effectively zero
    nonzero = (
        (merged["change_in_shares"].abs() > 0)
        | (merged["change_in_weight"].abs() > 1e-12)
        | (merged["change_in_adjusted_weight"].abs() > 1e-12)
    )
    merged = merged[nonzero]
    if merged.empty:
        return None

    merged["cik"] = cik
    merged["year"] = y_curr
    merged["quarter"] = q_curr
    return merged[_OUTPUT_COLS]


def _process_pair(df_prev, df_curr, aum_lookup, y_prev, q_prev, y_curr, q_curr, logger):
    """Iterate per-CIK, compute deltas, return one concatenated pandas DataFrame."""
    prev_groups = {k: v for k, v in df_prev.groupby("cik")} if len(df_prev) else {}
    curr_groups = {k: v for k, v in df_curr.groupby("cik")} if len(df_curr) else {}
    all_ciks = set(prev_groups) | set(curr_groups)

    results = []
    skipped_aum = 0
    for cik in all_ciks:
        aum_prev = aum_lookup.get((cik, y_prev, q_prev))
        aum_curr = aum_lookup.get((cik, y_curr, q_curr))
        if (
            aum_prev is None or aum_curr is None
            or aum_prev <= 0 or aum_curr <= 0
        ):
            skipped_aum += 1
            continue
        factor = aum_curr / aum_prev

        out = _cik_delta(
            prev_groups.get(cik), curr_groups.get(cik),
            factor, y_curr, q_curr, cik,
        )
        if out is not None:
            results.append(out)

    if skipped_aum:
        logger.info(f"    (skipped {skipped_aum} CIKs missing/zero AUM in one side)")

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.concat(results, ignore_index=True)


def run():
    logger = ETLLogger(name="Step6_ChangedHoldings", console_output=True)
    logger.info("=" * 80)
    logger.info("STEP 6: CHANGED_HOLDINGS (per-CIK loop, sliding 2-quarter window)")
    logger.info("=" * 80)

    handler = None
    try:
        handler = ConfigurablePostgresHandler(TARGET_DB)
        handler.connect()

        # Drop & recreate changed_holdings with the new schema
        handler.execute(f"DROP TABLE IF EXISTS {TGT_CHANGED_HOLDINGS}")
        handler.execute(f"""
            CREATE TABLE {TGT_CHANGED_HOLDINGS} (
                cik TEXT,
                cusip TEXT,
                year SMALLINT,
                quarter SMALLINT,
                change_in_shares BIGINT,
                change_in_weight DOUBLE PRECISION,
                change_in_adjusted_weight DOUBLE PRECISION
            )
        """)

        handler.execute(
            f"CREATE INDEX IF NOT EXISTS idx_nh_yq ON {TGT_NORMALIZED_HOLDINGS}(year, quarter)"
        )
        handler.execute(
            f"CREATE INDEX IF NOT EXISTS idx_aum_yq ON {TGT_CIK_AUM}(year, quarter)"
        )

        # Load all AUM once (small table; dict keyed by (cik, year, quarter))
        logger.info("Loading cik_aum lookup...")
        aum_df = handler.query(
            f"SELECT cik, year, quarter, total FROM {TGT_CIK_AUM}"
        )
        aum_lookup = {
            (row.cik, int(row.year), int(row.quarter)): float(row.total)
            for row in aum_df.itertuples(index=False)
        }
        logger.info(f"  Loaded {len(aum_lookup)} AUM entries")

        # Ordered list of quarters
        df_q = handler.query(
            f"SELECT DISTINCT year, quarter FROM {TGT_NORMALIZED_HOLDINGS} "
            f"ORDER BY year, quarter"
        )
        quarters = [(int(r["year"]), int(r["quarter"])) for _, r in df_q.iterrows()]
        logger.info(f"Found {len(quarters)} quarters; processing {len(quarters) - 1} pairs")

        # Prime the sliding window with the first quarter
        y0, q0 = quarters[0]
        logger.info(f"Loading {y0}_Q{q0} into memory...")
        df_prev = _load_quarter(handler, y0, q0)
        logger.info(f"  {len(df_prev)} rows")

        total = 0
        for i in range(1, len(quarters)):
            y_prev, q_prev = quarters[i - 1]
            y_curr, q_curr = quarters[i]
            label = f"{y_prev}_Q{q_prev} -> {y_curr}_Q{q_curr}"

            logger.info(f"Loading {y_curr}_Q{q_curr} into memory ({label})...")
            df_curr = _load_quarter(handler, y_curr, q_curr)

            result = _process_pair(
                df_prev, df_curr, aum_lookup,
                y_prev, q_prev, y_curr, q_curr, logger,
            )

            if not result.empty:
                result["change_in_shares"] = result["change_in_shares"].astype("int64")
                n = handler._copy_dataframe(TGT_CHANGED_HOLDINGS, result)
            else:
                n = 0
            total += n
            logger.info(f"  {label}: {n} rows")

            # Slide: curr becomes prev; free the old prev
            df_prev = df_curr

        logger.info(f"Total rows inserted: {total}")
        logger.info("Step 6 completed")

    except Exception as e:
        logger.error(f"Step 6 failed: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        if handler:
            handler.disconnect()
        logger.close()
