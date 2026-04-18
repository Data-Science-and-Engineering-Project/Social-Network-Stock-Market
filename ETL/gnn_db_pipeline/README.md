# GNN Database Pipeline

Transforms the `Social_13F` database into `13FGNN`, a clean, GNN-ready database of institutional portfolio holdings, AUM, quarterly returns, and quarter-over-quarter holding changes.

## Quick Start

```bash
# Full pipeline (all quarters)
./venv/Scripts/python.exe -m ETL.gnn_db_pipeline.run_pipeline

# Single-quarter test (re-runs steps 4+5 for that quarter only; step3, step6 still run end-to-end)
./venv/Scripts/python.exe -m ETL.gnn_db_pipeline.run_pipeline --test-quarter 2017_Q3

# Step 6 only, once steps 1–5 are already populated
./venv/Scripts/python.exe -c "from ETL.gnn_db_pipeline.steps import step6_changed_holdings; step6_changed_holdings.run()"

# Step 6 sanity test on a single pair (2018_Q2 -> 2018_Q3, writes to changed_holdings_test)
./venv/Scripts/python.exe -m ETL.gnn_db_pipeline.test_2018_q2_q3
```

## File Structure

```
ETL/gnn_db_pipeline/
├── config.py                      # Source/target DB + table-name constants
├── db_connector.py                # ConfigurablePostgresHandler (DOUBLE PRECISION for floats)
├── pipeline.py                    # GNNDBPipeline orchestrator
├── run_pipeline.py                # CLI entry point
├── test_2017_q3.py                # Correctness test for step4 weight sums
├── test_2018_q2_q3.py             # Correctness test for step6 per-CIK delta logic
└── steps/
    ├── step1_create_database.py       # CREATE DATABASE 13FGNN
    ├── step2_copy_tables.py           # Copy ticker_prices + ticker_to_cusip
    ├── step3_stocks_return.py         # Build stocks_return (price_t / price_{t-1})
    ├── step4_normalized_holdings.py   # Build normalized_holdings (weights + raw shares + price)
    ├── step5_cik_aum.py               # Build cik_aum (sum position values)
    └── step6_changed_holdings.py      # Build changed_holdings (3 delta columns)
```

## Pipeline Stages

| # | Step | Per-quarter? | Output table | Purpose |
|---|---|---|---|---|
| 1 | `step1_create_database` | no | — | Create `13FGNN` if it doesn't exist |
| 2 | `step2_copy_tables` | no | `ticker_prices`, `ticker_to_cusip` | Copy reference tables from `Social_13F` |
| 3 | `step3_stocks_return` | no | `stocks_return` | `price_t / price_{t-1}` per `(cusip, year, quarter)` |
| 4 | `step4_normalized_holdings` | yes | `normalized_holdings` | Per-position weights + raw shares/price |
| 5 | `step5_cik_aum` | yes | `cik_aum` | Per-fund total portfolio value per quarter |
| 6 | `step6_changed_holdings` | no | `changed_holdings` | Three per-position deltas between consecutive quarters |

Per-quarter steps run once per `(year, quarter)` tuple; `pipeline.py` discovers them from the source `holdings_filtered_new` table. The non-per-quarter steps run exactly once.

Orchestration order (see `pipeline.py::GNNDBPipeline.run`):

```
step1  ->  step2  ->  step3
                        ↓
            drop normalized_holdings + cik_aum   (so new schemas apply)
                        ↓
            for each (year, quarter):
                step4  ->  step5
                        ↓
                      step6
```

## Table Schemas

### `stocks_return`
| Column | Type | Notes |
|---|---|---|
| cusip | TEXT | |
| year | SMALLINT | curr quarter |
| quarter | SMALLINT | curr quarter |
| return | DOUBLE PRECISION | `price_t / price_{t-1}` |

One row per `(cusip, year, quarter)` for every quarter *except* the first one per cusip (no prior price to compare to).

### `normalized_holdings`
| Column | Type | Notes |
|---|---|---|
| cik | TEXT | fund |
| cusip | TEXT | security |
| year | SMALLINT | |
| quarter | SMALLINT | |
| shares | BIGINT | `sshprnamt` |
| price | DOUBLE PRECISION | quarterly price from `ticker_prices` |
| weight | DOUBLE PRECISION | `position_value / portfolio_total`, renormalized so each portfolio sums to 1.0 |

### `cik_aum`
| Column | Type | Notes |
|---|---|---|
| cik | TEXT | |
| year | SMALLINT | |
| quarter | SMALLINT | |
| total | DOUBLE PRECISION | `SUM(shares × price)` across that fund's portfolio for that quarter |

### `changed_holdings`
| Column | Type | Formula |
|---|---|---|
| cik | TEXT | |
| cusip | TEXT | |
| year | SMALLINT | **curr** quarter; e.g. `(2023, 2)` = Q1→Q2 of 2023 |
| quarter | SMALLINT | |
| change_in_shares | BIGINT | `shares_t − shares_{t-1}` |
| change_in_weight | DOUBLE PRECISION | `w_t − w_{t-1}` |
| change_in_adjusted_weight | DOUBLE PRECISION | `w_t × (AUM_t / AUM_{t-1}) − w_{t-1}` |

## Step 6 — Per-CIK Loop with Sliding 2-Quarter Window

`step6_changed_holdings.run()` cannot be written as a single vectorized join on the full `normalized_holdings` table: if `(cik, cusip, year, quarter)` has even occasional duplicate rows (e.g., from overlapping `ticker_to_cusip` windows or amended 13F filings), a full-outer join on `(cik, cusip)` between two quarters is cartesian and row counts explode (observed: 8× on the `(0000049205, 001055102, 2013_Q3)` example).

The per-CIK loop sidesteps this:

1. Load `cik_aum` once into a `(cik, year, quarter) -> total` dict.
2. Load **only the first quarter** of `normalized_holdings` into pandas as `df_prev`.
3. For each subsequent quarter:
   - Load `df_curr` (one SQL query per pair; `df_prev` reused from the previous iteration).
   - For each CIK appearing in either frame:
     - Skip if AUM is missing or zero in either quarter.
     - `df.groupby("cusip").sum()` on each side → dedups any `(cik, cusip)` duplicates by summing shares and weights.
     - Outer-merge the two per-CIK frames on `cusip`, fill missing side with 0.
     - Compute the three deltas; drop rows where all three are ~0.
   - `COPY` the pair's results into `changed_holdings`.
   - Slide: `df_prev = df_curr` (no re-query).

Memory footprint: ≤ 2 quarters (~1.25M rows each) + the AUM dict at any time.

## Why We Override `_get_sql_type` for Floats

The base `PostgresHandler._get_sql_type` maps pandas `float*` → `NUMERIC(18, 4)`, which truncates weights below ~5e-5 to zero. For a fund with thousands of small positions, that breaks the `SUM(weight) ≈ 1.0` invariant. `ConfigurablePostgresHandler._get_sql_type` overrides the mapping to `DOUBLE PRECISION`.

## Verification Queries

```sql
\c "13FGNN"

-- Per-portfolio weight sums ≈ 1.0
SELECT cik, year, quarter, COUNT(*) AS n_positions, SUM(weight) AS w_sum
FROM normalized_holdings
WHERE year = 2018 AND quarter = 3
GROUP BY cik, year, quarter
ORDER BY ABS(SUM(weight) - 1.0) DESC
LIMIT 5;

-- No duplicate (cik, cusip, year, quarter) rows in normalized_holdings
SELECT cik, cusip, year, quarter, COUNT(*) AS n
FROM normalized_holdings
GROUP BY cik, cusip, year, quarter
HAVING COUNT(*) > 1
LIMIT 10;

-- changed_holdings row counts per pair
SELECT year, quarter, COUNT(*) AS n_rows
FROM changed_holdings
GROUP BY year, quarter
ORDER BY year, quarter;

-- Bound-check: weight deltas should lie in [-1, 1]
SELECT
    MIN(change_in_weight) AS w_min, MAX(change_in_weight) AS w_max,
    MIN(change_in_adjusted_weight) AS adj_min, MAX(change_in_adjusted_weight) AS adj_max,
    COUNT(*) FILTER (WHERE ABS(change_in_weight) > 1) AS w_out_of_bounds
FROM changed_holdings;

-- stocks_return sanity
SELECT cusip, COUNT(*) AS n_quarters,
       MIN(return) AS r_min, MAX(return) AS r_max
FROM stocks_return
GROUP BY cusip
ORDER BY n_quarters DESC
LIMIT 5;
```

## Notes

- **Date-range join** in step4/step5: `period_start ∈ [trading_start_date, trading_end_date]` handles CUSIP→ticker re-mapping over time.
- **Holdings missing a price or ticker mapping are dropped** (inner joins). This is intentional — weights would be ill-defined otherwise.
- **Weight renormalization**: step4 renormalizes weights per portfolio (`weight /= SUM(weight)`) after computing them, so each portfolio's weights sum to exactly 1.0 (within floating-point).
- **First quarter per CIK has no `changed_holdings` row** (no `t-1` to compare against). Same for `stocks_return`.
