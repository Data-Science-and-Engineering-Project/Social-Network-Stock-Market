import pandas as pd
import os

# =========================
# Load parquet parts
# =========================
part_files = [
    "files/holdings_filtered_part001.parquet",
    "files/holdings_filtered_part002.parquet",
    "files/holdings_filtered_part003.parquet",
    "files/holdings_filtered_part004.parquet"
]

holdings_parts = []
for p in part_files:
    if os.path.exists(p):
        df_part = pd.read_parquet(p)
        holdings_parts.append(df_part)
        print(f"✅ Loaded {p}: {len(df_part):,} rows")
    else:
        print(f"⚠️ Missing {p}, skipped")

holdings = pd.concat(holdings_parts, ignore_index=True)
print(f"\n🎯 Total holdings rows: {len(holdings):,}")

# =========================
# Filter to 2018
# =========================
holdings = holdings[holdings["year"] == 2018]

# =========================
# Load lookup tables
# =========================
ticker_map = pd.read_parquet("../files/icker_to_cusip.parquet")
prices = pd.read_parquet("../files/ickerprices.parquet")

# =========================
# Normalize lookup tables
# =========================
ticker_map["cusip"] = ticker_map["cusip"].astype(str)
prices["period_start"] = pd.to_datetime(prices["period_start"])

# =========================
# Output directory
# =========================
output_dir = "../generated_csv_2018"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Generate quarterly CSVs
# =========================
for q in [1, 2, 3, 4]:

    df = holdings[holdings["quarter"] == q].copy()

    # Rename to DS-required format
    df = df.rename(columns={
        "cik": "CIK",
        "cusip": "CUSIP",
        "sshprnamt": "SSHPRNAMT",
        "period_start": "PERIOD_DATE"
    })

    # Type normalization
    df["CIK"] = df["CIK"].astype(str)
    df["CUSIP"] = df["CUSIP"].astype(str)
    df["PERIOD_DATE"] = pd.to_datetime(df["PERIOD_DATE"])

    # =========================
    # Join: CUSIP → ticker
    # =========================
    df = df.merge(
        ticker_map[["cusip", "ticker"]],
        left_on="CUSIP",
        right_on="cusip",
        how="left"
    ).drop(columns=["cusip"])

    # =========================
    # Join: ticker + period → price
    # =========================
    df = df.merge(
        prices[["ticker", "period_start", "price"]],
        left_on=["ticker", "PERIOD_DATE"],
        right_on=["ticker", "period_start"],
        how="left"
    ).drop(columns=["period_start"])

    # =========================
    # Compute VALUE
    # =========================
    df["VALUE"] = df["SSHPRNAMT"] * df["price"]

    # =========================
    # Final column order
    # =========================
    df = df[["CIK", "CUSIP", "VALUE", "SSHPRNAMT", "PERIOD_DATE"]]

    # =========================
    # Save CSV
    # =========================
    out_path = os.path.join(
        output_dir,
        f"short_Infotable_Q{q}_2018_A.csv"
    )
    df.to_csv(out_path, index=False)

    print(f"✅ Created {out_path} ({len(df):,} rows)")

print("\n🎉 All quarterly CSV files generated successfully.")
