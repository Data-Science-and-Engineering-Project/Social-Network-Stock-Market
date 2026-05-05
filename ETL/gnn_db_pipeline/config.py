"""Configuration constants for GNN Database Pipeline."""

# Database names
SOURCE_DB = "Social_13F"
TARGET_DB = "13FGNN"

# Source table names (read from Social_13F)
SRC_HOLDINGS_FILTERED = "holdings_filtered_new"
SRC_TICKER_PRICES = "ticker_prices"
SRC_TICKER_TO_CUSIP = "ticker_to_cusip"

# Target table names (write to 13FGNN)
TGT_TICKER_PRICES = "ticker_prices"
TGT_TICKER_TO_CUSIP = "ticker_to_cusip"
TGT_STOCKS_RETURN = "stocks_return"
TGT_NORMALIZED_HOLDINGS = "normalized_holdings"
TGT_CIK_AUM = "cik_aum"
TGT_CHANGED_HOLDINGS = "changed_holdings"
TGT_CHANGED_STAS = "changed_stas"
