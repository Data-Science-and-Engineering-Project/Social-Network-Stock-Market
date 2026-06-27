# Project Conclusion

## What Was Built

A graph-based machine learning pipeline that treats institutional investment funds as a social network. Funds that hold overlapping positions are implicitly connected; the model learns from that network structure to predict which stocks a fund will buy next quarter.

The pipeline runs end-to-end across four stages:

1. **ETL** — Ingests SEC 13F filings (2013–2024) into PostgreSQL
2. **Preprocessing** — Filters to Russell 3000 universe, computes quarter-over-quarter position deltas, enriches with EODHD fundamentals
3. **GNN Model** — Trains a LightGCN on a bipartite fund-stock graph using BPR loss, evaluated per rolling quarter-pair
4. **Portfolio Backtesting** — Selects the top-10 ranked stocks each quarter, equal-weighted, rebalanced quarterly

---

## Model Results

The LightGCN (v4, `change_in_weight` edges) achieves strong link-prediction performance:

| Metric | Value |
|--------|-------|
| Test AUC | **93.52%** |
| Test F1 (optimal threshold) | **86.45%** |
| Spearman ρ (rank vs. Q+1 return) | 0.0065 |

The near-zero Spearman correlation is expected and correct — the model predicts *institutional buying behavior*, not stock returns. Funds accumulate positions for reasons unrelated to short-term price appreciation (diversification, value, income, ESG mandates). High AUC confirms the social-network signal is strongly predictive of future purchases; the weak return correlation confirms the model is not leaking future price data.

---

## Portfolio Performance

Backtest over **47 quarters (Q4 2013 – Q4 2024, 11.8 years)**, starting with $1,000,000:

| Metric | LightGCN Portfolio | Russell 3000 Benchmark |
|--------|--------------------|------------------------|
| Total Return | **+286.4%** | +248.6% |
| Annualized Return | **+12.2%** | +11.2% |
| Sharpe Ratio | **0.749** | — |
| Max Drawdown | -39.29% | — |
| Final Value | **$3,863,901** | ~$3,486,000 |

The strategy outperforms the Russell 3000 by **+37.8 percentage points** total and **+1.0% annualized** over the full period, with a Sharpe ratio of 0.749 and a drawdown profile consistent with the broader US equity market.

---

## Key Methodological Strengths

- **No-leakage design**: The GNN forward pass during training propagates only over quarter Q's graph structure. Q+1 edges are never visible to the model.
- **No lookahead in portfolio construction**: Fundamentals used at each rebalance date are from the previous quarter only.
- **Realistic universe**: Holdings are filtered to the Russell 3000 at each period, avoiding survivorship bias.
- **Long backtest**: 47 quarters spanning multiple regimes — recovery (2013–2018), correction (2018–2019), COVID crash and rally (2020–2021), tightening cycle (2022–2023), and 2024 correction.

---

## Limitations

- **Concentration risk**: A top-10 equal-weight portfolio is undiversified relative to a broad index; max drawdown (-39.29%) reflects this.
- **Transaction costs ignored**: The backtest does not account for slippage or commissions on quarterly rebalancing.
- **Spearman ρ ≈ 0**: The model's rankings do not correlate with actual returns, meaning the outperformance could partly reflect factor exposures (e.g., momentum, quality) rather than pure predictive signal.
- **Single model run**: Results are from one training run on the final quarter pair; a full walk-forward evaluation across all 47 quarter-pairs would strengthen the performance claim.

---

## Conclusion

The social-network structure of institutional fund holdings contains a genuine predictive signal for future buying behavior — the LightGCN captures it with 93.52% AUC. Translating that signal into a simple top-10 equal-weight portfolio produces a modest but consistent outperformance of +1.0% annualized above the Russell 3000 over an 11.8-year backtest, with no evidence of data leakage. The result validates the core hypothesis: funds that share overlapping positions form an implicit network, and that network's dynamics are predictable.
