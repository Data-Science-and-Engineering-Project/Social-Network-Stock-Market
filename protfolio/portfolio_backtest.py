"""
Portfolio Backtest using LightGCN model rankings.

Rules:
- Rebalance only on the first day of each quarter
- Select top 10 stocks by model mean_score (equal weighted)
- Use only data available BEFORE the quarter starts (no lookahead)
- The model's predicts_year/predicts_quarter tells us which quarter
  the model is forecasting — we invest in those stocks for that quarter.
"""

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RANKS_FILE = DATA_DIR / "cusip_ranks_v4__change_in_weight.parquet"
RETURNS_FILE = DATA_DIR / "stocks_return.parquet"
FINANCIAL_FILE = DATA_DIR / "cusip_financial_data.parquet"

OUTPUT_HTML = Path(__file__).parent / "portfolio_report.html"

TOP_N = 10
INITIAL_VALUE = 1_000_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quarter_label(year: int, quarter: int) -> str:
    return f"{int(year)} Q{int(quarter)}"


def prev_quarter(year: int, quarter: int):
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def period_index(year: int, quarter: int) -> int:
    return int(year) * 4 + int(quarter) - 1


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    ranks = pd.read_parquet(RANKS_FILE)
    returns = pd.read_parquet(RETURNS_FILE)
    financial = pd.read_parquet(FINANCIAL_FILE)
    return ranks, returns, financial


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------
def run_backtest(ranks: pd.DataFrame, returns: pd.DataFrame, financial: pd.DataFrame):
    # All unique prediction quarters, sorted chronologically
    quarters = (
        ranks[["predicts_year", "predicts_quarter"]]
        .drop_duplicates()
        .copy()
    )
    quarters["period"] = quarters.apply(
        lambda r: period_index(r.predicts_year, r.predicts_quarter), axis=1
    )
    quarters = quarters.sort_values("period").reset_index(drop=True)

    # Index returns for fast lookup: (year, quarter, cusip) -> return value
    returns_idx = returns.set_index(["year", "quarter", "cusip"])["return"]

    # Index financial for fast lookup: (year, quarter, cusip) -> row
    financial_idx = financial.set_index(["year", "quarter", "cusip"])

    portfolio_history = []
    cumulative_value = INITIAL_VALUE
    peak_value = INITIAL_VALUE

    prev_holdings = set()

    for _, row in quarters.iterrows():
        py, pq = int(row.predicts_year), int(row.predicts_quarter)
        label = quarter_label(py, pq)

        # --- Select top-N stocks for this quarter ---
        q_ranks = ranks[
            (ranks.predicts_year == py) & (ranks.predicts_quarter == pq)
        ].nsmallest(TOP_N, "rank")[["cusip", "mean_score", "rank"]]

        top_cusips = q_ranks["cusip"].tolist()

        # --- Get actual returns for these stocks during this quarter ---
        stock_returns = {}
        for cusip in top_cusips:
            try:
                r = returns_idx.loc[(py, pq, cusip)]
                if pd.notna(r) and r > 0:
                    stock_returns[cusip] = r
            except KeyError:
                pass

        # Equal-weight among stocks that have return data
        available = list(stock_returns.keys())
        if available:
            port_return = sum(stock_returns[c] for c in available) / len(available)
        else:
            port_return = 1.0  # flat quarter, no data

        # --- Benchmark: equal-weight ALL ranked stocks that quarter ---
        all_cusips = ranks[
            (ranks.predicts_year == py) & (ranks.predicts_quarter == pq)
        ]["cusip"].tolist()
        bench_rets = []
        for cusip in all_cusips:
            try:
                r = returns_idx.loc[(py, pq, cusip)]
                if pd.notna(r) and r > 0:
                    bench_rets.append(r)
            except KeyError:
                pass
        bench_return = sum(bench_rets) / len(bench_rets) if bench_rets else 1.0

        # --- Update portfolio value ---
        prev_value = cumulative_value
        cumulative_value *= port_return
        peak_value = max(peak_value, cumulative_value)
        drawdown = (cumulative_value - peak_value) / peak_value

        # --- Financial data for top-N (use PREVIOUS quarter as that's available at decision time) ---
        prev_y, prev_q = prev_quarter(py, pq)
        holdings_detail = []
        for _, rr in q_ranks.iterrows():
            cusip = rr["cusip"]
            actual_ret = stock_returns.get(cusip, None)
            fin_row = {}
            try:
                fin_data = financial_idx.loc[(prev_y, prev_q, cusip)]
                fin_row = {
                    "pe_ratio": round(float(fin_data.get("pe_ratio", float("nan"))), 2),
                    "roe": round(float(fin_data.get("roe", float("nan"))), 4),
                    "ev_ebitda": round(float(fin_data.get("ev_ebitda", float("nan"))), 2),
                    "price_to_book": round(float(fin_data.get("price_to_book", float("nan"))), 2),
                    "debt_to_equity": round(float(fin_data.get("debt_to_equity", float("nan"))), 2),
                    "dividend_yield": round(float(fin_data.get("dividend_yield", float("nan"))), 4),
                }
            except KeyError:
                pass

            holdings_detail.append({
                "cusip": cusip,
                "rank": int(rr["rank"]),
                "mean_score": round(float(rr["mean_score"]), 4),
                "actual_return_pct": round((actual_ret - 1) * 100, 2) if actual_ret else None,
                **fin_row,
            })

        # --- Turnover ---
        current_set = set(top_cusips)
        entered = current_set - prev_holdings
        exited = prev_holdings - current_set
        prev_holdings = current_set

        portfolio_history.append({
            "label": label,
            "year": py,
            "quarter": pq,
            "period": int(row.period),
            "portfolio_return": round((port_return - 1) * 100, 4),
            "benchmark_return": round((bench_return - 1) * 100, 4),
            "portfolio_value": round(cumulative_value, 2),
            "drawdown": round(drawdown * 100, 4),
            "holdings": holdings_detail,
            "entered": sorted(entered),
            "exited": sorted(exited),
            "n_available": len(available),
        })

    return portfolio_history


# ---------------------------------------------------------------------------
# Compute summary metrics
# ---------------------------------------------------------------------------
def compute_metrics(history):
    port_rets = [h["portfolio_return"] / 100 + 1 for h in history]
    bench_rets = [h["benchmark_return"] / 100 + 1 for h in history]

    total_port = (port_rets[-1] if port_rets else 1.0)
    for r in port_rets[:-1]:
        total_port = None  # will compute properly below
    total_port = 1.0
    for r in port_rets:
        total_port *= r
    total_bench = 1.0
    for r in bench_rets:
        total_bench *= r

    n = len(port_rets)
    years = n / 4.0

    # Annualized return
    ann_port = (total_port ** (1 / years) - 1) * 100 if years > 0 else 0
    ann_bench = (total_bench ** (1 / years) - 1) * 100 if years > 0 else 0

    # Quarterly excess returns
    excess = [(p - 1) - (b - 1) for p, b in zip(port_rets, bench_rets)]
    mean_excess = sum(excess) / n if n > 0 else 0
    std_excess = math.sqrt(sum((x - mean_excess) ** 2 for x in excess) / n) if n > 1 else 0
    sharpe = (mean_excess / std_excess * math.sqrt(4)) if std_excess > 0 else 0

    max_dd = min(h["drawdown"] for h in history)

    return {
        "total_return_port": round((total_port - 1) * 100, 2),
        "total_return_bench": round((total_bench - 1) * 100, 2),
        "ann_return_port": round(ann_port, 2),
        "ann_return_bench": round(ann_bench, 2),
        "n_quarters": n,
        "years": round(years, 1),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------
def make_html(history, metrics):
    labels = [h["label"] for h in history]
    port_values = [h["portfolio_value"] for h in history]
    port_rets = [h["portfolio_return"] for h in history]
    bench_rets = [h["benchmark_return"] for h in history]
    drawdowns = [h["drawdown"] for h in history]

    # Cumulative benchmark value
    bench_value = [INITIAL_VALUE]
    for h in history:
        bench_value.append(bench_value[-1] * (1 + h["benchmark_return"] / 100))
    bench_value = bench_value[1:]

    # Most frequent stocks
    from collections import Counter
    all_cusips = []
    for h in history:
        all_cusips.extend([s["cusip"] for s in h["holdings"]])
    top_frequent = Counter(all_cusips).most_common(15)

    # Holdings tables HTML
    holdings_html = ""
    for h in history:
        entered_str = ", ".join(h["entered"]) if h["entered"] else "—"
        exited_str = ", ".join(h["exited"]) if h["exited"] else "—"
        port_ret_color = "#16a34a" if h["portfolio_return"] >= 0 else "#dc2626"
        bench_ret_color = "#16a34a" if h["benchmark_return"] >= 0 else "#dc2626"

        rows = ""
        for s in h["holdings"]:
            ret_val = s.get("actual_return_pct")
            ret_str = f"{ret_val:+.2f}%" if ret_val is not None else "N/A"
            ret_color = "#16a34a" if (ret_val or 0) >= 0 else "#dc2626"

            def fmt(v, decimals=2):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "N/A"
                return f"{v:.{decimals}f}"

            rows += f"""
            <tr>
              <td>{s['cusip']}</td>
              <td>{s['rank']}</td>
              <td>{s['mean_score']:.4f}</td>
              <td style="color:{ret_color};font-weight:600">{ret_str}</td>
              <td>{fmt(s.get('pe_ratio'))}</td>
              <td>{fmt(s.get('roe'), 3)}</td>
              <td>{fmt(s.get('ev_ebitda'))}</td>
              <td>{fmt(s.get('price_to_book'))}</td>
              <td>{fmt(s.get('debt_to_equity'))}</td>
              <td>{fmt(s.get('dividend_yield'), 4)}</td>
            </tr>"""

        holdings_html += f"""
        <div class="quarter-block" id="q-{h['year']}-{h['quarter']}">
          <div class="quarter-header" onclick="toggleQuarter('{h['year']}-{h['quarter']}')">
            <span class="q-label">{h['label']}</span>
            <span class="q-metrics">
              Portfolio: <span style="color:{port_ret_color};font-weight:700">{h['portfolio_return']:+.2f}%</span>
              &nbsp;|&nbsp; Benchmark: <span style="color:{bench_ret_color};font-weight:700">{h['benchmark_return']:+.2f}%</span>
              &nbsp;|&nbsp; Value: ${h['portfolio_value']:,.0f}
            </span>
            <span class="q-toggle">▼</span>
          </div>
          <div class="quarter-content" id="qc-{h['year']}-{h['quarter']}">
            <div class="turnover-row">
              <span class="pill entered">▲ Entered: {entered_str}</span>
              <span class="pill exited">▼ Exited: {exited_str}</span>
              <span class="pill info">Stocks with return data: {h['n_available']}/{TOP_N}</span>
            </div>
            <table class="holdings-table">
              <thead>
                <tr>
                  <th>CUSIP</th><th>Rank</th><th>Model Score</th><th>Actual Return</th>
                  <th>P/E</th><th>ROE</th><th>EV/EBITDA</th><th>P/B</th><th>D/E</th><th>Div Yield</th>
                </tr>
              </thead>
              <tbody>{rows}
              </tbody>
            </table>
          </div>
        </div>"""

    freq_rows = ""
    for cusip, count in top_frequent:
        pct = count / len(history) * 100
        freq_rows += f"<tr><td>{cusip}</td><td>{count}</td><td>{pct:.0f}%</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LightGCN Portfolio Backtest Report</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
    }}
    .header {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border-bottom: 1px solid #334155;
      padding: 2rem 2rem 1.5rem;
    }}
    .header h1 {{ font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }}
    .header p {{ color: #94a3b8; margin-top: 0.4rem; font-size: 0.9rem; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .metric-card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1.2rem;
    }}
    .metric-card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 0.3rem; }}
    .metric-card .sub {{ font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }}
    .positive {{ color: #4ade80; }}
    .negative {{ color: #f87171; }}
    .neutral {{ color: #60a5fa; }}
    .chart-card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .chart-card h2 {{ font-size: 1rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem; }}
    .section-title {{
      font-size: 1.1rem;
      font-weight: 600;
      color: #94a3b8;
      margin: 2rem 0 1rem;
      border-bottom: 1px solid #334155;
      padding-bottom: 0.5rem;
    }}
    .quarter-block {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      margin-bottom: 0.75rem;
      overflow: hidden;
    }}
    .quarter-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem 1.25rem;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s;
    }}
    .quarter-header:hover {{ background: #263245; }}
    .q-label {{ font-weight: 700; font-size: 1rem; color: #f1f5f9; min-width: 100px; }}
    .q-metrics {{ font-size: 0.85rem; color: #94a3b8; flex: 1; margin: 0 1rem; }}
    .q-toggle {{ color: #64748b; transition: transform 0.2s; }}
    .quarter-content {{ display: none; padding: 0 1.25rem 1.25rem; }}
    .quarter-content.open {{ display: block; }}
    .turnover-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }}
    .pill {{
      font-size: 0.75rem;
      padding: 0.3rem 0.75rem;
      border-radius: 999px;
      font-weight: 500;
    }}
    .pill.entered {{ background: #14532d33; color: #4ade80; border: 1px solid #16a34a55; }}
    .pill.exited {{ background: #7f1d1d33; color: #f87171; border: 1px solid #dc262655; }}
    .pill.info {{ background: #1e3a5f33; color: #60a5fa; border: 1px solid #2563eb55; }}
    .holdings-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }}
    .holdings-table th {{
      text-align: left;
      padding: 0.5rem 0.75rem;
      color: #64748b;
      font-weight: 600;
      border-bottom: 1px solid #334155;
      white-space: nowrap;
    }}
    .holdings-table td {{
      padding: 0.45rem 0.75rem;
      border-bottom: 1px solid #1e293b;
      color: #cbd5e1;
    }}
    .holdings-table tbody tr:hover {{ background: #263245; }}
    .freq-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      max-width: 480px;
    }}
    .freq-table th {{
      text-align: left;
      padding: 0.5rem 1rem;
      color: #64748b;
      font-weight: 600;
      border-bottom: 1px solid #334155;
    }}
    .freq-table td {{
      padding: 0.45rem 1rem;
      border-bottom: 1px solid #263245;
      color: #cbd5e1;
    }}
    .freq-table tbody tr:hover {{ background: #1e293b; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="header">
    <h1>LightGCN Portfolio Backtest</h1>
    <p>Top-{TOP_N} equal-weighted stocks per quarter &nbsp;|&nbsp; Rebalanced quarterly &nbsp;|&nbsp; No lookahead bias &nbsp;|&nbsp; {metrics['n_quarters']} quarters ({metrics['years']} years)</p>
  </div>
  <div class="container">

    <!-- Summary metrics -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">Total Return (Portfolio)</div>
        <div class="value {'positive' if metrics['total_return_port'] >= 0 else 'negative'}">{metrics['total_return_port']:+.1f}%</div>
        <div class="sub">from ${INITIAL_VALUE/1e6:.0f}M initial</div>
      </div>
      <div class="metric-card">
        <div class="label">Total Return (Benchmark)</div>
        <div class="value {'positive' if metrics['total_return_bench'] >= 0 else 'negative'}">{metrics['total_return_bench']:+.1f}%</div>
        <div class="sub">equal-weight all ranked stocks</div>
      </div>
      <div class="metric-card">
        <div class="label">Annualized Return</div>
        <div class="value neutral">{metrics['ann_return_port']:+.1f}%</div>
        <div class="sub">Benchmark: {metrics['ann_return_bench']:+.1f}%</div>
      </div>
      <div class="metric-card">
        <div class="label">Sharpe Ratio</div>
        <div class="value neutral">{metrics['sharpe']:.3f}</div>
        <div class="sub">annualized vs benchmark</div>
      </div>
      <div class="metric-card">
        <div class="label">Max Drawdown</div>
        <div class="value negative">{metrics['max_drawdown']:.2f}%</div>
        <div class="sub">from peak</div>
      </div>
      <div class="metric-card">
        <div class="label">Final Portfolio Value</div>
        <div class="value neutral">${port_values[-1]:,.0f}</div>
        <div class="sub">started at ${INITIAL_VALUE:,.0f}</div>
      </div>
    </div>

    <!-- Cumulative return chart -->
    <div class="chart-card">
      <h2>Cumulative Portfolio Value</h2>
      <div id="chart-cumulative" style="height:380px"></div>
    </div>

    <!-- Quarterly returns & drawdown side by side -->
    <div class="two-col">
      <div class="chart-card">
        <h2>Quarterly Returns (%)</h2>
        <div id="chart-quarterly" style="height:280px"></div>
      </div>
      <div class="chart-card">
        <h2>Drawdown (%)</h2>
        <div id="chart-drawdown" style="height:280px"></div>
      </div>
    </div>

    <!-- Most frequent stocks -->
    <div class="two-col">
      <div class="chart-card">
        <h2>Most Frequent Portfolio Holdings</h2>
        <table class="freq-table">
          <thead><tr><th>CUSIP</th><th>Quarters in Portfolio</th><th>Frequency</th></tr></thead>
          <tbody>{freq_rows}</tbody>
        </table>
      </div>
      <div class="chart-card">
        <h2>Average Quarterly Turnover</h2>
        <div id="chart-turnover" style="height:230px"></div>
      </div>
    </div>

    <!-- Holdings per quarter -->
    <div class="section-title">Portfolio Holdings by Quarter (click to expand)</div>
    {holdings_html}

  </div>

  <script>
    const LABELS = {json.dumps(labels)};
    const PORT_VAL = {json.dumps(port_values)};
    const BENCH_VAL = {json.dumps(bench_value)};
    const PORT_RETS = {json.dumps(port_rets)};
    const BENCH_RETS = {json.dumps(bench_rets)};
    const DRAWDOWNS = {json.dumps(drawdowns)};
    const ENTERED = {json.dumps([len(h['entered']) for h in history])};
    const EXITED = {json.dumps([len(h['exited']) for h in history])};

    const DARK_BG = '#1e293b';
    const GRID_COLOR = '#334155';
    const FONT_COLOR = '#94a3b8';

    const layoutBase = {{
      paper_bgcolor: DARK_BG,
      plot_bgcolor: DARK_BG,
      font: {{ color: FONT_COLOR, size: 11 }},
      xaxis: {{ gridcolor: GRID_COLOR, tickangle: -45 }},
      yaxis: {{ gridcolor: GRID_COLOR }},
      margin: {{ t: 20, b: 80, l: 60, r: 20 }},
      legend: {{ bgcolor: 'transparent' }},
    }};

    // Cumulative chart
    Plotly.newPlot('chart-cumulative', [
      {{
        x: LABELS, y: PORT_VAL, mode: 'lines', name: 'Portfolio (Top 10)',
        line: {{ color: '#60a5fa', width: 2.5 }},
        fill: 'tonexty', fillcolor: 'rgba(96,165,250,0.05)'
      }},
      {{
        x: LABELS, y: BENCH_VAL, mode: 'lines', name: 'Benchmark (Equal Weight All)',
        line: {{ color: '#f59e0b', width: 2, dash: 'dot' }}
      }}
    ], {{...layoutBase, yaxis: {{ ...layoutBase.yaxis, tickprefix: '$', tickformat: ',.0f' }}}}, {{responsive: true}});

    // Quarterly returns bar
    const retColors = PORT_RETS.map(r => r >= 0 ? 'rgba(74,222,128,0.8)' : 'rgba(248,113,113,0.8)');
    Plotly.newPlot('chart-quarterly', [
      {{
        x: LABELS, y: PORT_RETS, type: 'bar', name: 'Portfolio',
        marker: {{ color: retColors }},
      }},
      {{
        x: LABELS, y: BENCH_RETS, mode: 'lines', name: 'Benchmark',
        line: {{ color: '#f59e0b', width: 1.5, dash: 'dot' }}
      }}
    ], {{...layoutBase, yaxis: {{ ...layoutBase.yaxis, ticksuffix: '%' }}}}, {{responsive: true}});

    // Drawdown
    Plotly.newPlot('chart-drawdown', [{{
      x: LABELS, y: DRAWDOWNS, mode: 'lines',
      fill: 'tozeroy', fillcolor: 'rgba(248,113,113,0.2)',
      line: {{ color: '#f87171', width: 2 }},
      name: 'Drawdown'
    }}], {{...layoutBase, yaxis: {{ ...layoutBase.yaxis, ticksuffix: '%' }}}}, {{responsive: true}});

    // Turnover
    Plotly.newPlot('chart-turnover', [
      {{
        x: LABELS, y: ENTERED, type: 'bar', name: 'Stocks Entered',
        marker: {{ color: 'rgba(74,222,128,0.7)' }}
      }},
      {{
        x: LABELS, y: EXITED, type: 'bar', name: 'Stocks Exited',
        marker: {{ color: 'rgba(248,113,113,0.7)' }}
      }}
    ], {{...layoutBase, barmode: 'group', yaxis: {{ ...layoutBase.yaxis, dtick: 1 }}}}, {{responsive: true}});

    // Accordion toggle
    function toggleQuarter(id) {{
      const el = document.getElementById('qc-' + id);
      const parent = document.getElementById('q-' + id);
      const toggle = parent.querySelector('.q-toggle');
      el.classList.toggle('open');
      toggle.style.transform = el.classList.contains('open') ? 'rotate(180deg)' : '';
    }}

    // Open the first quarter by default
    if (LABELS.length > 0) {{
      const firstId = LABELS[0].replace(' ', '-').replace('Q', 'Q').replace(' Q', '-');
      // parse first label like "2013 Q4" -> "2013-4"
      const parts = LABELS[0].match(/(\\d+) Q(\\d+)/);
      if (parts) toggleQuarter(parts[1] + '-' + parts[2]);
    }}
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    ranks, returns, financial = load_data()
    print(f"  Ranks: {len(ranks):,} rows | Returns: {len(returns):,} rows | Financial: {len(financial):,} rows")

    print("Running backtest...")
    history = run_backtest(ranks, returns, financial)
    print(f"  Processed {len(history)} quarters")

    metrics = compute_metrics(history)
    print(f"\n  Portfolio total return : {metrics['total_return_port']:+.2f}%")
    print(f"  Benchmark total return : {metrics['total_return_bench']:+.2f}%")
    print(f"  Annualized return      : {metrics['ann_return_port']:+.2f}%")
    print(f"  Sharpe ratio           : {metrics['sharpe']:.3f}")
    print(f"  Max drawdown           : {metrics['max_drawdown']:.2f}%")

    print("\nGenerating HTML report...")
    html = make_html(history, metrics)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Report saved to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
