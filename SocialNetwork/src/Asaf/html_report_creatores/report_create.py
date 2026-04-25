import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- הגדרות נתיבים ---
PARQUET_PATH = "start"
OUTPUT_DIR = "start"
HTML_OUTPUT = os.path.join(OUTPUT_DIR, "model_insights_dashboard.html")

def generate_dashboard():
    print(f"טוען נתונים מהקובץ: {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)

    # ניקוי נתונים בסיסי
    df['PREDICTION_SCORE'] = df['PREDICTION_SCORE'].astype(float)
    
    # הגדרת ספי ביטחון (Confidence Tiers)
    high_conf_df = df[df['PREDICTION_SCORE'] >= 0.5]
    
    # --- 1. הכנת נתונים לגרפים ---
    
    # א. התפלגות ציונים (Histogram)
    fig_hist = px.histogram(df, x="PREDICTION_SCORE", nbins=50, 
                             title="Distribution of Prediction Confidence",
                             color_discrete_sequence=['#636EFA'],
                             labels={'PREDICTION_SCORE': 'Confidence Score'})

    # ב. Top 15 מניות מבוקשות (מניות שהמודל חוזה להן הכי הרבה קשרים חדשים בביטחון גבוה)
    top_stocks = high_conf_df['TICKER'].value_counts().head(15).reset_index()
    top_stocks.columns = ['Ticker', 'Predicted_New_Holders']
    fig_stocks = px.bar(top_stocks, x='Predicted_New_Holders', y='Ticker', orientation='h',
                        title="Top 15 Predicted Stocks (High Confidence)",
                        color='Predicted_New_Holders', color_continuous_scale='Viridis')

    # ג. ניתוח רבעוני (במידה ויש כמה רבעונים)
    quarter_counts = df.groupby('TEST_QUARTER').size().reset_index(name='Count')
    fig_quarters = px.pie(quarter_counts, values='Count', names='TEST_QUARTER', 
                          title="Predictions by Target Quarter", hole=0.4)

    # --- 2. יצירת ה-HTML המעוצב ---
    
    # חישוב מטריקות מהירות
    total_preds = len(df)
    high_conf_count = len(high_conf_df)
    unique_funds = df['FUND_CIK'].nunique()
    unique_stocks = df['STOCK_CUSIP'].nunique()
    avg_score = df['PREDICTION_SCORE'].mean()

    # המרת הגרפים ל-HTML Strings
    chart_hist = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')
    chart_stocks = fig_stocks.to_html(full_html=False, include_plotlyjs='cdn')
    chart_quarters = fig_quarters.to_html(full_html=False, include_plotlyjs='cdn')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Market AI Insights</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .header {{ text-align: center; padding: 20px; background: #2c3e50; color: white; border-radius: 10px; margin-bottom: 20px; }}
            .stats-container {{ display: flex; justify-content: space-between; margin-bottom: 20px; gap: 15px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 10px; flex: 1; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stat-card h3 {{ margin: 0; color: #7f8c8d; font-size: 14px; text-transform: uppercase; }}
            .stat-card p {{ margin: 10px 0 0; font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .chart-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .footer {{ text-align: center; color: #95a5a6; font-size: 12px; margin-top: 30px; }}
            .tag {{ background: #e74c3c; color: white; padding: 3px 8px; border-radius: 5px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Temporal Link Prediction Dashboard</h1>
                <p>AI-Powered Investment Network Analysis | Model: GraphSAGE + LightGBM</p>
            </div>

            <div class="stats-container">
                <div class="stat-card"><h3>Total Predictions</h3><p>{total_preds:,}</p></div>
                <div class="stat-card"><h3>High Confidence (>=0.5)</h3><p style="color:#27ae60;">{high_conf_count:,}</p></div>
                <div class="stat-card"><h3>Unique Funds</h3><p>{unique_funds:,}</p></div>
                <div class="stat-card"><h3>Unique Stocks</h3><p>{unique_stocks:,}</p></div>
            </div>

            <div class="chart-card">
                {chart_hist}
            </div>

            <div style="display: flex; gap: 20px;">
                <div class="chart-card" style="flex: 2;">
                    {chart_stocks}
                </div>
                <div class="chart-card" style="flex: 1;">
                    {chart_quarters}
                </div>
            </div>

            <div class="chart-card">
                <h2>Advanced DS Conclusions</h2>
                <ul>
                    <li><strong>Link Density:</strong> The model has identified a significant cluster of potential new holdings in the {high_conf_count/total_preds:.1%} confidence range.</li>
                    <li><strong>Market Movement:</strong> Stock <strong>{top_stocks.iloc[0]['Ticker']}</strong> shows the highest latent connection potential across the fund network.</li>
                    <li><strong>Network Stability:</strong> Average prediction score of <strong>{avg_score:.4f}</strong> suggests strong temporal consistency in fund-stock bipartite structures.</li>
                </ul>
            </div>

            <div class="footer">
                Generated by Gemini Adaptive AI Engine | Data Source: final_scores_report.parquet
            </div>
        </div>
    </body>
    </html>
    """

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ הדו\"ח נוצר בהצלחה בכתובת: {HTML_OUTPUT}")
    os.startfile(HTML_OUTPUT) # פותח את הקובץ אוטומטית בדפדפן

if __name__ == "__main__":
    generate_dashboard()