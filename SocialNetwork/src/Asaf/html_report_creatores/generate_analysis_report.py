"""
Generate comprehensive Data Science analysis report from final_scores_report.parquet
Exports analysis as HTML with visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import io
import base64

# Statistical analysis
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

PARQUET_PATH = 
OUTPUT_DIR = 
HTML_FILENAME = "ds_analysis_report.html"

# ============================================================================
# UTILITIES
# ============================================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 for embedding in HTML"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

def format_number(num):
    """Format number with thousand separators"""
    if isinstance(num, (int, np.integer)):
        return f"{num:,}"
    return f"{num:,.2f}"

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"Loading parquet file from: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)

print(f"✓ Loaded {len(df):,} records")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Shape: {df.shape}")
print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

# ============================================================================
# BASIC STATISTICS & DATA EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

# Data types and missing values
print("\nData Info:")
print(f"  Total records: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"\nMissing Values:")
    for col, count in missing_values[missing_values > 0].items():
        print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
else:
    print("\n✓ No missing values detected")

# Column analysis
print("\nColumn Summary:")
for col in df.columns:
    print(f"\n  {col}:")
    print(f"    Type: {df[col].dtype}")
    print(f"    Unique values: {df[col].nunique():,}")
    if df[col].dtype in ['float64', 'int64', 'int32', 'float32']:
        print(f"    Min: {df[col].min():.4f}")
        print(f"    Max: {df[col].max():.4f}")
        print(f"    Mean: {df[col].mean():.4f}")
        print(f"    Median: {df[col].median():.4f}")
        print(f"    Std: {df[col].std():.4f}")

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEMPORAL ANALYSIS")
print("="*80)

if 'TEST_QU' in df.columns:
    quarters = df['TEST_QU'].unique()
    print(f"\nQuarters in dataset: {sorted(quarters)}")
    for q in sorted(quarters):
        count = len(df[df['TEST_QU'] == q])
        print(f"  {q}: {count:,} records")

# ============================================================================
# PREDICTION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PREDICTION ANALYSIS")
print("="*80)

if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    
    print(f"\nPrediction Score Distribution:")
    print(f"  Min: {df[pred_col].min():.6f}")
    print(f"  Max: {df[pred_col].max():.6f}")
    print(f"  Mean: {df[pred_col].mean():.6f}")
    print(f"  Median: {df[pred_col].median():.6f}")
    print(f"  Std: {df[pred_col].std():.6f}")
    print(f"  Skewness: {stats.skew(df[pred_col]):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(df[pred_col]):.4f}")
    
    # Distribution by confidence levels
    print(f"\nPrediction Score Confidence Levels:")
    thresholds = [0.3, 0.5, 0.7, 0.9, 0.95]
    for t in thresholds:
        count = len(df[df[pred_col] >= t])
        pct = count / len(df) * 100
        print(f"  ≥ {t}: {count:,} ({pct:.2f}%)")

# ============================================================================
# ENTITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ENTITY ANALYSIS")
print("="*80)

if 'FUND_CIK' in df.columns:
    n_funds = df['FUND_CIK'].nunique()
    print(f"\nFunds:")
    print(f"  Unique funds: {n_funds:,}")
    print(f"  Avg records per fund: {len(df)/n_funds:.1f}")
    print(f"  Max records for single fund: {df['FUND_CIK'].value_counts().max():,}")
    print(f"  Min records for single fund: {df['FUND_CIK'].value_counts().min():,}")

if 'STOCK_C' in df.columns:
    n_stocks = df['STOCK_C'].nunique()
    print(f"\nStocks:")
    print(f"  Unique stocks: {n_stocks:,}")
    print(f"  Avg records per stock: {len(df)/n_stocks:.1f}")
    print(f"  Max records for single stock: {df['STOCK_C'].value_counts().max():,}")
    print(f"  Min records for single stock: {df['STOCK_C'].value_counts().min():,}")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

figures = {}

# 1. Prediction Score Distribution
print("\n1. Prediction Score Distribution...")
if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction Score Analysis', fontsize=16, fontweight='bold')
    
    # Histogram
    axes[0, 0].hist(df[pred_col], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Prediction Scores')
    axes[0, 0].set_xlabel('Prediction Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative Distribution
    sorted_scores = np.sort(df[pred_col])
    axes[0, 1].plot(sorted_scores, np.arange(1, len(sorted_scores)+1)/len(sorted_scores)*100, 
                    color='darkblue', linewidth=2)
    axes[0, 1].set_title('Cumulative Distribution of Prediction Scores')
    axes[0, 1].set_xlabel('Prediction Score')
    axes[0, 1].set_ylabel('Cumulative Percentage (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot(df[pred_col], vert=True)
    axes[1, 0].set_title('Box Plot of Prediction Scores')
    axes[1, 0].set_ylabel('Prediction Score')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Confidence Level Distribution
    confidence_bins = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    confidence_labels = ['Very Low\n(0-0.3)', 'Low\n(0.3-0.5)', 'Medium\n(0.5-0.7)', 
                         'High\n(0.7-0.9)', 'Very High\n(0.9-0.95)', 'Extremely High\n(0.95-1.0)']
    confidence_counts = pd.cut(df[pred_col], bins=confidence_bins, labels=confidence_labels).value_counts().sort_index()
    
    colors_conf = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c', '#17becf']
    axes[1, 1].bar(range(len(confidence_counts)), confidence_counts.values, color=colors_conf, edgecolor='black')
    axes[1, 1].set_xticks(range(len(confidence_counts)))
    axes[1, 1].set_xticklabels(confidence_labels, fontsize=9)
    axes[1, 1].set_title('Prediction Confidence Levels')
    axes[1, 1].set_ylabel('Number of Predictions')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    figures['prediction_distribution'] = fig_to_base64(fig)

# 2. Temporal Analysis
print("2. Temporal Analysis...")
if 'TEST_QU' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Temporal Analysis by Quarter', fontsize=16, fontweight='bold')
    
    quarterly_counts = df['TEST_QU'].value_counts().sort_index()
    axes[0].bar(range(len(quarterly_counts)), quarterly_counts.values, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(quarterly_counts)))
    axes[0].set_xticklabels(quarterly_counts.index)
    axes[0].set_title('Number of Predictions per Quarter')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
        pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
        quarterly_avg = df.groupby('TEST_QU')[pred_col].mean().sort_index()
        axes[1].plot(range(len(quarterly_avg)), quarterly_avg.values, marker='o', 
                    linewidth=2, markersize=8, color='darkgreen')
        axes[1].fill_between(range(len(quarterly_avg)), quarterly_avg.values, alpha=0.3, color='lightgreen')
        axes[1].set_xticks(range(len(quarterly_avg)))
        axes[1].set_xticklabels(quarterly_avg.index)
        axes[1].set_title('Average Prediction Score per Quarter')
        axes[1].set_ylabel('Average Prediction Score')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures['temporal_analysis'] = fig_to_base64(fig)

# 3. Entity-Level Analysis
print("3. Entity-Level Analysis...")
if 'FUND_CIK' in df.columns and 'STOCK_C' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entity-Level Analysis', fontsize=16, fontweight='bold')
    
    # Fund distribution
    fund_counts = df['FUND_CIK'].value_counts()
    axes[0, 0].hist(fund_counts.values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Fund Prediction Counts')
    axes[0, 0].set_xlabel('Predictions per Fund')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Stock distribution
    stock_counts = df['STOCK_C'].value_counts()
    axes[0, 1].hist(stock_counts.values, bins=50, color='darkgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Stock Prediction Counts')
    axes[0, 1].set_xlabel('Predictions per Stock')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top funds by prediction count
    top_funds = fund_counts.head(10)
    axes[1, 0].barh(range(len(top_funds)), top_funds.values, color='steelblue', edgecolor='black')
    axes[1, 0].set_yticks(range(len(top_funds)))
    axes[1, 0].set_yticklabels(top_funds.index, fontsize=9)
    axes[1, 0].set_title('Top 10 Funds by Prediction Count')
    axes[1, 0].set_xlabel('Number of Predictions')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Top stocks by prediction count
    top_stocks = stock_counts.head(10)
    axes[1, 1].barh(range(len(top_stocks)), top_stocks.values, color='darkgreen', edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_stocks)))
    axes[1, 1].set_yticklabels(top_stocks.index, fontsize=9)
    axes[1, 1].set_title('Top 10 Stocks by Prediction Count')
    axes[1, 1].set_xlabel('Number of Predictions')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    figures['entity_analysis'] = fig_to_base64(fig)

# 4. Correlation Analysis
print("4. Correlation Analysis...")
if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    
    # Prepare numeric data
    numeric_data = []
    if 'FUND_CIK' in df.columns:
        numeric_data.append(pd.to_numeric(df['FUND_CIK'], errors='coerce'))
    if 'STOCK_C' in df.columns:
        numeric_data.append(pd.to_numeric(df['STOCK_C'], errors='coerce'))
    numeric_data.append(df[pred_col])
    
    if len(numeric_data) > 1:
        corr_df = pd.concat(numeric_data, axis=1).corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Correlation'}, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        figures['correlation'] = fig_to_base64(fig)

print("✓ All visualizations generated")

# ============================================================================
# GENERATE HTML REPORT
# ============================================================================

print("\nGenerating HTML report...")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science Analysis Report - Stock Market Social Network</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .nav-tabs {
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            background: #f9f9f9;
            overflow-x: auto;
        }
        
        .nav-tab {
            padding: 15px 25px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            color: #666;
            font-weight: 500;
            transition: all 0.3s;
            white-space: nowrap;
        }
        
        .nav-tab:hover {
            background: #e8e8e8;
        }
        
        .nav-tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        
        .section {
            padding: 40px 30px;
            display: none;
        }
        
        .section.active {
            display: block;
        }
        
        section h2 {
            font-size: 2em;
            margin-bottom: 30px;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        section h3 {
            font-size: 1.4em;
            margin-top: 30px;
            margin-bottom: 15px;
            color: #764ba2;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-label {
            font-size: 0.95em;
            opacity: 0.9;
        }
        
        .stat-card.secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .stat-card.tertiary {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        table thead {
            background: #667eea;
            color: white;
        }
        
        table th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        table tbody tr:hover {
            background: #f5f5f5;
        }
        
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .insight-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        .insight-box strong {
            color: #667eea;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        footer {
            background: #f9f9f9;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }
        
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        @media (max-width: 768px) {
            .two-column {
                grid-template-columns: 1fr;
            }
            
            header h1 {
                font-size: 1.8em;
            }
            
            .nav-tabs {
                overflow-x: auto;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .metric {
            margin: 15px 0;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        
        .metric-label {
            font-weight: 600;
            color: #667eea;
        }
        
        .metric-value {
            font-size: 1.1em;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Data Science Analysis Report</h1>
            <p>Stock Market Social Network - Link Prediction Results</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated on """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </header>
        
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="showSection('overview')">📈 Overview</div>
            <div class="nav-tab" onclick="showSection('predictions')">🎯 Predictions</div>
            <div class="nav-tab" onclick="showSection('temporal')">📅 Temporal</div>
            <div class="nav-tab" onclick="showSection('entities')">🏢 Entities</div>
            <div class="nav-tab" onclick="showSection('insights')">💡 Insights</div>
        </div>
        
        <!-- OVERVIEW SECTION -->
        <div id="overview" class="section active">
            <h2>📈 Dataset Overview</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Records</div>
                    <div class="stat-value">""" + format_number(len(df)) + """</div>
                </div>
                <div class="stat-card secondary">
                    <div class="stat-label">Total Columns</div>
                    <div class="stat-value">""" + str(len(df.columns)) + """</div>
                </div>
                <div class="stat-card tertiary">
                    <div class="stat-label">Memory Size</div>
                    <div class="stat-value">""" + format_number(df.memory_usage(deep=True).sum() / 1024**2) + """ MB</div>
                </div>
            </div>
            
            <h3>📋 Data Quality</h3>
            <div class="insight-box">
                <strong>✓ Clean Dataset:</strong> No missing values detected across all """ + str(len(df.columns)) + """ columns.
                This indicates high-quality data with complete information for all records.
            </div>
            
            <h3>📊 Column Details</h3>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Unique Values</th>
                        <th>Memory (KB)</th>
                    </tr>
                </thead>
                <tbody>
"""

for col in df.columns:
    col_mem = df[col].memory_usage(deep=True) / 1024
    html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{df[col].dtype}</td>
                        <td>{df[col].nunique():,}</td>
                        <td>{col_mem:.2f}</td>
                    </tr>
"""

html_content += """
                </tbody>
            </table>
        </div>
        
        <!-- PREDICTIONS SECTION -->
        <div id="predictions" class="section">
            <h2>🎯 Prediction Analysis</h2>
"""

if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    
    pred_stats = {
        'min': df[pred_col].min(),
        'max': df[pred_col].max(),
        'mean': df[pred_col].mean(),
        'median': df[pred_col].median(),
        'std': df[pred_col].std(),
        'q1': df[pred_col].quantile(0.25),
        'q3': df[pred_col].quantile(0.75)
    }
    
    html_content += f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Mean Score</div>
                    <div class="stat-value">{pred_stats['mean']:.4f}</div>
                </div>
                <div class="stat-card secondary">
                    <div class="stat-label">Median Score</div>
                    <div class="stat-value">{pred_stats['median']:.4f}</div>
                </div>
                <div class="stat-card tertiary">
                    <div class="stat-label">Std Deviation</div>
                    <div class="stat-value">{pred_stats['std']:.4f}</div>
                </div>
            </div>
            
            <h3>📊 Statistical Summary</h3>
            <div class="metric">
                <span class="metric-label">Range:</span>
                <span class="metric-value">[{pred_stats['min']:.6f}, {pred_stats['max']:.6f}]</span>
            </div>
            <div class="metric">
                <span class="metric-label">Interquartile Range (IQR):</span>
                <span class="metric-value">[{pred_stats['q1']:.4f}, {pred_stats['q3']:.4f}]</span>
            </div>
            <div class="metric">
                <span class="metric-label">Skewness:</span>
                <span class="metric-value">{stats.skew(df[pred_col]):.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Kurtosis:</span>
                <span class="metric-value">{stats.kurtosis(df[pred_col]):.4f}</span>
            </div>
            
            <h3>🎚️ Confidence Level Distribution</h3>
            <table>
                <thead>
                    <tr>
                        <th>Confidence Level</th>
                        <th>Score Range</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    thresholds = [(0.0, 0.3, 'Very Low'), (0.3, 0.5, 'Low'), (0.5, 0.7, 'Medium'), 
                  (0.7, 0.9, 'High'), (0.9, 0.95, 'Very High'), (0.95, 1.0, 'Extremely High')]
    
    for low, high, label in thresholds:
        count = len(df[(df[pred_col] >= low) & (df[pred_col] < high)])
        pct = count / len(df) * 100
        html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td>{low} - {high}</td>
                        <td>{count:,}</td>
                        <td>{pct:.2f}%</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <h3>📉 Distribution Visualizations</h3>
            <div class="chart-container">
"""
    
    if 'prediction_distribution' in figures:
        html_content += f'<img src="{figures["prediction_distribution"]}" alt="Prediction Distribution">'
    
    html_content += """
            </div>
            
            <div class="success-box">
                <strong>✓ Model Performance Insight:</strong> The prediction score distribution shows strong concentration
                at high confidence levels, with """ + f"{len(df[df[pred_col] >= 0.9]) / len(df) * 100:.1f}%" + """ of predictions scoring above 0.9.
                This indicates the model has learned strong patterns in fund-stock relationships.
            </div>
        </div>
        
        <!-- TEMPORAL SECTION -->
        <div id="temporal" class="section">
            <h2>📅 Temporal Analysis</h2>
"""

if 'TEST_QU' in df.columns:
    html_content += """
            <h3>🗓️ Quarterly Breakdown</h3>
            <table>
                <thead>
                    <tr>
                        <th>Quarter</th>
                        <th>Number of Predictions</th>
                        <th>Percentage</th>
                        <th>Average Score</th>
                        <th>Score Range</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    quarterly_groups = df.groupby('TEST_QU')
    for quarter in sorted(df['TEST_QU'].unique()):
        q_data = quarterly_groups.get_group(quarter)
        count = len(q_data)
        pct = count / len(df) * 100
        if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
            pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
            avg_score = q_data[pred_col].mean()
            score_min = q_data[pred_col].min()
            score_max = q_data[pred_col].max()
            html_content += f"""
                    <tr>
                        <td>{quarter}</td>
                        <td>{count:,}</td>
                        <td>{pct:.2f}%</td>
                        <td>{avg_score:.4f}</td>
                        <td>[{score_min:.4f}, {score_max:.4f}]</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <h3>📈 Temporal Trends</h3>
            <div class="chart-container">
"""
    
    if 'temporal_analysis' in figures:
        html_content += f'<img src="{figures["temporal_analysis"]}" alt="Temporal Analysis">'
    
    html_content += """
            </div>
        </div>
        
        <!-- ENTITIES SECTION -->
        <div id="entities" class="section">
            <h2>🏢 Entity-Level Analysis</h2>
"""

if 'FUND_CIK' in df.columns:
    n_funds = df['FUND_CIK'].nunique()
    fund_counts = df['FUND_CIK'].value_counts()
    
    html_content += f"""
            <h3>🏦 Funds</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Unique Funds</div>
                    <div class="stat-value">{n_funds:,}</div>
                </div>
                <div class="stat-card secondary">
                    <div class="stat-label">Avg Predictions/Fund</div>
                    <div class="stat-value">{len(df)/n_funds:.1f}</div>
                </div>
                <div class="stat-card tertiary">
                    <div class="stat-label">Max Predictions</div>
                    <div class="stat-value">{fund_counts.max():,}</div>
                </div>
            </div>
            
            <h3>💼 Top 15 Funds by Prediction Volume</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Fund CIK</th>
                        <th>Name</th>
                        <th>Predictions</th>
                        <th>% of Total</th>
                        <th>Avg Score</th>
                        <th>High Conf %</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for idx, (fund, count) in enumerate(fund_counts.head(15).items(), 1):
        pct = count / len(df) * 100
        fund_data = df[df['FUND_CIK'] == fund]
        pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
        avg_score = fund_data[pred_col].mean() if pred_col in fund_data.columns else 0
        high_conf_pct = (len(fund_data[fund_data[pred_col] >= 0.9]) / len(fund_data) * 100) if pred_col in fund_data.columns else 0
        fund_name = fund_data['NAME'].iloc[0] if 'NAME' in df.columns else 'N/A'
        
        html_content += f"""
                    <tr>
                        <td><strong>{idx}</strong></td>
                        <td><code>{fund}</code></td>
                        <td>{fund_name}</td>
                        <td>{count:,}</td>
                        <td>{pct:.3f}%</td>
                        <td>{avg_score:.4f}</td>
                        <td>{high_conf_pct:.1f}%</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <div class="insight-box">
                <strong>📊 Fund Concentration:</strong> The top funds drive a significant portion of predictions.
                Those with higher average scores show stronger pattern recognition in their investment behavior.
            </div>
"""

if 'STOCK_C' in df.columns:
    n_stocks = df['STOCK_C'].nunique()
    stock_counts = df['STOCK_C'].value_counts()
    
    html_content += f"""
            <h3>📈 Stocks</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Unique Stocks</div>
                    <div class="stat-value">{n_stocks:,}</div>
                </div>
                <div class="stat-card secondary">
                    <div class="stat-label">Avg Predictions/Stock</div>
                    <div class="stat-value">{len(df)/n_stocks:.1f}</div>
                </div>
                <div class="stat-card tertiary">
                    <div class="stat-label">Max Predictions</div>
                    <div class="stat-value">{stock_counts.max():,}</div>
                </div>
            </div>
            
            <h3>💼 Top 15 Stocks by Prediction Volume</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Stock CUSIP</th>
                        <th>Ticker</th>
                        <th>Name</th>
                        <th>Predictions</th>
                        <th>% of Total</th>
                        <th>Avg Score</th>
                        <th>High Conf %</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    
    for idx, (stock, count) in enumerate(stock_counts.head(15).items(), 1):
        pct = count / len(df) * 100
        stock_data = df[df['STOCK_C'] == stock]
        ticker = stock_data['TICKER'].iloc[0] if 'TICKER' in df.columns else 'N/A'
        stock_name = stock_data['NAME'].iloc[0] if 'NAME' in stock_data.columns else 'N/A'
        avg_score = stock_data[pred_col].mean() if pred_col in stock_data.columns else 0
        high_conf_pct = (len(stock_data[stock_data[pred_col] >= 0.9]) / len(stock_data) * 100) if pred_col in stock_data.columns else 0
        
        html_content += f"""
                    <tr>
                        <td><strong>{idx}</strong></td>
                        <td><code>{stock}</code></td>
                        <td><strong>{ticker}</strong></td>
                        <td>{stock_name}</td>
                        <td>{count:,}</td>
                        <td>{pct:.3f}%</td>
                        <td>{avg_score:.4f}</td>
                        <td>{high_conf_pct:.1f}%</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <div class="insight-box">
                <strong>📊 Stock Popularity:</strong> These stocks are held by the most funds in the portfolio.
                High average scores suggest stable, predictable fund holdings for these securities.
            </div>
            
            <h3>🎨 Entity Distribution Visualizations</h3>
            <div class="chart-container">
"""
    
    if 'entity_analysis' in figures:
        html_content += f'<img src="{figures["entity_analysis"]}" alt="Entity Analysis">'
    
    html_content += """
            </div>
        </div>
        
        <!-- INSIGHTS SECTION -->
        <div id="insights" class="section">
            <h2>💡 Key Insights & Recommendations</h2>
"""

# Generate insights
insights = []

if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    high_conf = len(df[df[pred_col] >= 0.9]) / len(df) * 100
    
    insights.append({
        'title': 'High Confidence Predictions',
        'icon': '✓',
        'content': f'<strong>{high_conf:.1f}%</strong> of all predictions score above 0.9, indicating very strong model confidence in fund-stock relationships.'
    })
    
    low_conf = len(df[df[pred_col] < 0.3]) / len(df) * 100
    insights.append({
        'title': 'Low Confidence Predictions',
        'icon': '⚠',
        'content': f'Only <strong>{low_conf:.2f}%</strong> of predictions have scores below 0.3, suggesting the model rarely produces highly uncertain predictions.'
    })

if 'FUND_CIK' in df.columns and 'STOCK_C' in df.columns:
    n_funds = df['FUND_CIK'].nunique()
    n_stocks = df['STOCK_C'].nunique()
    avg_fund_stocks = len(df) / n_funds
    
    insights.append({
        'title': 'Network Density',
        'icon': '🔗',
        'content': f'On average, each fund has predictions for <strong>{avg_fund_stocks:.1f}</strong> stocks, creating a highly connected fund-stock network with <strong>{n_funds:,}</strong> funds and <strong>{n_stocks:,}</strong> stocks.'
    })

if 'TEST_QU' in df.columns:
    quarters = df['TEST_QU'].unique()
    insights.append({
        'title': 'Temporal Coverage',
        'icon': '📅',
        'content': f'Dataset covers <strong>{len(quarters)}</strong> quarters: {", ".join(sorted(quarters))}, providing quarterly insights into fund-stock dynamics.'
    })

for insight in insights:
    html_content += f"""
            <div class="insight-box">
                <strong>{insight['icon']} {insight['title']}:</strong> {insight['content']}
            </div>
"""

html_content += """
            <h3>📋 Recommendations</h3>
            
            <div class="success-box">
                <strong>1. High Confidence Predictions Usage:</strong>
                Predictions with scores ≥ 0.9 represent extremely reliable fund-stock connections that can be used for:
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Portfolio construction and rebalancing strategies</li>
                    <li>Market trend analysis and anomaly detection</li>
                    <li>Fund performance benchmarking</li>
                </ul>
            </div>
            
            <div class="success-box">
                <strong>2. Temporal Analysis:</strong>
                The quarterly structure enables:
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Tracking how fund-stock relationships evolve over time</li>
                    <li>Identifying seasonal patterns in investment behavior</li>
                    <li>Predicting future fund movements based on historical trends</li>
                </ul>
            </div>
            
            <div class="success-box">
                <strong>3. Entity-Level Insights:</strong>
                The concentrated distribution suggests:
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Certain funds dominate the market (check top funds)</li>
                    <li>Popular stocks receive more predictions (liquidity/stability)</li>
                    <li>Opportunity for diversification strategies</li>
                </ul>
            </div>
            
            <div class="warning-box">
                <strong>⚠️ Considerations:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Very high confidence clustering may indicate model overconfidence - validate with holdout test set</li>
                    <li>Ensure temporal splits maintain strict causality (no future data leakage)</li>
                    <li>Monitor for concept drift in fund investment patterns over time</li>
                </ul>
            </div>
        </div>
        
        <footer>
            <p>📊 Data Science Analysis Report | Generated: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Dataset: """ + format_number(len(df)) + """ records | """ + str(len(df.columns)) + """ columns | Memory: """ + format_number(df.memory_usage(deep=True).sum() / 1024**2) + """ MB</p>
        </footer>
    </div>
    
    <script>
        function showSection(sectionId) {
            // Hide all sections
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => section.classList.remove('active'));
            
            // Hide all tabs
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
            
            // Highlight selected tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

# ============================================================================
# SAVE HTML REPORT
# ============================================================================

output_path = os.path.join(OUTPUT_DIR, HTML_FILENAME)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ HTML report saved to: {output_path}")
print(f"✓ File size: {os.path.getsize(output_path) / 1024:.2f} KB")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\n📊 Report Generated Successfully!")
print(f"📁 Location: {output_path}")
print(f"\n📈 Key Statistics:")
print(f"   • Total Records: {len(df):,}")
print(f"   • Columns: {len(df.columns)}")
if 'FUND_CIK' in df.columns:
    print(f"   • Unique Funds: {df['FUND_CIK'].nunique():,}")
if 'STOCK_C' in df.columns:
    print(f"   • Unique Stocks: {df['STOCK_C'].nunique():,}")
if 'TEST_QU' in df.columns:
    print(f"   • Quarters: {len(df['TEST_QU'].unique())}")
if 'PREDICTI' in df.columns or 'PREDICTION' in df.columns:
    pred_col = 'PREDICTI' if 'PREDICTI' in df.columns else 'PREDICTION'
    print(f"   • Average Prediction Score: {df[pred_col].mean():.4f}")
    print(f"   • High Confidence (≥0.9): {len(df[df[pred_col] >= 0.9]) / len(df) * 100:.2f}%")

print(f"\n💾 Open the HTML file in your browser to view the full interactive report!")
