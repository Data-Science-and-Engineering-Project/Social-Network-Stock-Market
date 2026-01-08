#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate managed money (AUM) per quarter and visualize it.

This script:
1. Joins holdings_filtered_new with ticker_to_cusip to get ticker from cusip
2. Joins with ticker_prices to get price for each holding
3. Calculates value = sshprnamt * price for each holding
4. Sums values per period_start (quarter start date)
5. Creates a visualization showing managed money over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ETL.data_handlers.db_data_handler.postgres_handler import PostgresHandler


def calculate_managed_money_per_quarter():
    """
    Calculate managed money (AUM) per quarter by joining holdings with prices.
    
    Returns:
        pd.DataFrame with columns: period_start, total_managed_money
    """
    # Initialize database handler
    handler = PostgresHandler()
    
    try:
        # Connect to database
        print("Connecting to database...")
        handler.connect()
        
        # SQL query to calculate managed money per quarter
        # Join holdings_filtered_new -> ticker_to_cusip -> ticker_prices
        # Note: ticker_prices.period_start is stored as text, need to cast to date
        # Calculate value = sshprnamt * price, then sum by period_start
        query = """
        SELECT 
            hf.period_start,
            hf.year,
            hf.quarter,
            SUM(hf.sshprnamt * tp.price) AS total_managed_money,
            COUNT(*) AS holding_count
        FROM holdings_filtered_new hf
        INNER JOIN ticker_to_cusip ttc ON hf.cusip = ttc.cusip
        INNER JOIN ticker_prices tp ON ttc.ticker = tp.ticker 
            AND hf.period_start = CAST(tp.period_start AS DATE)
        WHERE 
            hf.sshprnamt IS NOT NULL 
            AND hf.sshprnamt > 0
            AND tp.price IS NOT NULL
            AND tp.price > 0
            AND tp.period_start IS NOT NULL
            AND tp.period_start != ''
        GROUP BY hf.period_start, hf.year, hf.quarter
        ORDER BY hf.period_start;
        """
        
        print("Executing query to calculate managed money per quarter...")
        print("This may take a few moments...")
        
        # Execute query and load into DataFrame
        # Suppress pandas warning about psycopg2 connection
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            df = pd.read_sql_query(query, handler.connection)
        
        print(f"Found {len(df)} quarters with data")
        
        return df
        
    except Exception as e:
        print(f"Error calculating managed money: {str(e)}")
        raise
    finally:
        handler.disconnect()


def create_visualization(df: pd.DataFrame, output_file: str = "managed_money_per_quarter.png"):
    """
    Create a visualization of managed money over time.
    
    Args:
        df: DataFrame with period_start and total_managed_money columns
        output_file: Path to save the visualization
    """
    if df.empty:
        print("No data to visualize")
        return
    
    # Convert period_start to datetime if it's not already
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Sort by period_start to ensure proper ordering
    df = df.sort_values('period_start')
    
    # Create quarter label for display (e.g., "Q2 2013")
    df['quarter_label'] = df.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the data
    ax.plot(df['period_start'], df['total_managed_money'], 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    # Format y-axis to show values in billions or millions
    max_value = df['total_managed_money'].max()
    if max_value >= 1e12:
        # Trillions
        df['total_managed_money_formatted'] = df['total_managed_money'] / 1e12
        ylabel = 'Total Managed Money (Trillions USD)'
    elif max_value >= 1e9:
        # Billions
        df['total_managed_money_formatted'] = df['total_managed_money'] / 1e9
        ylabel = 'Total Managed Money (Billions USD)'
    else:
        # Millions
        df['total_managed_money_formatted'] = df['total_managed_money'] / 1e6
        ylabel = 'Total Managed Money (Millions USD)'
    
    # Replot with formatted values
    ax.clear()
    ax.plot(df['period_start'], df['total_managed_money_formatted'], 
            marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Managed Money')
    
    # Formatting
    ax.set_xlabel('Quarter Start Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title('Total Managed Money (AUM) by Quarter', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Format x-axis dates - use MonthLocator with interval=3 for quarters
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations for each point
    for idx, row in df.iterrows():
        ax.annotate(f'{row["total_managed_money_formatted"]:.2f}',
                   (row['period_start'], row['total_managed_money_formatted']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Show the plot
    plt.show()


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics about managed money.
    
    Args:
        df: DataFrame with period_start and total_managed_money columns
    """
    if df.empty:
        print("No data available")
        return
    
    print("\n" + "="*80)
    print("MANAGED MONEY SUMMARY")
    print("="*80)
    print(f"\nTotal Quarters: {len(df)}")
    
    # Create quarter labels for display
    df_display = df.copy()
    df_display['quarter_label'] = df_display.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
    
    first_quarter = df_display.iloc[0]['quarter_label']
    last_quarter = df_display.iloc[-1]['quarter_label']
    print(f"Date Range: {first_quarter} to {last_quarter}")
    
    print(f"\nManaged Money Statistics:")
    print(f"  Minimum: ${df['total_managed_money'].min():,.2f}")
    print(f"  Maximum: ${df['total_managed_money'].max():,.2f}")
    print(f"  Average: ${df['total_managed_money'].mean():,.2f}")
    print(f"  Median:  ${df['total_managed_money'].median():,.2f}")
    
    if 'holding_count' in df.columns:
        print(f"\nTotal Holdings Processed: {df['holding_count'].sum():,}")
        print(f"Average Holdings per Quarter: {df['holding_count'].mean():.0f}")
    
    latest = df_display.iloc[-1]
    print(f"\nLatest Quarter ({latest['quarter_label']}):")
    print(f"  Managed Money: ${latest['total_managed_money']:,.2f}")
    if 'holding_count' in latest:
        print(f"  Holdings Count: {int(latest['holding_count']):,}")
    print("="*80 + "\n")


def main():
    """Main function to execute the script."""
    print("="*80)
    print("CALCULATING MANAGED MONEY PER QUARTER")
    print("="*80)
    
    try:
        # Calculate managed money per quarter
        df = calculate_managed_money_per_quarter()
        
        if df.empty:
            print("No data found. Please check your database tables.")
            return
        
        # Print summary
        print_summary(df)
        
        # Create visualization
        print("\nCreating visualization...")
        create_visualization(df)
        
        # Optionally save to CSV
        csv_file = "managed_money_per_quarter.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nData saved to: {csv_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

