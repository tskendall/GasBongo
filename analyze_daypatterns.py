import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_weekday_and_weekly_analysis():
    print("🤖 Initialising Weekday Seasonal Bias & Weekly Change Engine...")
    
    history_file = os.path.join("data", "kalshi_gas_60day_history.csv")
    
    if not os.path.exists(history_file):
        print(f"❌ Error: Missing core tracking database file at '{history_file}'")
        return
        
    df = pd.read_csv(history_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    if 'US_National' in df.columns:
        df = df.rename(columns={'US_National': 'US'})
        
    tracked_markets = [col for col in ['US', 'CA', 'NC', 'GA', 'SC', 'OR', 'TX', 'NY'] if col in df.columns]
    
    # ----------------------------------------------------
    # PHASE 1: WEEKDAY BIAS MATRIX (DAY OF WEEK EXTRACTION)
    # ----------------------------------------------------
    # Extract the string day name matching each row date index
    df['Day_of_Week'] = df['Date'].dt.day_name()
    
    # Compute raw day-over-day dollar deltas
    delta_df = df[['Date', 'Day_of_Week']].copy()
    for m in tracked_markets:
        delta_df[m] = df[m].diff()
    delta_df = delta_df.dropna().reset_index(drop=True)
    
    print(f"Auditing weekday behavioral loops over {len(df)} total dataset entries...")
    
    # Establish a strict chronological sorting key for reporting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print("\n================ AVERAGE DAILY CHANGE BY DAY OF WEEK (CENTS) ================")
    # Group by the weekday tag, compute the mean change, multiply by 100 for clean cents display
    weekday_means = delta_df.groupby('Day_of_Week')[tracked_markets].mean() * 100
    weekday_means = weekday_means.reindex(weekday_order)
    print(weekday_means.round(3).to_string())
    print("*Insight: Look for heavy positive/negative spikes to detect programmatic price resets.")

    print("\n=============== MAXIMUM DAILY SPIKE BOUNDARY BY WEEKDAY ================")
    weekday_max = delta_df.groupby('Day_of_Week')[tracked_markets].max() * 100
    weekday_max = weekday_max.reindex(weekday_order)
    print(weekday_max.round(2).to_string())
    
    # ----------------------------------------------------
    # PHASE 2: WEEKDAY DELTA CO-MOVEMENT CORRELATION
    # ----------------------------------------------------
    print("\n============= DAY-OF-WEEK MOVEMENT CORRELATION ARRAY =============")
    # Pivot the data frame to evaluate if states move together on identical weekdays
    weekday_corr = delta_df[tracked_markets].corr()
    print(weekday_corr.round(3).to_string())
    
    # Trigger secondary script tracking segment for weekly shifts and plotting
    output_dir = "plots_summary"
    os.makedirs(output_dir, exist_ok=True)
    generate_weekly_trend_analysis(df, tracked_markets, output_dir, weekday_order, delta_df)

def generate_weekly_trend_analysis(df, tracked_markets, output_dir, weekday_order, delta_df):
    """
    Calculates rolling weekly net shifts, maps weekly trend correlations,
    and exports a comparison chart displaying weekday seasonal movements.
    """
    print("\n📈 Executing Phase 3: Computing Rolling 7-Day Net Shifts...")
    
    # Calculate rolling 7-day total net price change: Price(T) - Price(T-7)
    weekly_shift_df = df[['Date']].copy()
    for m in tracked_markets:
        weekly_shift_df[m] = df[m] - df[m].shift(7)
    weekly_shift_df = weekly_shift_df.dropna().reset_index(drop=True)
    
    # Compute correlation strictly across long-term weekly trends
    weekly_corr = weekly_shift_df[tracked_markets].corr()
    
    print("\n================= WEEKLY CHANGE (7-DAY NET) CORRELATION =================")
    print(weekly_corr.round(3).to_string())
    print("\n*Insight: High weekly correlation isolates long-term macro trend alignments.")

    # ----------------------------------------------------
    # PHASE 4: GENERATE WEEKDAY SEASONAL BOXPLOT GRAPH
    # ----------------------------------------------------
    print("\n🎨 Generating Weekday Volatility Portfolio Visuals...")
    
    # Focus visualization on your core trading group to keep panels crisp
    visual_focus = [f for f in ['US', 'TX', 'NC', 'GA'] if f in delta_df.columns]
    
    fig, axes = plt.subplots(len(visual_focus), 1, figsize=(12, 3 * len(visual_focus)), sharex=True)
    if len(visual_focus) == 1:
        axes = [axes]
        
    market_colors = {'US': '#1f77b4', 'TX': '#d62728', 'NC': '#ff7f0e', 'GA': '#2ca02c'}
    
    for idx, m in enumerate(visual_focus):
        ax = axes[idx]
        color = market_colors.get(m, '#7f7f7f')
        
        # Build structured lists of data points grouped by each day
        day_data_buckets = []
        for day in weekday_order:
            day_points = delta_df[delta_df['Day_of_Week'] == day][m] * 100
            day_data_buckets.append(day_points)
            
        # Draw a distribution boxplot to expose outliers, medians, and ranges for that day
        bp = ax.boxplot(day_data_buckets, labels=weekday_order, patch_artist=True,
                        showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
        
        # Color the boxes cleanly matching your platform hex style guidelines
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.6)
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
            
        ax.set_title(f"Intraday Weekday Volatility Spectrum: {m}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Daily Change (¢)", fontsize=9)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.6)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%+.1f¢'))

    plt.xlabel("Calendar Day of Week Horizon", fontsize=10)
    plt.suptitle("Gas Market Seasonality Audit: Weekday Perturbations & Rolling Weekly Trends", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, "market_weekday_seasonality.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    print(f"✅ Success! Seasonal weekday chart portfolio written to: '{plot_filename}'")

if __name__ == "__main__":
    run_weekday_and_weekly_analysis()
