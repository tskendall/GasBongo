import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def render_layered_market_history(market_name, target_date_str):
    print(f"🤖 Extracting intraday historical snapshots for {market_name} ({target_date_str})...")
    
    ob_filepath = os.path.join("obfiles", f"ob_{market_name}_{target_date_str}.csv")
    output_dir = "plots_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(ob_filepath):
        print(f"❌ Error: No logged orderbook file found at {ob_filepath}")
        return
        
    # Read the cumulative long-form dataset
    df = pd.read_csv(ob_filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract unique logging snapshots available in the file
    all_timestamps = sorted(df['Timestamp'].unique())
    total_snapshots = len(all_timestamps)
    print(f"Found {total_snapshots} total intraday logging blocks in this file.")
    
    if total_snapshots == 0:
        return
        
    # Sample up to 6 evenly-spaced timestamps across the day to prevent chart clutter
    indices = np.linspace(0, total_snapshots - 1, min(6, total_snapshots), dtype=int)
    selected_timestamps = [all_timestamps[idx] for idx in indices]
    
    plt.figure(figsize=(11, 6))
    
    # Generate a smooth color gradient from morning (light blue) to night (dark purple)
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(selected_timestamps)))

    for idx, ts in enumerate(selected_timestamps):
        ts_df = df[df['Timestamp'] == ts].sort_values(by='Strike_Rung')
        
        # Derive the mid-market contract probability point
        ts_df['Mid_Price'] = (ts_df['Yes_Bid'] + ts_df['Yes_Ask']) / 2
        
        # Clean string time label for the legend (HH:MM)
        time_label = pd.to_datetime(ts).strftime('%H:%M')
        
        # Plot this specific time step's probability curve line
        plt.plot(ts_df['Strike_Rung'], ts_df['Mid_Price'], color=colors[idx], 
                 linewidth=2, marker='o', alpha=0.8, label=f"Snapshot @ {time_label}")
                 
    # Layer on the previous day's historical spot close for baseline context
    history_file = os.path.join("data", "kalshi_gas_60day_history.csv")
    if os.path.exists(history_file):
        try:
            df_hist = pd.read_csv(history_file)
            hist_col = 'US_National' if (market_name == 'US' and 'US_National' in df_hist.columns) else market_name
            if hist_col in df_hist.columns:
                last_settled = df_hist[hist_col].dropna().iloc[-1]
                plt.axvline(x=last_settled, color='gray', linestyle='-', linewidth=1.5, 
                            label=f"Prev Day Settled Close (${last_settled:.4f})")
        except Exception:
            pass
            
    plt.title(f"Intraday S-Curve Evolution: {market_name} (Target Market: {target_date_str})")
    plt.xlabel("Contract Strike Rungs ($)")
    plt.ylabel("Probability Premium / Midpoint Price ($)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize='small')
    
    plot_filename = os.path.join(output_dir, f"summary_drift_{market_name}_{target_date_str}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"✅ Success! Session drift visualization saved to: '{plot_filename}'")

if __name__ == "__main__":
    # Execute an analytical test on today's tracking data matrix strings
    today_str = datetime.now().strftime('%Y-%m-%d')
    tomorrow_str = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Run the visualization loop for your core focus markets
    render_layered_market_history('US', tomorrow_str)
    render_layered_market_history('NC', tomorrow_str)
