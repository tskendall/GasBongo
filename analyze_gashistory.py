import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_historical_distribution_analysis():
    print("🤖 Bootstrapping Historical Data Audit & Correlation Engine...")
    
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
    print(f"Analyzing {len(df)} total operational days across markets: {tracked_markets}\n")
    
    # ----------------------------------------------------
    # PHASE 1: COMPUTE PURE DAILY CHANGES (DELTAS)
    # ----------------------------------------------------
    # Daily Delta Change Matrix for tracking raw jumps/velocity
    delta_df = df[['Date']].copy()
    for m in tracked_markets:
        delta_df[m] = df[m].diff()
    delta_df = delta_df.dropna().reset_index(drop=True)
    
    # Raw mathematical diff frame for correlation logic metrics
    delta_matrix = df[tracked_markets].diff().dropna().reset_index(drop=True)
    
    # ----------------------------------------------------
    # PHASE 2: REGIONAL CORRELATION ARRAYS (STDOUT PRINT)
    # ----------------------------------------------------
    price_corr = df[tracked_markets].corr()
    delta_corr = delta_matrix.corr()
    
    print("================== PRICE LEVEL CORRELATION MATRIX ==================")
    print(price_corr.round(3).to_string())
    print("\n================ DAILY CHANGE (DELTA) CORRELATION ==================")
    print(delta_corr.round(3).to_string())
    print("\n*Insight: High delta correlation means volatility spikes simultaneously across nodes.")
    
    # ----------------------------------------------------
    # PHASE 3: RISK DISTRIBUTION VOLATILITY STATS
    # ----------------------------------------------------
    stats_records = []
    for m in tracked_markets:
        m_deltas = delta_matrix[m]
        min_cents = m_deltas.min() * 100
        max_cents = m_deltas.max() * 100
        range_cents = max_cents - min_cents
        avg_abs_move = m_deltas.abs().mean() * 100
        
        stats_records.append({
            'Market': m,
            'Max_Daily_Drop': f"{min_cents:+.2f}¢",
            'Max_Daily_Spike': f"{max_cents:+.2f}¢",
            'Total_Volatility_Range': f"{range_cents:.2f}¢",
            'Avg_Absolute_Daily_Shift': f"{avg_abs_move:.2f}¢"
        })
        
    stats_df = pd.DataFrame(stats_records)
    print("\n================== REGIONAL DAILY MOVEMENT STATS ===================")
    print(stats_df.to_string(index=False))
    
    # ----------------------------------------------------
    # PHASE 4: TARGET PROFILE ANALYSIS (NC, TX, US EXTREMES)
    # ----------------------------------------------------
    print("\n================= STRATEGIC MARKET PERTURBATION FOCUS =================")
    focus_markets = [f for f in ['US', 'TX', 'NC'] if f in delta_matrix.columns]
    
    for m in focus_markets:
        m_deltas = delta_matrix[m]
        idx_max_up = m_deltas.idxmax()
        idx_max_down = m_deltas.idxmin()
        
        date_max_up = df.loc[idx_max_up + 1, 'Date'].strftime('%Y-%m-%d')
        date_max_down = df.loc[idx_max_down + 1, 'Date'].strftime('%Y-%m-%d')
        
        print(f"📍 {m:4} | Max Daily SPIKE: {m_deltas.max()*100:+.3f}¢ on {date_max_up} | Max Daily DROP: {m_deltas.min()*100:+.3f}¢ on {date_max_down}")
    print("=======================================================================")

    # Trigger visualization suite
    output_dir = "plots_summary"
    os.makedirs(output_dir, exist_ok=True)
    market_colors = {'US': '#1f77b4', 'CA': '#e377c2', 'NC': '#ff7f0e', 'GA': '#2ca02c', 'SC': '#bcbd22', 'OR': '#9467bd', 'TX': '#d62728', 'NY': '#17becf'}
    spike_date = pd.to_datetime('2026-09-16')
    
    generate_isolated_facets(df, tracked_markets, market_colors, spike_date, output_dir)

def generate_isolated_facets(df, tracked_markets, market_colors, spike_date, output_dir):
    """
    Generates isolated high-resolution panels with independent local zooming to capture micro perturbations.
    """
    print("\n📈 Executing Method 1: Generating Isolated Facet Panel Grid...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    
    for idx, m in enumerate(tracked_markets):
        ax = axes[idx]
        color = market_colors.get(m, '#7f7f7f')
        
        # Plot raw, un-smoothed historical closes with clear daily markers
        ax.plot(df['Date'], df[m], color=color, linewidth=1.5, marker='o', markersize=3, label=f"{m} Raw")
        
        # Lock independent local axis framing limits around the min/max boundaries
        y_min = float(df[m].min()) - 0.015
        y_max = float(df[m].max()) + 0.015
        ax.set_ylim(y_min, y_max)
        
        if df['Date'].min() <= spike_date <= df['Date'].max():
            ax.axvline(x=spike_date, color='black', linestyle=':', alpha=0.5, linewidth=1.0)
            
        ax.set_title(f"Market Profile: {m}", fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('$%.3f'))
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        
    for unused_idx in range(len(tracked_markets), len(axes)):
        fig.delaxes(axes[unused_idx])
        
    plt.suptitle("Isolated Market Profiles: Raw Prices & Intraday Trajectories", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    facet_filename = os.path.join(output_dir, "macro_market_high_res_panels.png")
    plt.savefig(facet_filename, dpi=150)
    plt.close()
    print(f"✅ Success! Isolated panel file generated at: '{facet_filename}'")
    
    # Direct chaining step to call the daily delta comparison overlay
    # Recomputes internal diffs from main script loop to stay data decoupled
    delta_df = df[['Date']].copy()
    for m in tracked_markets:
        delta_df[m] = df[m].diff()
    delta_df = delta_df.dropna().reset_index(drop=True)
    
    generate_overlaid_deltas(delta_df, tracked_markets, market_colors, spike_date, output_dir)

def generate_overlaid_deltas(delta_df, tracked_markets, market_colors, spike_date, output_dir):
    """
    Overlays pure daily variation jumps (cents per gallon) on a single unified grid timeline canvas.
    """
    print("📈 Executing Method 2: Generating Overlaid Delta Volatility Chart...")
    plt.figure(figsize=(13, 6.5))
    
    for m in tracked_markets:
        color = market_colors.get(m, '#7f7f7f')
        linewidth = 2.0 if m in ['US', 'NC', 'GA', 'TX'] else 1.2
        alpha = 0.85 if m in ['US', 'NC', 'GA', 'TX'] else 0.45
        
        # Multiply by 100 to present relative movements cleanly in cents
        plt.plot(delta_df['Date'], delta_df[m] * 100, color=color, 
                 linewidth=linewidth, alpha=alpha, marker='o', markersize=3, label=f"{m} Day Delta")

    if delta_df['Date'].min() <= spike_date <= delta_df['Date'].max():
        plt.axvline(x=spike_date, color='black', linestyle='-', alpha=0.7, linewidth=1.2)
        plt.text(spike_date + pd.Timedelta(days=1), delta_df[tracked_markets].max().max() * 100 - 1.2, 
                 "9/16 Volatility Wave", color='black', fontsize=9, style='italic', fontweight='bold')

    plt.title("Kalshi Gas Architecture: Overlaid Daily Price Changes & Perturbations (Cents)", fontsize=12, fontweight='bold')
    plt.xlabel("Observation Timeline", fontsize=10)
    plt.ylabel("Daily Step Shift Velocity (¢ per gallon)", fontsize=10)
    
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%+.2f¢'))
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5) # Zero reference waterline
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize='small', frameon=True)
    
    delta_filename = os.path.join(output_dir, "macro_market_overlaid_deltas.png")
    plt.tight_layout()
    plt.savefig(delta_filename, dpi=150)
    plt.close()
    print(f"✅ Success! Overlaid delta file generated at: '{delta_filename}'")

if __name__ == "__main__":
    run_historical_distribution_analysis()
