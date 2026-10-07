import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_implied_consensus(df_snap):
    """
    Finds the exact sub-cent price where the market's implied probability 
    crosses 50% ($0.50 mid-market premium) using linear interpolation.
    """
    df_snap = df_snap.sort_values(by='Strike_Rung').reset_index(drop=True)
    df_snap['Mid_Price'] = (df_snap['Yes_Bid'] + df_snap['Yes_Ask']) / 2
    
    for i in range(len(df_snap) - 1):
        p_high = df_snap.loc[i, 'Mid_Price']
        p_low = df_snap.loc[i+1, 'Mid_Price']
        
        if p_high >= 0.50 >= p_low and p_high != p_low:
            s_low = df_snap.loc[i, 'Strike_Rung']
            s_high = df_snap.loc[i+1, 'Strike_Rung']
            
            consensus = s_low + (0.50 - p_high) * (s_high - s_low) / (p_low - p_high)
            return round(consensus, 4)
            
    if not df_snap.empty:
        idx_closest = (df_snap['Mid_Price'] - 0.50).abs().idxmin()
        return df_snap.loc[idx_closest, 'Strike_Rung']
    return None

def log_and_analyze_tomorrow():
    current_time = datetime.now()
    
    # Check command-line arguments for plotting overrides
    should_plot = "--plot" in sys.argv
    
    target_series = {
        'US': 'KXAAAGASD',
        'TX': 'KXAAAGASDTX',
        'NC': 'KXAAAGASDNC',
        'SC': 'KXAAAGASDSC',
        'GA': 'KXAAAGASDGA'
    }
    
    tomorrow_date = current_time + timedelta(days=1)
    
    date_str = current_time.strftime('%Y-%m-%d')        
    target_date_str = tomorrow_date.strftime('%Y-%m-%d')  
    time_str = current_time.strftime('%H:%M:%S')
    
    formatted_date = tomorrow_date.strftime('%y%b%d').upper()
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("obfiles", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    log_filepath = os.path.join("logs", f"system_log_{date_str}.csv")
    alert_log_path = os.path.join("logs", "radar_alerts_running.txt")
    
    market_snapshots = {}
    
    for market_name, series_prefix in target_series.items():
        event_ticker = f"{series_prefix}-{formatted_date}"
        url = f"https://external-api.kalshi.com/trade-api/v2/events/{event_ticker}"
        
        status_flag = "NOT_ACTIVE"
        total_volume = 0.0
        total_open_interest = 0.0
        markets_list = []
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                status_flag = "ACTIVE"
                event_data = response.json()
                markets_list = event_data.get('markets', [])
                
                for m in markets_list:
                    total_volume += float(m.get('volume_fp', 0) or 0)
                    total_open_interest += float(m.get('open_interest_fp', 0) or 0)
            elif response.status_code == 404:
                status_flag = "NOT_YET_OPEN"
        except Exception as e:
            status_flag = f"ERROR: {str(e)}"
            
        consensus_price = None
        
        if status_flag == "ACTIVE" and markets_list:
            snapshot_records = []
            
            for m in markets_list:
                market_ticker = m.get('ticker')
                strike_label = m.get('floor_strike') or m.get('cap_strike')
                if strike_label is None or not market_ticker:
                    continue
                
                yes_bid_val = float(m.get('yes_bid_dollars') or 0.0)
                yes_ask_val = float(m.get('yes_ask_dollars') or 1.0)
                total_bid_liquidity = 0.0
                total_ask_liquidity = 0.0
                
                ob_url = f"https://external-api.kalshi.com/trade-api/v2/market/{market_ticker}/orderbook"
                
                try:
                    ob_res = requests.get(ob_url, timeout=5)
                    if ob_res.status_code == 200:
                        ob_fp = ob_res.json().get('orderbook_fp', {})
                        
                        yes_entries = ob_fp.get('yes', [])
                        if yes_entries:
                            yes_bid_val = float(yes_entries[-1]) 
                            total_bid_liquidity = sum(float(entry) for entry in yes_entries)
                            
                        no_entries = ob_fp.get('no', [])
                        if no_entries:
                            best_no_bid = float(no_entries[-1]) 
                            yes_ask_val = round(1.0 - best_no_bid, 4) 
                            total_ask_liquidity = sum(float(entry) for entry in no_entries)
                except Exception:
                    pass
                
                if yes_bid_val == 0.0 and yes_ask_val == 1.0:
                    yes_bid_val = float(m.get('yes_bid_dollars') or 0.0)
                    yes_ask_val = float(m.get('yes_ask_dollars') or 1.0)
                
                snapshot_records.append({
                    'Timestamp': f"{date_str} {time_str}",
                    'Target_Market_Date': target_date_str,
                    'Strike_Rung': float(strike_label),
                    'Yes_Bid': yes_bid_val,
                    'Yes_Ask': yes_ask_val,
                    'Last_Traded_Price': float(m.get('last_price_dollars') or 0.0),
                    'Rung_Volume': float(m.get('volume_fp', 0) or 0.0),
                    'Rung_Open_Interest': float(m.get('open_interest_fp', 0) or 0.0),
                    'Bid_Depth_Units': total_bid_liquidity,
                    'Ask_Depth_Units': total_ask_liquidity
                })
                time.sleep(0.02)
                
            if snapshot_records:
                snap_df = pd.DataFrame(snapshot_records).sort_values(by='Strike_Rung')
                market_snapshots[market_name] = snap_df
                consensus_price = calculate_implied_consensus(snap_df)
                
                ob_filepath = os.path.join("obfiles", f"ob_{market_name}_{target_date_str}.csv")
                snap_df.to_csv(ob_filepath, mode='a', index=False, header=not os.path.exists(ob_filepath))
        
        log_record = {
            'Date': date_str, 'Time': time_str, 'Market_Key': market_name,
            'Target_Market_Date': target_date_str, 'Event_Ticker': event_ticker if status_flag == "ACTIVE" else "None",
            'Status': status_flag, 'Total_Volume': total_volume, 'Total_Open_Interest': total_open_interest,
            'Implied_Consensus': consensus_price if consensus_price else "None"
        }
        pd.DataFrame([log_record]).to_csv(log_filepath, mode='a', index=False, header=not os.path.exists(log_filepath))
        
        if market_name in market_snapshots:
            analyze_consensus_arbitrage(market_name, market_snapshots[market_name], consensus_price, alert_log_path, time_str)
            
            if should_plot:
                generate_visual_plots(market_name, target_date_str, time_str, market_snapshots[market_name], consensus_price)
                
    print(f"✅ [Run Finished at {time_str}] Plotting state: {should_plot}")

def analyze_consensus_arbitrage(market_name, df_snap, consensus_price, alert_log_path, time_str):
    if not consensus_price or consensus_price == "None":
        return
        
    df_snap = df_snap.sort_values(by='Strike_Rung').reset_index(drop=True)
    alert_lines = []
    
    for i, row in df_snap.iterrows():
        strike = row['Strike_Rung']
        yes_bid = row['Yes_Bid']
        yes_ask = row['Yes_Ask']
        spread = yes_ask - yes_bid
        
        if strike < consensus_price:
            safety_margin = consensus_price - strike
            if yes_ask < 0.70 and yes_ask > 0:
                alert_lines.append(f"[{time_str}] 🔥 UNDERPRICED YES: {market_name} Strike ${strike:.3f} is BELOW consensus by {safety_margin*100:.2f}¢. Ask: ${yes_ask:.2f} (Spread: ${spread:.2f})")
                      
        elif strike > consensus_price:
            safety_margin = strike - consensus_price
            synthetic_no_cost = round(1.00 - yes_bid, 2)
            if yes_bid > 0.30 and yes_bid < 1.0:
                alert_lines.append(f"[{time_str}] 💥 OVERPRICED FADE: {market_name} Strike ${strike:.3f} is ABOVE consensus by {safety_margin*100:.2f}¢. YES Bid: ${yes_bid:.2f}. Synthetic NO cost: ${synthetic_no_cost:.2f}")

        if i > 0:
            mid_today = (yes_bid + yes_ask) / 2
            mid_prev = (df_snap.loc[i-1, 'Yes_Bid'] + df_snap.loc[i-1, 'Yes_Ask']) / 2
            if mid_today > mid_prev and spread < 0.35:
                alert_lines.append(f"[{time_str}] ⚠️ BOOK INVERSION: {market_name} Strike ${strike:.3f} Midpoint (${mid_today:.2f}) is higher than lower Strike ${df_snap.loc[i-1, 'Strike_Rung']:.3f} (${mid_prev:.2f})")

    if alert_lines:
        with open(alert_log_path, "a") as f:
            for line in alert_lines:
                f.write(line + "\n")

def generate_visual_plots(market_name, target_date_str, time_str, df_snap, consensus_price):
    """
    Renders tomorrow's full option chain distributions without clipping,
    layers the consensus line, and uses a smart endpoint-agnostic parser
    to extract true state-level settled closing prices dynamically.
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(df_snap['Strike_Rung'], df_snap['Yes_Ask'], color='red', marker='o', linestyle='--', alpha=0.7, label='Market Ask (Yes)')
    plt.plot(df_snap['Strike_Rung'], df_snap['Yes_Bid'], color='green', marker='o', linestyle='--', alpha=0.7, label='Market Bid (Yes)')
    plt.fill_between(df_snap['Strike_Rung'], df_snap['Yes_Bid'], df_snap['Yes_Ask'], color='gray', alpha=0.12, label='Spread Depth')
    
    valid_last = df_snap[df_snap['Last_Traded_Price'] > 0]
    if not valid_last.empty:
        # FIXED: Changed the x-axis array reference to valid_last['Strike_Rung'] to match scatter sizing rules
        plt.scatter(valid_last['Strike_Rung'], valid_last['Last_Traded_Price'], color='blue', zorder=5, s=35, label='Last Price')
        
    if consensus_price:
        plt.axvline(x=consensus_price, color='blue', linestyle='--', linewidth=1.5, label=f"Implied Market Consensus (${consensus_price:.4f})")
        
    # --- ENDPOINT RESILIENT STATE SETTLEMENT ENGINE ---
    last_settled_price = None
    series_prefixes = {'US': 'KXAAAGASD', 'TX': 'KXAAAGASDTX', 'NC': 'KXAAAGASDNC', 'SC': 'KXAAAGASDSC', 'GA': 'KXAAAGASDGA'}
    prefix = series_prefixes.get(market_name, 'KXAAAGASD')
    
    today_ticker_date = datetime.now().strftime('%y%b%d').upper()
    # FIXED: Replaced yesterday_event_ticker with the correct active scope string today_event_ticker
    today_event_ticker = f"{prefix}-{today_ticker_date}"
    
    url = f"https://external-api.kalshi.com/trade-api/v2/events/{today_event_ticker}"
    
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            payload = res.json()
            markets_list = payload.get('markets', [])
            
            if isinstance(markets_list, list) and len(markets_list) > 0:
                # Direct check on first item inside plural list layout
                exp_val = markets_list[0].get('expiration_value')
                if exp_val and str(exp_val).strip():
                    last_settled_price = float(exp_val)
                    
            if last_settled_price is None and isinstance(markets_list, list):
                # Fallback scan block if top-level element is blank
                yes_strikes = []
                for m in markets_list:
                    if m.get('status') == 'finalized' and m.get('result') == 'yes':
                        strike = m.get('floor_strike') or m.get('cap_strike')
                        if strike is not None:
                            yes_strikes.append(float(strike))
                if yes_strikes:
                    last_settled_price = max(yes_strikes)
    except Exception:
        pass

    if last_settled_price:
        plt.axvline(x=last_settled_price, color='gray', linestyle='-', linewidth=1.5, label=f"Prev Day Settled Close (${last_settled_price:.4f})")
            
    plt.title(f"Complete Option Chain Curve: {market_name} (For Target: {target_date_str})")
    plt.xlabel("Contract Strike Rungs ($)")
    plt.ylabel("Probability Premium ($)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize='small')
    
    clean_time = time_str.replace(":", "-")
    plot_filename = os.path.join("plots", f"{market_name}_{target_date_str}_{clean_time}.png")
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=120)
    plt.close()

if __name__ == "__main__":
    log_and_analyze_tomorrow()
