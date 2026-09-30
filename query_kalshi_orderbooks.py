import os
import json
import requests
import pandas as pd
from datetime import datetime

def log_intraday_orderbooks():
    current_time = datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')
    time_str = current_time.strftime('%H:%M:%S')
    
    # Target configurations (Suffixes map directly to Kalshi root tracking tokens)
    target_series = {
        'US_National': 'KXAAAGASD',
        'TX': 'KXAAAGASDTX',
        'NC': 'KXAAAGASDNC'
    }
    
    # Kalshi internal date format generation for Today's explicit ticker (e.g., 26SEP29)
    formatted_date = current_time.strftime('%y%b%d').upper()
    
    # Establish strict destination folder routing
    os.makedirs("logs", exist_ok=True)
    os.makedirs("obfiles", exist_ok=True)
    
    log_filepath = os.path.join("logs", f"system_log_{date_str}.csv")
    
    # Loop over tracked regions
    for market_name, series_prefix in target_series.items():
        event_ticker = f"{series_prefix}-{formatted_date}"
        url = f"https://external-api.kalshi.com/trade-api/v2/events/{event_ticker}"
        
        # Default initialization metrics for our tracking rows
        status_flag = "NOT_ACTIVE"
        total_volume = 0.0
        total_open_interest = 0.0
        markets_list = []
        
        try:
            # Execute direct lightweight single event inquiry
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                status_flag = "ACTIVE"
                event_data = response.json()
                markets_list = event_data.get('markets', [])
                
                # Accumulate structural macro interest figures across all rungs
                for m in markets_list:
                    total_volume += float(m.get('volume_fp', 0) or 0)
                    total_open_interest += float(m.get('open_interest_fp', 0) or 0)
            elif response.status_code == 404:
                # Target event doesn't exist yet on Kalshi's active production grid
                status_flag = "NOT_FOUND"
                
        except Exception as e:
            status_flag = f"ERROR: {str(e)}"
            
        # ----------------------------------------------------
        # TRANSACTION 1: WRITE TO DAILY STATUS LOG
        # ----------------------------------------------------
        log_record = {
            'Date': date_str,
            'Time': time_str,
            'Market_Key': market_name,
            'Event_Ticker': event_ticker if status_flag == "ACTIVE" else "None",
            'Status': status_flag,
            'Total_Volume': total_volume,
            'Total_Open_Interest': total_open_interest
        }
        
        log_df = pd.DataFrame([log_record])
        # Append cleanly to daily file, create header only if file is fresh
        header_needed = not os.path.exists(log_filepath)
        log_df.to_csv(log_filepath, mode='a', index=False, header=header_needed)
        
        # ----------------------------------------------------
        # TRANSACTION 2: SNAPSHOT THE ORDER BOOKS (IF ACTIVE)
        # ----------------------------------------------------
        if status_flag == "ACTIVE" and markets_list:
            snapshot_records = []
            
            for m in markets_list:
                strike_label = m.get('floor_strike') or m.get('cap_strike')
                if strike_label is None:
                    continue
                    
                snapshot_records.append({
                    'Timestamp': f"{date_str} {time_str}",
                    'Strike_Rung': float(strike_label),
                    'Yes_Bid': float(m.get('yes_bid_dollars', 0) or 0),
                    'Yes_Ask': float(m.get('yes_ask_dollars', 0) or 0),
                    # Bonus Critical Metrics for EV Modeling:
                    'Last_Traded_Price': float(m.get('last_price_dollars', 0) or 0),
                    'Rung_Volume': float(m.get('volume_fp', 0) or 0),
                    'Rung_Open_Interest': float(m.get('open_interest_fp', 0) or 0)
                })
                
            if snapshot_records:
                snap_df = pd.DataFrame(snapshot_records).sort_values(by='Strike_Rung')
                # Save into an isolated file for this specific market and date
                ob_filepath = os.path.join("obfiles", f"ob_{market_name}_{date_str}.csv")
                ob_header_needed = not os.path.exists(ob_filepath)
                snap_df.to_csv(ob_filepath, mode='a', index=False, header=ob_header_needed)
                
    print(f"⏰ [Snapshot Captured at {time_str}] Logs appended. Order books synced for active nodes.")

if __name__ == "__main__":
    log_intraday_orderbooks()
