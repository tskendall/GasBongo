import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

def backfill_missing_history_forward():
    print("🔄 Running incremental forward-shifted history backfiller...")
    
    history_file = os.path.join("data", "kalshi_gas_60day_history.csv")
    target_series = {
        'US_National': 'KXAAAGASD',
        'CA': 'KXAAAGASDCA',
        'NC': 'KXAAAGASDNC',
        'GA': 'KXAAAGASDGA',
        'SC': 'KXAAAGASDSC',
        'OR': 'KXAAAGASDOR',
        'TX': 'KXAAAGASDTX',
        'NY': 'KXAAAGASDNY'
    }
    
    # 1. Establish the starting date baseline out of local files
    if os.path.exists(history_file):
        try:
            existing_df = pd.read_csv(history_file)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            last_recorded_date = existing_df['Date'].max()
            print(f"Existing file found. Last recorded entry: {last_recorded_date.strftime('%Y-%m-%d')}")
            start_date = last_recorded_date + timedelta(days=1)
        except Exception as e:
            print(f"Error reading existing file, starting fresh: {e}")
            start_date = datetime.now() - timedelta(days=140)
            existing_df = pd.DataFrame(columns=['Date'] + list(target_series.keys()))
    else:
        print("No history file detected. Initializing a fresh 140-day baseline...")
        start_date = datetime.now() - timedelta(days=140)
        existing_df = pd.DataFrame(columns=['Date'] + list(target_series.keys()))

    # CHANGED: Extend the target range ceiling calculation one full day forward
    target_ceiling = datetime.now()
    if start_date > target_ceiling:
        print("✅ History file is completely up-to-date and forward-aligned. No gaps to fill.")
        return

    # Generate explicit date lists to fill the missing gaps
    missing_dates = pd.date_range(start=start_date, end=target_ceiling, freq='D')
    print(f"Identified {len(missing_dates)} timeline slots to evaluate and recover.")
    
    new_records = []

    # 2. Iterate through missing dates to collect settled metrics
    for date_obj in missing_dates:
        date_str = date_obj.strftime('%Y-%m-%d')
        formatted_date = date_obj.strftime('%y%b%d').upper()
        row_data = {'Date': date_str}
        
        has_data_for_day = False
        print(f"  Pinging Kalshi for forward index slot: {date_str}...")
        
        for market_name, series_prefix in target_series.items():
            event_ticker = f"{series_prefix}-{formatted_date}"
            url = f"https://external-api.kalshi.com/trade-api/v2/events/{event_ticker}"
            
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    markets = response.json().get('markets', [])
                    if markets:
                        # Direct lookup into Kalshi's raw structural value key
                        exp_value = markets.get('expiration_value') if isinstance(markets, dict) else markets.get('expiration_value')
                        if exp_value and exp_value.strip():
                            row_data[market_name] = float(exp_value)
                            has_data_for_day = True
                time.sleep(0.03) # Polite rate spacing
            except Exception:
                continue
                
        if has_data_for_day:
            new_records.append(row_data)

    # 3. Append safely without breaking old historical logs
    if new_records:
        new_df = pd.DataFrame(new_records)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # Merge new rows with the historical database cleanly
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.sort_values(by='Date').drop_duplicates(subset=['Date'], keep='last')
        
        # Keep tracking strings cleanly aligned
        final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
        
        final_df.to_csv(history_file, index=False)
        print(f"💾 Success! Append complete. Added {len(new_records)} forward rows to '{history_file}'")
    else:
        print("Target horizon slots haven't finalized on Kalshi's grids yet. Exiting cleanly.")

if __name__ == "__main__":
    backfill_missing_history_forward()
