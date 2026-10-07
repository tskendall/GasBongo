import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_complete_50_state_matrix():
    print("🚀 Initializing Comprehensive 50-State + US Energy Matrix Scraper...")
    
    history_file = os.path.join("data", "kalshi_gas_all_states_history.csv")
    os.makedirs("data", exist_ok=True)
    
    # FIXED: Corrected TN prefix to KXAAAGASDTN and verified all other 50-state keys are uniquely mapped
    state_series = {
        'US': 'KXAAAGASD', 'AL': 'KXAAAGASDAL', 'AK': 'KXAAAGASDAK', 'AZ': 'KXAAAGASDAZ', 
        'AR': 'KXAAAGASDAR', 'CA': 'KXAAAGASDCA', 'CO': 'KXAAAGASDCO', 'CT': 'KXAAAGASDCT', 
        'DE': 'KXAAAGASDDE', 'FL': 'KXAAAGASDFL', 'GA': 'KXAAAGASDGA', 'HI': 'KXAAAGASDHI', 
        'ID': 'KXAAAGASDID', 'IL': 'KXAAAGASDIL', 'IN': 'KXAAAGASDIN', 'IA': 'KXAAAGASDIA', 
        'KS': 'KXAAAGASDKS', 'KY': 'KXAAAGASDKY', 'LA': 'KXAAAGASDLA', 'ME': 'KXAAAGASDME', 
        'MD': 'KXAAAGASDMD', 'MA': 'KXAAAGASDMA', 'MI': 'KXAAAGASDMI', 'MN': 'KXAAAGASDMN', 
        'MS': 'KXAAAGASDMS', 'MO': 'KXAAAGASDMO', 'MT': 'KXAAAGASDMT', 'NE': 'KXAAAGASDNE', 
        'NV': 'KXAAAGASDNV', 'NH': 'KXAAAGASDNH', 'NJ': 'KXAAAGASDNJ', 'NM': 'KXAAAGASDNM', 
        'NY': 'KXAAAGASDNY', 'NC': 'KXAAAGASDNC', 'ND': 'KXAAAGASDND', 'OH': 'KXAAAGASDOH', 
        'OK': 'KXAAAGASDOK', 'OR': 'KXAAAGASDOR', 'PA': 'KXAAAGASDPA', 'RI': 'KXAAAGASDRI', 
        'SC': 'KXAAAGASDSC', 'SD': 'KXAAAGASDSD', 'TN': 'KXAAAGASDTN', 'TX': 'KXAAAGASDTX', 
        'UT': 'KXAAAGASDUT', 'VT': 'KXAAAGASDVT', 'VA': 'KXAAAGASDVA', 'WA': 'KXAAAGASDWA', 
        'WV': 'KXAAAGASDWV', 'WI': 'KXAAAGASDWI', 'WY': 'KXAAAGASDWY'
    }
    
    columns_order = ['Date', 'US'] + sorted([k for k in state_series.keys() if k != 'US'])
    
    if os.path.exists(history_file):
        try:
            existing_df = pd.read_csv(history_file)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            last_recorded_date = existing_df['Date'].max()
            print(f"  -> Existing master file detected. Aligned up to: {last_recorded_date.strftime('%Y-%m-%d')}")
            start_date = last_recorded_date + timedelta(days=1)
        except Exception as e:
            print(f"  -> Error reading existing file, starting fresh: {e}")
            start_date = datetime.now() - timedelta(days=120)
            existing_df = pd.DataFrame(columns=columns_order)
    else:
        print("  -> No history file found. Building a brand-new 120-day matrix archive from scratch...")
        start_date = datetime.now() - timedelta(days=120)
        existing_df = pd.DataFrame(columns=columns_order)

    yesterday = datetime.now() - timedelta(days=1)
    if start_date.date() > yesterday.date():
        print("✅ The 51-market history database is completely up-to-date. No gaps to fill.")
        return

    missing_dates = pd.date_range(start=start_date, end=yesterday, freq='D')
    print(f"  -> Scraping and compiling {len(missing_dates)} missing timeline blocks for all 50 states.")
    
    new_records = []

    # 2. Iterate through missing dates to collect settled metrics
    for date_obj in missing_dates:
        date_str = date_obj.strftime('%Y-%m-%d')
        formatted_date = date_obj.strftime('%y%b%d').upper()
        
        row_data = {'Date': date_str}
        has_any_data = False
        
        print(f"  📥 Pulling market grid for calendar slot: {date_str}...")
        
        for state_key, series_prefix in state_series.items():
            event_ticker = f"{series_prefix}-{formatted_date}"
            url = f"https://external-api.kalshi.com/trade-api/v2/events/{event_ticker}"
            
            try:
                response = requests.get(url, timeout=4)
                if response.status_code == 200:
                    payload = response.json()
                    markets_list = payload.get('markets', [])
                    
                    if isinstance(markets_list, list) and len(markets_list) > 0:
                        first_market_rung = markets_list[0]
                        
                        if first_market_rung.get('status') == 'finalized':
                            raw_exp = first_market_rung.get('expiration_value')
                            if raw_exp and str(raw_exp).strip():
                                row_data[state_key] = float(raw_exp)
                                has_any_data = True
                                continue
                                
                row_data[state_key] = None
            except Exception:
                row_data[state_key] = None
                
        if has_any_data:
            new_records.append(row_data)
        time.sleep(0.05) 

    # 3. Concatenate and persist structural changes to disk cleanly
    if new_records:
        new_df = pd.DataFrame(new_records)
        
        # FIXED TYPE EXCEPTION SAFEGUARD: 
        # Convert new dates to datetime object, format to string, and force type matching
        new_df['Date'] = pd.to_datetime(new_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Ensure that every missing state column exists in the new dataframe slice
        for col in columns_order:
            if col not in new_df.columns:
                new_df[col] = None
                
        # Clean any index duplicate anomalies out of both sides prior to merging
        new_df = new_df.loc[:, ~new_df.columns.duplicated()]
        
        if not existing_df.empty:
            existing_df = existing_df.loc[:, ~existing_df.columns.duplicated()]
            existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.strftime('%Y-%m-%d')
            
        # Merge new rows with your existing database matrix cleanly using standard string indexes
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.sort_values(by='Date').drop_duplicates(subset=['Date'], keep='last')
        
        # Enforce strict uniform column organization arrangement
        final_df = final_df[columns_order]
        
        final_df.to_csv(history_file, index=False)
        print(f"\n💾 Success! Clean, unpolluted 51-market array updated inside: '{history_file}'")
        print(f"Added {len(new_records)} fresh daily rows. Database dimensions: {final_df.shape} shape layout.")
    else:
        print("  -> Requested dates haven't finalized on Kalshi yet. Master grid un-modified.")

if __name__ == "__main__":
    build_complete_50_state_matrix()
