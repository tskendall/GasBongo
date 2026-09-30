import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_kalshi_140day_multistate_matrix():
    print("Initiating expanded 140-day (20-week) multi-state tracking pipeline...")
    
    # Corrected full collection mappings based on live Kalshi series nomenclature
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
    
    # Enforce strict data/ folder tracking
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "kalshi_gas_60day_history.csv") # Kept same name for your tracking scripts
    
    # Construct exact timelines going back 140 days, starting at 0 to capture yesterday/today flawlessly
    today = datetime.now()
    date_list = [today - timedelta(days=i) for i in range(0, 140)]
    
    all_extracted_records = []
    
    for market_name, series_prefix in target_series.items():
        print(f"\nExtracting historical timeline for column: {market_name}...")
        
        for date_obj in date_list:
            # Format ticker using Kalshi's strict uppercase YYMMMDD format (e.g., 26SEP28)
            formatted_date = date_obj.strftime('%y%b%d').upper()
            event_ticker = f"{series_prefix}-{formatted_date}"
            
            url = f"https://external-api.kalshi.com/trade-api/v2/events/{event_ticker}"
            
            try:
                # Optimized single event lookup
                response = requests.get(url, timeout=5)
                
                if response.status_code == 404:
                    continue  # Gracefully pass structural reporting skips or holidays
                elif response.status_code != 200:
                    continue
                    
                markets = response.json().get('markets', [])
                if not markets:
                    continue
                    
                # Dig straight into the first rung object to capture Kalshi's raw settlement entry
                # This bypasses the need to check conditional outcome rungs manually
                exp_value = markets[0].get('expiration_value')
                
                if exp_value and exp_value.strip():
                    try:
                        actual_price = float(exp_value)
                        all_extracted_records.append({
                            'Date': date_obj.strftime('%Y-%m-%d'),
                            'Market': market_name,
                            'Price': actual_price
                        })
                    except ValueError:
                        continue
                        
                # 50ms sleep to stay completely underneath Kalshi's public rate limiter thresholds
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  Network pause on {event_ticker}: {str(e)}")
                time.sleep(0.5)
                continue

    if not all_extracted_records:
        print("\n❌ Error: Failed to gather any matching metrics. Double check network availability.")
        return
        
    # Transform raw data records into your matrix format
    raw_df = pd.DataFrame(all_extracted_records)
    processed_df = raw_df.groupby(['Date', 'Market'])['Price'].max().reset_index()
    
    print("\nPivoting data structure into row/column layout...")
    final_matrix = processed_df.pivot(index='Date', columns='Market', values='Price')
    
    # Linearly fill weekend or structural lag reporting delays
    final_matrix = final_matrix.interpolate(method='linear').ffill().bfill()
    
    # Save the structured file straight into your data/ folder
    final_matrix.to_csv(output_filepath)
    print(f"\n✅ Success! Expanded 140-day dataset written to: '{output_filepath}'")
    print(final_matrix.head(5))  # Verify the oldest historical points
    print(final_matrix.tail(5))  # Verify yesterday's integration points

if __name__ == "__main__":
    fetch_kalshi_140day_multistate_matrix()
