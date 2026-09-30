import pandas as pd
import requests

def discover_by_event_search():
    print("Scanning Kalshi Global Events Directory...")
    
    # We query the primary events endpoint instead of series to see active market dates
    url = "https://external-api.kalshi.com/trade-api/v2/markets"
    params = {
        'limit': 200  # Pull a wide batch of active events
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"Failed API response. Status: {response.status_code}")
            return
            
        events = response.json().get('events', [])
        print(f"Total events retrieved for scanning: {len(events)}")
        
        matches = []
        for e in events:
            title = e.get('title', '').lower()
            ticker = e.get('event_ticker', '')
            
            # Text check for either gas or gasoline indicators
            if 'gas' in title or 'gasoline' in title:
                matches.append({
                    'Event_Ticker': ticker,
                    'Event_Title': e.get('title'),
                    'Series_Ticker': e.get('series_ticker')
                })
                
        df = pd.DataFrame(matches)
        if df.empty:
            print("\n⚠️ No gas markets are actively open in this specific 200-event window.")
            print("Let's query the specific known series block directly.")
            query_known_gas_block()
        else:
            print("\n=== Discovered Kalshi Gas Event Keys ===")
            print(df.to_string(index=False))
            
    except Exception as err:
        print(f"Connection Error: {str(err)}")

def query_known_gas_block():
    # Direct check against the benchmark US National Gas Series code
    url = "https://external-api.kalshi.com/trade-api/v2/markets"
    params = {'series_ticker': 'KXAAAGASD', 'limit': 10}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        markets = response.json().get('markets', [])
        if markets:
            print(f"Found active markets for ticker KXAAAGASD! Example ticker: {markets[0].get('ticker')}")
        else:
            print("Ticker block returned empty. Markets may be cycling right now.")

if __name__ == "__main__":
    discover_by_event_search()
