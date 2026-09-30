import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_comprehensive_features():
    print("Initiating Macroeconomic & Financial Predictor Feature Extraction...")
    
    # Core Commodities + Macro Economic Gauges + Refining Margin Proxy
    tickers = {
        'RBOB_Wholesale': 'RB=F',
        'WTI_Crude': 'CL=F',
        'Gold_Spot': 'GC=F',
        'SP500_Index': '^GSPC',
        'Refinery_Equity_Proxy': 'OIH' # Captures unexpected weekly EIA inventory builds/draws instantly
    }
    
    # 250-day lookback buffer ensures full 140-day merge overlap after lagging features
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=250)
    
    print(f"Downloading historical market metrics from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        raw_data = yf.download(list(tickers.values()), start=start_date, end=end_date)["Close"]
        
        inv_tickers = {v: k for k, v in tickers.items()}
        market_df = raw_data.rename(columns=inv_tickers)
        
        # Reindex to force a daily calendar 7 days a week, filling stock market weekend gaps
        market_df.index = pd.to_datetime(market_df.index)
        full_date_range = pd.date_range(start=market_df.index.min(), end=market_df.index.max(), freq='D')
        final_df = market_df.reindex(full_date_range).ffill().bfill()
        
        final_df.index.name = 'Date'
        final_df = final_df.reset_index()
        final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
        
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, "market_predictor_features.csv")
        
        final_df.to_csv(output_filepath, index=False)
        print(f"✅ Success! Expanded predictor matrix saved to: '{output_filepath}'")
        print(final_df.tail(5))
        
    except Exception as e:
        print(f"❌ Failed to extract macro data from Yahoo Finance: {str(e)}")

if __name__ == "__main__":
    fetch_comprehensive_features()
