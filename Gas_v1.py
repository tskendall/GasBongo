import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STEP 1: GATHER EXTERNAL MARKET DATA
# ==========================================
def fetch_market_features(start_date, end_date):
    print("Fetching RBOB Gasoline and WTI Crude Oil data...")
    # RB=F is wholesale RBOB gasoline futures; CL=F is WTI Crude Oil futures
    tickers = {"RBOB": "RB=F", "WTI": "CL=F"}
    
    # Download data from Yahoo Finance
    market_data = yf.download(list(tickers.values()), start=start_date, end=end_date)["Close"]
    
    # Rename columns for clarity
    inv_tickers = {v: k for k, v in tickers.items()}
    market_data = market_data.rename(columns=inv_tickers)
    
    # Forward-fill weekend gaps (markets are closed, retail pumps are not)
    market_data = market_data.ffill().bfill()
    return market_data.reset_index()

# ==========================================
# STEP 2: ENGINEER FEATURE LAGS
# ==========================================
def engineer_lags(df):
    print("Engineering multi-day lag features...")
    # RBOB wholesale impacts the pump quickly (2 to 5 days)
    df['RBOB_lag_2'] = df['RBOB'].shift(2)
    df['RBOB_lag_3'] = df['RBOB'].shift(3)
    df['RBOB_lag_5'] = df['RBOB'].shift(5)
    
    # Crude oil ripples down over a longer window (5 to 10 days)
    df['WTI_lag_5'] = df['WTI'].shift(5)
    df['WTI_lag_7'] = df['WTI'].shift(7)
    df['WTI_lag_10'] = df['WTI'].shift(10)
    
    # Target Variable: Tomorrow's AAA Retail Price (Day T+1)
    df['Target_AAA_Tomorrow'] = df['Retail_AAA'].shift(-1)
    
    # Today's retail price is also a strong baseline feature
    df['Retail_AAA_Today'] = df['Retail_AAA']
    
    # Clean up edge rows caused by shifting
    return df.dropna().reset_index(drop=True)

# ==========================================
# STEP 3: PIPELINE EXECUTION & TRAINING
# ==========================================

# 1. Simulate pulling your target data (Replace this block with your actual AAA historical DataFrame)
dates = pd.date_range(start="2025-01-01", end="2026-06-01", freq='D')
mock_aaa = 3.10 + np.cumsum(np.random.normal(0, 0.02, len(dates))) # simulated state pump prices
df_aaa = pd.DataFrame({'Date': dates, 'Retail_AAA': mock_aaa})

# 2. Fetch the market features from yfinance
df_market = fetch_market_features(start_date="2025-01-01", end_date="2026-06-01")

# 3. Align datasets on the Date column
master_df = pd.merge(df_aaa, df_market, on='Date', how='inner')

# 4. Generate the lags
processed_df = engineer_lags(master_df)

# 5. Define Feature matrix (X) and Target vector (y)
feature_cols = ['Retail_AAA_Today', 'RBOB_lag_2', 'RBOB_lag_3', 'RBOB_lag_5', 'WTI_lag_5', 'WTI_lag_7', 'WTI_lag_10']
X = processed_df[feature_cols]
y = processed_df['Target_AAA_Tomorrow']

# 6. Time-Series Train/Test Split (Never shuffle time-series data!)
split_idx = int(len(processed_df) * 0.85)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTraining XGBoost Regressor on {len(X_train)} days, testing on {len(X_test)} days...")

# 7. Initialize and fit XGBoost
model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate Predictions
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n=== Model Performance ===")
print(f"Mean Absolute Error (MAE): ${mae:.4f} (Average margin of error per gallon)")
print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
print(f"R² Prediction Variance Score: {r2:.4f}")
