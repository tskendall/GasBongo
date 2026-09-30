import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def run_prediction_pipeline():
    print("🤖 Loading datasets from local storage...")
    
    kalshi_path = os.path.join("data", "kalshi_gas_60day_history.csv")
    features_path = os.path.join("data", "market_predictor_features.csv")
    
    if not os.path.exists(kalshi_path) or not os.path.exists(features_path):
        print("❌ Error: Missing files in data/. Run collection scripts first.")
        return
        
    df_kalshi = pd.read_csv(kalshi_path)
    df_features = pd.read_csv(features_path)
    
    df_kalshi['Date'] = pd.to_datetime(df_kalshi['Date'])
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    # Merge datasets
    master_df = pd.merge(df_kalshi, df_features, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    print(f"Dataset successfully unified. Total aligned timeframe: {len(master_df)} days.")
    
    target_markets = ['US_National', 'CA', 'NC', 'GA', 'SC', 'OR']
    overall_results = {}
    tomorrow_predictions = {}
    
    print("\n--- Starting Feature Engineering & Model Training ---")
    
    for market in target_markets:
        # Isolate baseline columns + we need the 'Date' to extract Day of the Week
        market_df = master_df[['Date', market, 'WTI_Crude', 'RBOB_Wholesale']].copy()
        
        # 1. Day of the Week Feature Engineering (One-Hot Encoding)
        # .day_name() gives 'Monday', 'Tuesday', etc.
        market_df['Day_Name'] = market_df['Date'].dt.day_name()
        
        # Create binary columns for each day (0 or 1)
        day_dummies = pd.get_dummies(market_df['Day_Name'], prefix='Is').astype(int)
        market_df = pd.concat([market_df, day_dummies], axis=1)
        
        # Ensure all 7 days columns exist even if lookback is short
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            col_name = f'Is_{day}'
            if col_name not in market_df.columns:
                market_df[col_name] = 0
        
        # 2. Commodity Price Lags
        market_df['RBOB_lag_2'] = market_df['RBOB_Wholesale'].shift(2)
        market_df['RBOB_lag_3'] = market_df['RBOB_Wholesale'].shift(3)
        market_df['RBOB_lag_5'] = market_df['RBOB_Wholesale'].shift(5)
        market_df['WTI_lag_5'] = market_df['WTI_Crude'].shift(5)
        
        # 3. Retail Price Lags
        market_df['Price_Today'] = market_df[market]
        market_df['Price_Lag_1'] = market_df[market].shift(1)
        
        # Target: Tomorrow's Retail Price (Day T+1)
        market_df['Target_Tomorrow'] = market_df[market].shift(-1)
        
        # Extract the live row for tomorrow's prediction before dropping NaNs
        live_predict_row = market_df.tail(1)
        market_df = market_df.dropna().reset_index(drop=True)
        
        # Define exact columns going into the model matrices
        base_features = ['Price_Today', 'Price_Lag_1', 'RBOB_lag_2', 'RBOB_lag_3', 'RBOB_lag_5', 'WTI_lag_5']
        day_features = ['Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday', 'Is_Saturday', 'Is_Sunday']
        feature_cols = base_features + day_features
        
        X = market_df[feature_cols]
        y = market_df['Target_Tomorrow']
        
        # Time-Series Split (85% Train, 15% Test)
        split_idx = int(len(market_df) * 0.85)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Regularized model definitions
        models = {
            'Ridge_Regression': Ridge(alpha=1.0),
            'Random_Forest': RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
            'Regularized_XGBoost': XGBRegressor(
                n_estimators=50, 
                learning_rate=0.03, 
                max_depth=3, 
                reg_alpha=0.1,    
                reg_lambda=1.0,   
                random_state=42
            )
        }
        
        market_results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            market_results[name] = {'MAE': round(mae, 4), 'R2': round(r2, 4)}
            
            # Formulate next-day prediction targets
            X_live = live_predict_row[feature_cols]
            tomorrow_preds = model.predict(X_live)
            
            if market not in tomorrow_predictions:
                tomorrow_predictions[market] = {}
            tomorrow_predictions[market][name] = round(float(tomorrow_preds), 4)
            
        overall_results[market] = market_results

    print("\n================== BACKTEST EVALUATION SUMMARY ==================")
    for market, metrics in overall_results.items():
        print(f"\n📍 Market Area: {market}")
        for model_name, score in metrics.items():
            print(f"  -> {model_name:20} | MAE: ${score['MAE']:.4f} | R² Score: {score['R2']:.4f}")

    print("\n🔮 ============= TOMORROW'S RETAIL GAS PRICE FORECASTS =============")
    print(f"Generated on current market data snapshot: {datetime.now().strftime('%Y-%m-%d')}")
    
    predict_df = pd.DataFrame(tomorrow_predictions).T
    print(predict_df.to_string())
    print("====================================================================")

if __name__ == "__main__":
    run_prediction_pipeline()
