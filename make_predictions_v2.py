import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def run_prediction_pipeline():
    print("🤖 Loading datasets from local storage...")
    
    kalshi_path = os.path.join("data", "kalshi_gas_60day_history.csv")
    features_path = os.path.join("data", "market_predictor_features.csv")
    
    df_kalshi = pd.read_csv(kalshi_path)
    df_features = pd.read_csv(features_path)
    
    df_kalshi['Date'] = pd.to_datetime(df_kalshi['Date'])
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    master_df = pd.merge(df_kalshi, df_features, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    
    target_markets = ['US_National', 'CA', 'NC', 'GA', 'SC', 'OR']
    tomorrow_predictions = {}
    
    print("\n--- Training Delta Regression Architecture ---")
    
    for market in target_markets:
        market_df = master_df[['Date', market, 'WTI_Crude', 'RBOB_Wholesale']].copy()
        
        # One-Hot Encode Days of the Week
        market_df['Day_Name'] = market_df['Date'].dt.day_name()
        day_dummies = pd.get_dummies(market_df['Day_Name'], prefix='Is').astype(int)
        market_df = pd.concat([market_df, day_dummies], axis=1)
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            col_name = f'Is_{day}'
            if col_name not in market_df.columns:
                market_df[col_name] = 0
                
        # TRANSFORMATION: Predict Delta (Tomorrow's Price minus Today's Price)
        market_df['Price_Today'] = market_df[market]
        market_df['Target_Delta_Tomorrow'] = market_df[market].shift(-1) - market_df['Price_Today']
        
        # Calculate shifts on features relative to today's wholesale environment
        market_df['RBOB_Delta_2d'] = market_df['RBOB_Wholesale'] - market_df['RBOB_Wholesale'].shift(2)
        market_df['RBOB_Delta_5d'] = market_df['RBOB_Wholesale'] - market_df['RBOB_Wholesale'].shift(5)
        market_df['WTI_Delta_5d'] = market_df['WTI_Crude'] - market_df['WTI_Crude'].shift(5)
        
        # Isolate the live prediction row before dropping edge rows
        live_predict_row = market_df.tail(1)
        market_df = market_df.dropna().reset_index(drop=True)
        
        feature_cols = ['RBOB_Delta_2d', 'RBOB_Delta_5d', 'WTI_Delta_5d', 
                        'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday', 'Is_Saturday', 'Is_Sunday']
        
        X = market_df[feature_cols]
        y = market_df['Target_Delta_Tomorrow']
        
        # Train/Test evaluation partition (Last 10 days)
        split_idx = int(len(market_df) * 0.85)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit Ridge Delta Model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        # Backtest Performance tracking
        test_deltas = model.predict(X_test)
        # Reconstruct absolute price predictions for the test split error calculation
        actual_prices = master_df[market].iloc[X_test.index + 1].values
        base_prices = master_df[market].iloc[X_test.index].values
        predicted_prices = base_prices + test_deltas
        
        mae = mean_absolute_error(actual_prices, predicted_prices)
        
        # Generate Tomorrow's absolute target projection
        X_live = live_predict_row[feature_cols]
        predicted_delta = float(model.predict(X_live)[0])
        current_actual_price = float(live_predict_row['Price_Today'].values[0])
        
        tomorrow_predictions[market] = {
            'Current_Price': round(current_actual_price, 4),
            'Predicted_Delta': round(predicted_delta, 4),
            'Tomorrow_Forecast': round(current_actual_price + predicted_delta, 4),
            'Backtest_MAE': round(mae, 4)
        }
        
    print("\n================== BACKTEST EVALUATION SUMMARY ==================")
    for market, metrics in tomorrow_predictions.items():
        print(f"📍 {market:12} | Backtest MAE: ${metrics['Backtest_MAE']:.4f} per gallon")

    print("\n🔮 ============= TOMORROW'S RETAIL GAS PRICE FORECASTS =============")
    print(f"Generated on current market snapshot: {datetime.now().strftime('%Y-%m-%d')}")
    predict_df = pd.DataFrame(tomorrow_predictions).T
    print(predict_df[['Current_Price', 'Predicted_Delta', 'Tomorrow_Forecast']].to_string())
    print("====================================================================")

if __name__ == "__main__":
    run_prediction_pipeline()
