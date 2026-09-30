import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error

def run_high_precision_delta_engine():
    print("🤖 Unifying multi-asset feature matrices from local storage...")
    
    # Reading from the active file written by your Kalshi script
    kalshi_path = os.path.join("data", "kalshi_gas_60day_history.csv")
    features_path = os.path.join("data", "market_predictor_features.csv")
    
    if not os.path.exists(kalshi_path) or not os.path.exists(features_path):
        print("❌ Error: Missing required files inside the data/ directory.")
        return
        
    df_kalshi = pd.read_csv(kalshi_path)
    df_features = pd.read_csv(features_path)
    
    df_kalshi['Date'] = pd.to_datetime(df_kalshi['Date'])
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    master_df = pd.merge(df_kalshi, df_features, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    print(f"Dataset unified successfully. Total aligned timeframe: {len(master_df)} days.")
    
    # ----------------------------------------------------
    # TRANSFORM FEATURE SET A: MACRO FINANCIAL PERCENT CHANGES
    # ----------------------------------------------------
    master_df['Gold_Pct_1d'] = master_df['Gold_Spot'].pct_change(1)
    master_df['Gold_Pct_3d'] = master_df['Gold_Spot'].pct_change(3)
    master_df['SP500_Pct_1d'] = master_df['SP500_Index'].pct_change(1)
    master_df['Refinery_Pct_1d'] = master_df['Refinery_Equity_Proxy'].pct_change(1)
    master_df['Refinery_Pct_3d'] = master_df['Refinery_Equity_Proxy'].pct_change(3)
    
    # ----------------------------------------------------
    # TRANSFORM FEATURE SET B: COMMODITY ABSOLUTE PRICE DELTAS
    # ----------------------------------------------------
    master_df['RBOB_Delta_1d'] = master_df['RBOB_Wholesale'] - master_df['RBOB_Wholesale'].shift(1)
    master_df['RBOB_Delta_2d'] = master_df['RBOB_Wholesale'] - master_df['RBOB_Wholesale'].shift(2)
    master_df['RBOB_Delta_5d'] = master_df['RBOB_Wholesale'] - master_df['RBOB_Wholesale'].shift(5)
    
    master_df['WTI_Delta_1d'] = master_df['WTI_Crude'] - master_df['WTI_Crude'].shift(1)
    master_df['WTI_Delta_5d'] = master_df['WTI_Crude'] - master_df['WTI_Crude'].shift(5)
    
    # ----------------------------------------------------
    # TRANSFORM FEATURE SET C: STATE-LEVEL PRICE DRIVERS & SPILLOVERS
    # ----------------------------------------------------
    target_markets = ['US_National', 'CA', 'NC', 'GA', 'SC', 'OR', 'TX', 'NY']
    for m in target_markets:
        master_df[f'{m}_Delta_1d'] = master_df[m] - master_df[m].shift(1)
        master_df[f'{m}_Delta_2d'] = master_df[m] - master_df[m].shift(2)
        
    # Inject one-hot encoded Day of Week variables
    master_df['Day_Name'] = master_df['Date'].dt.day_name()
    day_dummies = pd.get_dummies(master_df['Day_Name'], prefix='Is').astype(int)
    master_df = pd.concat([master_df, day_dummies], axis=1)
    
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        col_name = f'Is_{day}'
        if col_name not in master_df.columns:
            master_df[col_name] = 0
            
    # Clean out initial lookback shift rows
    master_df = master_df.dropna(subset=['RBOB_Delta_5d', 'Gold_Pct_3d']).reset_index(drop=True)
    
    regional_dependencies = {
        'US_National': ['NY_Delta_1d', 'TX_Delta_1d', 'CA_Delta_1d'],
        'CA': ['US_National_Delta_1d', 'OR_Delta_1d'],
        'NC': ['TX_Delta_1d', 'GA_Delta_1d', 'SC_Delta_1d', 'GA_Delta_2d'], 
        'GA': ['TX_Delta_1d', 'SC_Delta_1d', 'NC_Delta_1d'],
        'SC': ['TX_Delta_1d', 'GA_Delta_1d', 'NC_Delta_1d'],
        'OR': ['CA_Delta_1d', 'CA_Delta_2d'],                              
        'TX': ['US_National_Delta_1d', 'GA_Delta_1d'],
        'NY': ['US_National_Delta_1d', 'RBOB_Delta_1d']
    }
    
    tomorrow_predictions = {}
    
    print("\n--- Training Hyper-Precise Huber Delta Networks ---")
    
    for target in target_markets:
        base_features = [
            f'{target}_Delta_1d', f'{target}_Delta_2d',
            'RBOB_Delta_1d', 'RBOB_Delta_2d', 'RBOB_Delta_5d',
            'WTI_Delta_1d', 'WTI_Delta_5d',
            'Gold_Pct_1d', 'Gold_Pct_3d', 'SP500_Pct_1d', 'Refinery_Pct_1d', 'Refinery_Pct_3d',
            'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday', 'Is_Saturday', 'Is_Sunday'
        ]
        
        neighbor_features = regional_dependencies.get(target, [])
        feature_cols = base_features + neighbor_features
        
        # Target ($Y$): Next-day relative price change step
        # Create explicit target and tracking absolute value columns BEFORE dropping splits
        master_df['Tmp_Target_Price'] = master_df[target].shift(-1)
        master_df['Tmp_Target_Delta'] = master_df['Tmp_Target_Price'] - master_df[target]
        
        # Pull live row for tomorrow's forecast before clearing tracking NaNs
        live_predict_row = master_df.tail(1)
        working_df = master_df.dropna(subset=['Tmp_Target_Delta']).reset_index(drop=True)
        
        X = working_df[feature_cols]
        y = working_df['Tmp_Target_Delta']
        
        # Time-Series Evaluation Split (Last 10 days)
        split_idx = int(len(working_df) * 0.85)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit Huber regressor to minimize risk of erratic headline anomalies/outliers
        model = HuberRegressor(max_iter=3000, alpha=1.5)
        model.fit(X_train, y_train)
        
        # SAFE BACKTEST PERFORMANCE TRACKING (No positional indexing addition mismatches)
        test_deltas = model.predict(X_test)
        actual_prices = working_df['Tmp_Target_Price'].iloc[X_test.index].values
        base_prices = working_df[target].iloc[X_test.index].values
        predicted_prices = base_prices + test_deltas
        
        mae = mean_absolute_error(actual_prices, predicted_prices)
        
        # Generate tomorrow's actionable forecast targets
        X_live = live_predict_row[feature_cols]
        predicted_delta = float(model.predict(X_live))
        current_actual_price = float(live_predict_row[target].values)
        
        tomorrow_predictions[target] = {
            'Current_Price': current_actual_price,
            'Predicted_Delta': predicted_delta,
            'Tomorrow_Forecast': current_actual_price + predicted_delta,
            'MAE_Cents': mae * 100 
        }
        
    print("\n================== BACKTEST EVALUATION SUMMARY ==================")
    for market, metrics in tomorrow_predictions.items():
        print(f"📍 {market:12} | Model Average Error: {metrics['MAE_Cents']:.3f} cents per gallon")

    print("\n🔮 ============= TOMORROW'S SUB-CENT GAS PRICE FORECASTS =============")
    print(f"Generated on current system date: {datetime.now().strftime('%Y-%m-%d')}")
    
    predict_df = pd.DataFrame(tomorrow_predictions).T
    print(predict_df[['Current_Price', 'Predicted_Delta', 'Tomorrow_Forecast']].to_string(formatters={
        'Current_Price': '{:,.4f}'.format,
        'Predicted_Delta': '{:+.4f}'.format,
        'Tomorrow_Forecast': '{:,.4f}'.format
    }))
    print("====================================================================")

if __name__ == "__main__":
    run_high_precision_delta_engine()
