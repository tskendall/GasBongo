import os
import pandas as pd
from sklearn.linear_model import HuberRegressor

def analyze_model_weights():
    print("🤖 Extracting model coefficients and structural feature weights...")
    
    kalshi_path = os.path.join("data", "kalshi_gas_60day_history.csv")
    features_path = os.path.join("data", "market_predictor_features.csv")
    
    df_kalshi = pd.read_csv(kalshi_path)
    df_features = pd.read_csv(features_path)
    
    df_kalshi['Date'] = pd.to_datetime(df_kalshi['Date'])
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    master_df = pd.merge(df_kalshi, df_features, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    
    # Mirror the precise transformation pipeline from your main engine
    master_df['Gold_Pct_1d'] = master_df['Gold_Spot'].pct_change(1)
    master_df['Gold_Pct_3d'] = master_df['Gold_Spot'].pct_change(3)
    master_df['SP500_Pct_1d'] = master_df['SP500_Index'].pct_change(1)
    master_df['Refinery_Pct_1d'] = master_df['Refinery_Equity_Proxy'].pct_change(1)
    master_df['Refinery_Pct_3d'] = master_df['Refinery_Equity_Proxy'].pct_change(3)
    
    master_df['RBOB_Delta_1d'] = master_df['RBOB_Wholesale'] - master_df['RBOB_Wholesale'].shift(1)
    master_df['RBOB_Delta_2d'] = master_df['RBOB_Wholesale'] - master_df['RBOB_Wholesale'].shift(2)
    master_df['RBOB_Delta_5d'] = master_df['RBOB_Wholesale'] - master_df['RBOB_Wholesale'].shift(5)
    master_df['WTI_Delta_1d'] = master_df['WTI_Crude'] - master_df['WTI_Crude'].shift(1)
    master_df['WTI_Delta_5d'] = master_df['WTI_Crude'] - master_df['WTI_Crude'].shift(5)
    
    target_markets = ['US_National', 'CA', 'NC', 'GA', 'SC', 'OR', 'TX', 'NY']
    for m in target_markets:
        master_df[f'{m}_Delta_1d'] = master_df[m] - master_df[m].shift(1)
        master_df[f'{m}_Delta_2d'] = master_df[m] - master_df[m].shift(2)
        
    master_df['Day_Name'] = master_df['Date'].dt.day_name()
    day_dummies = pd.get_dummies(master_df['Day_Name'], prefix='Is').astype(int)
    master_df = pd.concat([master_df, day_dummies], axis=1)
    
    master_df = master_df.dropna(subset=['RBOB_Delta_5d', 'Gold_Pct_3d']).reset_index(drop=True)
    
    # Focus analysis on US National and your home market (NC)
    focus_targets = ['US_National', 'NC']
    regional_dependencies = {
        'US_National': ['NY_Delta_1d', 'TX_Delta_1d', 'CA_Delta_1d'],
        'NC': ['TX_Delta_1d', 'GA_Delta_1d', 'SC_Delta_1d', 'GA_Delta_2d']
    }
    
    for target in focus_targets:
        base_features = [
            f'{target}_Delta_1d', f'{target}_Delta_2d',
            'RBOB_Delta_1d', 'RBOB_Delta_2d', 'RBOB_Delta_5d',
            'WTI_Delta_1d', 'WTI_Delta_5d',
            'Gold_Pct_1d', 'Gold_Pct_3d', 'SP500_Pct_1d', 'Refinery_Pct_1d', 'Refinery_Pct_3d',
            'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday', 'Is_Saturday', 'Is_Sunday'
        ]
        
        feature_cols = base_features + regional_dependencies[target]
        
        y_column = f'{target}_Target_Delta'
        master_df[y_column] = master_df[target].shift(-1) - master_df[target]
        working_df = master_df.dropna(subset=[y_column]).reset_index(drop=True)
        
        model = HuberRegressor(max_iter=3000, alpha=1.5)
        model.fit(working_df[feature_cols], working_df[y_column])
        
        # Format weights into a clean human-readable dataframe
        # Multiplying coefficients by 100 transforms the weight into "Cents of pump movement per 1 unit of feature change"
        importance_df = pd.DataFrame({
            'Predictor_Feature': feature_cols,
            'Weight_Impact_Cents': model.coef_ * 100
        })
        
        # Sort by absolute impact power
        importance_df['Abs_Impact'] = importance_df['Weight_Impact_Cents'].abs()
        importance_df = importance_df.sort_values(by='Abs_Impact', ascending=False).drop(columns=['Abs_Impact'])
        
        print(f"\n================ RANKED DRIVERS FOR {target.upper()} ================")
        print(f"Intercept (Baseline daily drift): {model.intercept_ * 100:+.4f} cents")
        print(importance_df.to_string(index=False, formatters={'Weight_Impact_Cents': '{:+.4f}¢'.format}))

if __name__ == "__main__":
    analyze_model_weights()
