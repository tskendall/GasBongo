import os
import pandas as pd

df_k = pd.read_csv(os.path.join("data", "kalshi_gas_60day_history.csv"))
df_f = pd.read_csv(os.path.join("data", "market_predictor_features.csv"))

print("--- Kalshi File Head ---")
print(df_k.head(3))
print("--- Features File Head ---")
print(df_f.head(3))

# Check the merge result
df_k['Date'] = pd.to_datetime(df_k['Date'])
df_f['Date'] = pd.to_datetime(df_f['Date'])
master = pd.merge(df_k, df_f, on='Date', how='inner').sort_values('Date')

print("\n--- Merged Master Head ---")
print(master[['Date', 'US_National', 'RBOB_Wholesale', 'WTI_Crude']].head(5))
print("\n--- Merged Master Tail ---")
print(master[['Date', 'US_National', 'RBOB_Wholesale', 'WTI_Crude']].tail(5))
