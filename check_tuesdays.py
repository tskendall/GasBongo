import os
import pandas as pd

history_file = os.path.join("data", "kalshi_gas_60day_history.csv")
df = pd.read_csv(history_file)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Standardise column naming rules safely
if 'US_National' in df.columns:
    df = df.rename(columns={'US_National': 'US'})

# Add day name strings and calculate daily deltas (Day T - Day T-1)
df['Day_of_Week'] = df['Date'].dt.day_name()
df['NC_Delta'] = df['NC'].diff() * 100 # Converted to cents

# Isolate ALL Tuesdays chronologically
all_tuesdays = df[df['Day_of_Week'] == 'Tuesday'].copy()

print("================== NORTH CAROLINA: FULL HISTORICAL TUESDAYS LOG ==================")
if not all_tuesdays.empty:
    print(all_tuesdays[['Date', 'NC_Delta', 'NC', 'GA', 'TX', 'US']].to_string(index=False))
else:
    print("No Tuesday entries found in the history file.")
print("==================================================================================")
