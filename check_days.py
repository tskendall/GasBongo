import os
import sys
import pandas as pd

def run_weekday_market_audit():
    history_file = os.path.join("data", "kalshi_gas_60day_history.csv")
    
    if not os.path.exists(history_file):
        print(f"❌ Error: Missing core tracking database file at '{history_file}'")
        return
        
    df = pd.read_csv(history_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if 'US_National' in df.columns:
        df = df.rename(columns={'US_National': 'US'})

    # ----------------------------------------------------
    # FUZZY DAY-OF-WEEK NAME MAPPING DICTIONARY
    # ----------------------------------------------------
    day_mapping = {
        'mon': 'Monday', 'monday': 'Monday',
        'tue': 'Tuesday', 'tues': 'Tuesday', 'tuesday': 'Tuesday',
        'wed': 'Wednesday', 'weds': 'Wednesday', 'wednesday': 'Wednesday',
        'thu': 'Thursday', 'thur': 'Thursday', 'thurs': 'Thursday', 'thursday': 'Thursday',
        'fri': 'Friday', 'friday': 'Friday',
        'sat': 'Saturday', 'satday': 'Saturday', 'saturday': 'Saturday',
        'sun': 'Sunday', 'sunday': 'Sunday'
    }

    # Read and normalize weekday argument
    raw_day_arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else 'tuesday'
    target_day = day_mapping.get(raw_day_arg)
    
    if not target_day:
        print(f"❌ Error: '{raw_day_arg}' is not a recognized weekday string shortcut.")
        return

    # ----------------------------------------------------
    # MULTI-MARKET COMMA SEPARATED PARSING
    # ----------------------------------------------------
    raw_market_arg = sys.argv[2].strip() if len(sys.argv) > 2 else 'NC'
    # Split the string by commas, capitalize, and strip out whitespace
    target_markets = [m.strip().upper() for m in raw_market_arg.split(',') if m.strip()]
    
    valid_columns = [col for col in df.columns if col != 'Date']
    
    # Validate each requested market column against what is inside the CSV file
    for m in target_markets:
        if m not in valid_columns:
            print(f"❌ Error: Market '{m}' is missing from the history file columns. Available: {valid_columns}")
            return

    # Add day name string labels to the dataframe
    df['Day_of_Week'] = df['Date'].dt.day_name()
    
    # Compute relative delta changes (cents) for EVERY requested focus market
    for m in target_markets:
        df[f'{m}_Delta'] = df[m].diff() * 100
        
    # ----------------------------------------------------
    # GENERAL ARTIFACT FILTER CLEANER
    # ----------------------------------------------------
    # Exclude rows where the first focus market is completely stagnant (delta == 0.0)
    # to eliminate the summer merge placeholders on your upcoming historical scrape
    primary_m = target_markets[0]
    cleaned_df = df[df[f'{primary_m}_Delta'] != 0.0].copy()
    
    # Isolate strictly to your targeted weekday name slice
    filtered_log = cleaned_df[cleaned_df['Day_of_Week'] == target_day].copy()

    print(f"\n================== HISTORICAL {target_day.upper()} SURVEY GRIDS ==================")
    print(f"Auditing tracking array group: {target_markets}")
    print(f"Total unpolluted {target_day} entries found: {len(filtered_log)}")
    
    if not filtered_log.empty:
        # Dynamically build the side-by-side display columns
        display_columns = ['Date']
        
        # We interlock [Market_Delta, Market_Price] continuously across the output grid row
        for m in target_markets:
            display_columns.append(f'{m}_Delta')
            display_columns.append(m)
            
        print("\n" + filtered_log[display_columns].to_string(index=False))
        
        # ----------------------------------------------------
        # EXPORT CONSOLE RISK SUMMARIES FOR EACH STATE
        # ----------------------------------------------------
        print(f"\n📈 Real {target_day} Volatility Breakdowns:")
        for m in target_markets:
            m_deltas = filtered_log[f'{m}_Delta']
            avg_move = m_deltas.mean()
            max_spike = m_deltas.max()
            max_drop = m_deltas.min()
            
            print(f"  -> [{m}] Avg Shift: {avg_move:+.3f}¢ | Max Spike: {max_spike:+.2f}¢ | Max Drop: {max_drop:+.2f}¢")
    else:
        print(f"No active, unpolluted {target_day} data points discovered for this group configuration.")
    print("==================================================================================")

if __name__ == "__main__":
    run_weekday_market_audit()
