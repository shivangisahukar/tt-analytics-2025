import pandas as pd
import numpy as np
import matplotlib
# MacOS Fix: Use the 'Agg' backend to save files without blocking the terminal
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

# Configuration
INPUT_FILE = "data/processed/master_long_dataset.csv"
OUTPUT_CSV = "data/processed/features_master.csv"
OUTPUT_DIR = "data/insights"

def run_advanced_feature_pipeline():
    # 1. Load Data and Ensure Directories Exist
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure your mapping pipeline ran correctly.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Force consistent data types to avoid matching errors
    df = pd.read_csv(INPUT_FILE)
    df['ttfi_id'] = df['ttfi_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    df['points_earned'] = pd.to_numeric(df['points_earned'], errors='coerce').fillna(0)
    df['season_year'] = pd.to_numeric(df['season_year'], errors='coerce')
    
    latest_year = 2024
    latest_df = df[df['season_year'] == latest_year].copy()
    
    if latest_df.empty:
        print(f"Error: No data found for the year {latest_year}.")
        return

    # Identify Top 50 based on seasonal aggregate
    player_totals = latest_df.groupby('ttfi_id').agg({
        'player_name': 'first',
        'total_seasonal_points': 'max'
    }).nlargest(50, 'total_seasonal_points')
    
    top_50_ids = player_totals.index.tolist()
    feature_results = []
    decay_factor = 0.8 

    for pid in top_50_ids:
        player_history = df[df['ttfi_id'] == pid]
        latest_season = player_history[player_history['season_year'] == latest_year]
        
        # FEATURE 1: Decay-Weighted Momentum
        yearly_sums = player_history.groupby('season_year')['points_earned'].sum()
        momentum = sum(pts * (decay_factor ** (latest_year - yr)) for yr, pts in yearly_sums.items())
        
        # FEATURE 2: Volatility (Consistency) 
        # Using ALL historical points (2020-2024) for statistical stability
        history_points = player_history['points_earned']
        volatility = history_points.std() if len(history_points) >= 2 else np.nan   
        
        # FEATURE 3: Weighted Pressure Score (SNR = 2x)
        pressure = sum(row['points_earned'] * (2.0 if 'Senior' in str(row['tournament_name']) else 1.0)
                       for _, row in latest_season.iterrows())
        
        feature_results.append({
            'ttfi_id': pid,
            'player_name': player_totals.loc[pid, 'player_name'],
            'institution': latest_season['state_institution'].iloc[0] if not latest_season.empty else "N/A",
            'momentum_score': round(momentum, 2),
            'volatility_index': round(volatility, 2),
            'pressure_score': pressure,
            'total_pts': player_totals.loc[pid, 'total_seasonal_points'] # <-- ADD THIS LINE
        })

    features_df = pd.DataFrame(feature_results)
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Feature matrix saved to {OUTPUT_CSV}")

    # --- BLOCKING-FREE PLOTTING ---
    
    # 1. Momentum Chart
    plt.figure(figsize=(12, 8))
    top_m = features_df.nlargest(10, 'momentum_score')
    y_pos = np.arange(len(top_m))
    bars = plt.barh(y_pos, top_m['momentum_score'], color='skyblue')
    plt.yticks(y_pos, top_m['player_name'])
    plt.bar_label(bars, padding=5, fmt='%.1f')
    plt.title('Top 10: Decay-Weighted Momentum', fontsize=14)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.3) 
    plt.savefig(os.path.join(OUTPUT_DIR, "momentum_score.png"))
    plt.close()

    # 2. Consistency Chart (Lower is Better)
    plt.figure(figsize=(14, 10)) 
    consistent_df = features_df.dropna(subset=['volatility_index']).nsmallest(10, 'volatility_index')

    if not consistent_df.empty:
        y_pos = np.arange(len(consistent_df))
        bars = plt.barh(y_pos, consistent_df['volatility_index'], color='#2ecc71', edgecolor='black')
        plt.yticks(y_pos, labels=consistent_df['player_name'], fontsize=11)
        plt.bar_label(bars, padding=10, fmt='%.2f', fontweight='bold')
        plt.title('Top 10: Most Consistent Players (5-Year History)', fontsize=16, pad=25)
        plt.xlabel('Volatility Index (Lower is More Consistent)')
        plt.gca().invert_yaxis() 
        plt.subplots_adjust(left=0.35) 
        plt.savefig(os.path.join(OUTPUT_DIR, "consistency_index.png"))
        print("SUCCESS: Consistency graph generated.")
    else:
        print("Warning: Insufficient historical data to calculate consistency.")
    plt.close()

    # 3. Pressure Score Chart
    plt.figure(figsize=(12, 8))
    top_p = features_df.nlargest(10, 'pressure_score')
    y_pos = np.arange(len(top_p))
    bars = plt.barh(y_pos, top_p['pressure_score'], color='lightgreen')
    plt.yticks(y_pos, top_p['player_name'])
    plt.bar_label(bars, padding=5, fmt='%.1f')
    plt.title('Top 10: Weighted Pressure Score (Senior Nationals Weighted 2x)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "pressure_score.png"))
    plt.close()

    # 4. Institutional Synergy Pie
    plt.figure(figsize=(8, 8))
    inst_counts = features_df['institution'].value_counts().head(5)
    plt.pie(inst_counts, labels=inst_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Institutional Synergy (Top 50 Representation)', fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "institutional_synergy.png"))
    plt.close()

    print(f"SUCCESS: 4 analysis images saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_advanced_feature_pipeline()