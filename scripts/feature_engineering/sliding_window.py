import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data/processed/master_long_dataset.csv"
OUTPUT_FILE = "data/processed/supervised_timeseries_data.csv"

def create_advanced_sliding_window():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run mapping script first!")
        return

    # Load and clean data
    df = pd.read_csv(INPUT_FILE)
    df['season_year'] = pd.to_numeric(df['season_year'])
    df['points_earned'] = pd.to_numeric(df['points_earned'], errors='coerce').fillna(0)

    # Calculate Career Volatility (2020-2024)
    career_stats = df.groupby('ttfi_id')['points_earned'].std().reset_index()
    career_stats.columns = ['ttfi_id', 'career_volatility']

    # Annual Aggregation for the Sliding Window
    annual_df = df.groupby(['ttfi_id', 'player_name', 'season_year']).agg({
        'total_seasonal_points': 'max',
        'state_institution': 'first'
    }).reset_index()

    annual_df = annual_df.sort_values(['ttfi_id', 'season_year'])

    # Merge career consistency into the annual timeline
    annual_df = annual_df.merge(career_stats, on='ttfi_id', how='left')

    # Create Time-Series Lags (T-1, T-2, T-3)
    annual_df['pts_lag_1'] = annual_df.groupby('ttfi_id')['total_seasonal_points'].shift(1)
    annual_df['pts_lag_2'] = annual_df.groupby('ttfi_id')['total_seasonal_points'].shift(2)
    annual_df['pts_lag_3'] = annual_df.groupby('ttfi_id')['total_seasonal_points'].shift(3)

    # Momentum: YoY Growth
    annual_df['momentum_yoy'] = annual_df['pts_lag_1'] - annual_df['pts_lag_2']

    # Fill missing volatility with the global mean
    avg_vol = annual_df['career_volatility'].mean()
    annual_df['career_volatility'] = annual_df['career_volatility'].fillna(avg_vol)
    
    # Drop rows without enough history to establish a 2026 trend
    supervised_df = annual_df.dropna(subset=['pts_lag_3'])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    supervised_df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Supervised dataset with Career Volatility created ({len(supervised_df)} rows).")

if __name__ == "__main__":
    create_advanced_sliding_window()