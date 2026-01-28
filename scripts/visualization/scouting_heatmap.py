'''
How to Read Your New Heatmap

The Growth Spike (Rank Jump): * Dark Green: This player is expected to "leap" multiple ranks by 2026. These are your breakout candidates.

Yellow/Red: This player is expected to maintain their rank or drop slightly.

The Consistency Filter (Career Volatility): * Dark Red: This player is erratic. Their high predicted rank might be based on a few massive wins, making them "High Risk."

Light Green/Yellow: This player has low volatility. You can trust that their 2026 rank is backed by years of stable, professional performance.'''

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_FILE = "data/processed/supervised_timeseries_data.csv"
OUTPUT_DIR = "data/insights"

def generate_scouting_heatmap():
    if not os.path.exists(INPUT_FILE):
        print("Error: supervised_timeseries_data.csv not found.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    features = ['pts_lag_1', 'pts_lag_2', 'pts_lag_3', 'momentum_yoy', 'career_volatility']
    target = 'total_seasonal_points'
    
    latest_2024 = df[df['season_year'] == 2024].copy()
    latest_2024['actual_rank_2024'] = latest_2024['total_seasonal_points'].rank(ascending=False, method='min').astype(int)

    # 2. 10-ROUND ENSEMBLE CONSENSUS
    n_seeds = 10
    all_preds = []
    print("Running 10-Round Ensemble for Heatmap baseline...")
    
    for i in range(n_seeds):
        model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=6,
            random_state=100+i, early_stopping_rounds=50, n_jobs=-1
        )
        
        # Split for validation
        train_df = df[df['season_year'] < 2024]
        val_df = df[df['season_year'] == 2024]
        
        model.fit(train_df[features], train_df[target], 
                  eval_set=[(val_df[features], val_df[target])], verbose=False)
        
        all_preds.append(model.predict(latest_2024[features]))

    # 3. CONSOLIDATE METRICS
    latest_2024['predicted_2026_points'] = np.mean(all_preds, axis=0).round(2)
    latest_2024['predicted_rank_2026'] = latest_2024['predicted_2026_points'].rank(ascending=False, method='min').astype(int)
    latest_2024['rank_jump'] = latest_2024['actual_rank_2024'] - latest_2024['predicted_rank_2026']

    # 4. PREPARE HEATMAP DATA (Top 20 Players)
    top_20 = latest_2024.nsmallest(20, 'predicted_rank_2026').copy()
    
    # Selecting the two key comparison metrics
    heatmap_data = top_20.set_index('player_name')[['rank_jump', 'career_volatility']]
    
    # 5. VISUALIZATION
    plt.figure(figsize=(12, 10))
    # 'RdYlGn' cmap: Green = Positive Rank Jump, Red = High Volatility (Risk)
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, 
                linewidths=.5, cbar_kws={'label': 'Magnitude'})
    
    plt.title('Top 20 Players: 2026 Projected Rank Jump vs. Career Volatility', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Player Name', fontsize=12)
    plt.xlabel('Scouting Metrics', fontsize=12)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "scouting_heatmap_top20.png")
    plt.tight_layout()
    plt.savefig(save_path)
    
    print(f"SUCCESS: Scouting Heatmap saved to {save_path}")

if __name__ == "__main__":
    generate_scouting_heatmap()