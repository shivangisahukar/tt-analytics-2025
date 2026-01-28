import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# Configuration
INPUT_FILE = "data/processed/supervised_timeseries_data.csv"
INSIGHT_DIR = "data/insights"

def run_ensemble_scouting_report():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure sliding_window.py ran successfully.")
        return

    # 1. Load the supervised dataset
    df = pd.read_csv(INPUT_FILE)
    
    # Features trained on 5-year history
    features = ['pts_lag_1', 'pts_lag_2', 'pts_lag_3', 'momentum_yoy', 'career_volatility']
    target = 'total_seasonal_points'

    # Baseline for 2026 prediction is the 2024 season state
    latest_2024 = df[df['season_year'] == 2024].copy()
    
    # 2. THE ENSEMBLE ENGINE (10-Round Consensus)
    n_seeds = 10
    all_round_predictions = []

    print(f"Starting 10-Round Ensemble Forecast for the 2026 Season...")

    for i in range(n_seeds):
        # Deterministic seeding for 100% consistency
        current_seed = 100 + i
        
        # Walk-forward split
        train_df = df[df['season_year'] < 2024]
        val_df = df[df['season_year'] == 2024]

        model = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.02,  # Fine-tuned for ensemble stability
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=current_seed,
            early_stopping_rounds=50,
            n_jobs=-1
        )

        model.fit(
            train_df[features], train_df[target],
            eval_set=[(val_df[features], val_df[target])],
            verbose=False
        )

        # Generate prediction and store
        round_forecast = model.predict(latest_2024[features])
        all_round_predictions.append(round_forecast)
        print(f"  > consensus Round {i+1} locked.")

    # 3. CONSOLIDATING RESULTS
    latest_2024['predicted_2026_points'] = np.mean(all_round_predictions, axis=0).round(2)
    
    # --- RANKING LOGIC ---
    # Actual Rank 2024
    latest_2024['actual_rank_2024'] = latest_2024['total_seasonal_points'].rank(ascending=False, method='min').astype(int)
    
    # Predicted Rank 2026
    latest_2024['predicted_rank_2026'] = latest_2024['predicted_2026_points'].rank(ascending=False, method='min').astype(int)
    
    # Rank Jump (Baseline - Prediction)
    latest_2024['rank_jump'] = latest_2024['actual_rank_2024'] - latest_2024['predicted_rank_2026']

    # 4. ADD VISUAL INDICATORS (The Scouting Arrow)
    def get_movement_arrow(jump):
        if jump > 0: return f"↑ (+{jump})"
        elif jump < 0: return f"↓ ({jump})"
        else: return "↔ (Stable)"

    latest_2024['scouting_trend'] = latest_2024['rank_jump'].apply(get_movement_arrow)

    # 5. SAVE FINAL REPORT
    os.makedirs(INSIGHT_DIR, exist_ok=True)
    report_path = os.path.join(INSIGHT_DIR, "ensemble_2026_scouting_report.csv")
    
    scouting_view = latest_2024[[
        'player_name', 
        'actual_rank_2024', 
        'predicted_rank_2026', 
        'scouting_trend',
        'total_seasonal_points', 
        'predicted_2026_points'
    ]].sort_values('predicted_rank_2026')

    scouting_view.to_csv(report_path, index=False)
    
    print(f"\nSUCCESS: 2026 Scouting Report saved to {report_path}")
    print("\n--- Top 10 Projected 2026 Leaderboard ---")
    print(scouting_view.head(10).to_string(index=False))

if __name__ == "__main__":
    run_ensemble_scouting_report()