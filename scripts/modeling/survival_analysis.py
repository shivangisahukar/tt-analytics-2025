import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os

# Configuration
INPUT_FILE = "data/processed/master_long_dataset.csv"
OUTPUT_DIR = "data/insights"

def run_survival_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    df['season_year'] = pd.to_numeric(df['season_year'])
    
    # 2. Define "Survival" Data
    # For each player, find the start year and end year in the Top 50
    # We define an 'Event' as a player who was in Top 50 but is no longer there in 2024.
    
    career_stats = df.groupby('ttfi_id').agg({
        'season_year': ['min', 'max', 'count'],
        'player_name': 'first'
    })
    career_stats.columns = ['start_year', 'end_year', 'years_active', 'player_name']
    
    # Duration: How many years have they survived?
    career_stats['duration'] = (career_stats['end_year'] - career_stats['start_year']) + 1
    
    # Observed (Event): 1 if they are NOT in 2024 (their career 'ended'), 0 if they are still active
    career_stats['observed'] = (career_stats['end_year'] < 2024).astype(int)

    # 3. Fit the Kaplan-Meier Model
    kmf = KaplanMeierFitter()
    kmf.fit(durations=career_stats['duration'], event_observed=career_stats['observed'])

    # 4. Visualization
    plt.figure(figsize=(10, 7))
    kmf.plot_survival_function()
    
    plt.title('Player Longevity: Probability of Staying in Top 50', fontsize=15, fontweight='bold')
    plt.xlabel('Years in National Top 50', fontsize=12)
    plt.ylabel('Survival Probability (Percentage)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "career_survival_curve.png"))
    
    # 5. Export Longevity Risk Report
    # Lower survival probability = Higher risk of falling out of 2026 rankings
    career_stats['survival_prob_at_current_age'] = kmf.predict(career_stats['duration']).values
    
    report_path = os.path.join(OUTPUT_DIR, "career_longevity_report.csv")
    career_stats.sort_values('survival_prob_at_current_age').to_csv(report_path)
    
    print(f"SUCCESS: Survival Curve and Longevity Report saved to {OUTPUT_DIR}/")
    print("\n--- Average Career Half-Life ---")
    print(f"Median Survival Time: {kmf.median_survival_time_} years")

if __name__ == "__main__":
    # Ensure lifelines is installed: pip install lifelines
    run_survival_analysis()