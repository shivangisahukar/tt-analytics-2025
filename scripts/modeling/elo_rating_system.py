import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data/processed/master_long_dataset.csv"
OUTPUT_DIR = "data/insights"
K_FACTOR = 32 # Standard sensitivity for skill rating changes

def calculate_elo(rating_a, rating_b, actual_a):
    """Calculates the new ratings for two players after a match."""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + K_FACTOR * (actual_a - expected_a)
    return round(new_rating_a, 2)

def run_elo_simulation():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Load and Sort Data Chronologically
    df = pd.read_csv(INPUT_FILE)
    df['season_year'] = pd.to_numeric(df['season_year'])
    # Sorting by year to ensure ratings evolve over time
    df = df.sort_values(by=['season_year', 'tournament_name'])

    # 2. Initialize Ratings
    unique_players = df['ttfi_id'].unique()
    player_ratings = {pid: 1500 for pid in unique_players}
    player_names = df.set_index('ttfi_id')['player_name'].to_dict()

    # 3. Simulate Tournament "Virtual Matches"
    # We group by tournament to compare players playing in the same event
    grouped = df.groupby(['season_year', 'tournament_name'])
    
    print("Simulating Elo ratings across 5 seasons...")
    
    for (year, tourney), group in grouped:
        # Sort players in this tournament by points (Highest to lowest)
        standings = group.sort_values(by='points_earned', ascending=False)
        pids = standings['ttfi_id'].tolist()
        
        # Every player 'plays' everyone else in the tournament
        # Higher points = Win, Lower points = Loss
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p1, p2 = pids[i], pids[j]
                
                r1, r2 = player_ratings[p1], player_ratings[p2]
                
                # p1 finished higher than p2, so p1 wins
                player_ratings[p1] = calculate_elo(r1, r2, 1)
                player_ratings[p2] = calculate_elo(r2, r1, 0)

    # 4. Convert Results to DataFrame
    elo_df = pd.DataFrame([
        {'ttfi_id': pid, 'player_name': player_names[pid], 'elo_rating': rating}
        for pid, rating in player_ratings.items()
    ])

    # 5. Generate Win Probability Matrix (Top 5 Players)
    top_5 = elo_df.nlargest(5, 'elo_rating')
    print("\n--- Current Top 5 by Elo Rating (Skill Level) ---")
    print(top_5[['player_name', 'elo_rating']].to_string(index=False))

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    elo_df.sort_values('elo_rating', ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "player_elo_ratings.csv"), index=False
    )
    print(f"\nSUCCESS: Elo ratings saved to {OUTPUT_DIR}/player_elo_ratings.csv")

if __name__ == "__main__":
    run_elo_simulation()