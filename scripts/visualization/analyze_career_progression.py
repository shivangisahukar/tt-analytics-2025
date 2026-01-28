#This script precisely identifies the player with the highest point delta between 2020 and 2024 and generates a detailed growth chart.



import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # MacOS specific backend fix
import matplotlib.pyplot as plt
import os
import re

# Configuration
RAW_DIR = "data/raw"
OUTPUT_DIR = "data/insights"
YEARS = [2020, 2021, 2022, 2023, 2024]

def extract_yearly_totals():
    data_list = []
    
    for year in YEARS:
        filename = f"TTFI_FINAL_RANKING_{year}.csv"
        path = os.path.join(RAW_DIR, filename)
        
        if not os.path.exists(path):
            print(f"Warning: {filename} not found in {RAW_DIR}")
            continue
            
        # Load CSV (headers are on Row 2, index 2)
        df_raw = pd.read_csv(path, header=None)
        headers = [str(x).upper().strip() for x in df_raw.iloc[2]]
        
        # Map critical columns
        col_map = {}
        for i, h in enumerate(headers):
            if 'TTFI ID' in h: col_map['ttfi_id'] = i
            elif 'NAME' in h: col_map['player_name'] = i
            elif 'POINTS' in h and 'BEST' not in h: col_map['points'] = i

        # Extract data from row 3 onwards
        for _, row in df_raw.iloc[3:].iterrows():
            try:
                pid = str(row[col_map['ttfi_id']]).strip().replace('.0', '')
                name = str(row[col_map['player_name']]).strip()
                pts = str(row[col_map['points']]).replace(',', '').strip()
                pts_val = float(pts) if pts and pts != 'nan' else 0.0
                
                if pid and name != 'nan':
                    data_list.append({'Year': year, 'ID': pid, 'Name': name, 'Points': pts_val})
            except Exception:
                continue

    return pd.DataFrame(data_list)

def analyze_most_progress():
    df = extract_yearly_totals()
    
    # Identify players present in both the start (2020) and end (2024) years
    p2020 = set(df[df['Year'] == 2020]['ID'])
    p2024 = set(df[df['Year'] == 2024]['ID'])
    consistent_players = p2020.intersection(p2024)
    
    growth_stats = []
    for pid in consistent_players:
        start_pts = df[(df['ID'] == pid) & (df['Year'] == 2020)]['Points'].max()
        end_pts = df[(df['ID'] == pid) & (df['Year'] == 2024)]['Points'].max()
        name = df[df['ID'] == pid]['Name'].iloc[0]
        growth_stats.append({'ID': pid, 'Name': name, 'Start': start_pts, 'End': end_pts, 'Growth': end_pts - start_pts})
    
    # Find the top player
    top_player = pd.DataFrame(growth_stats).sort_values(by='Growth', ascending=False).iloc[0]
    history = df[df['ID'] == top_player['ID']].sort_values(by='Year')
    
    # --- VIBRANT VISUALIZATION ---
    plt.figure(figsize=(12, 7), facecolor='#f4f4f4')
    years = history['Year'].tolist()
    points = history['Points'].tolist()
    
    # Plotting the main line
    plt.plot(years, points, marker='o', linestyle='-', color='#1a73e8', linewidth=4, markersize=12, label='Career Points')
    
    # Annotate total points and yearly jumps
    for i in range(len(years)):
        plt.text(years[i], points[i] + 8, f"{int(points[i])}", ha='center', fontsize=11, fontweight='bold')
        if i > 0:
            jump = points[i] - points[i-1]
            plt.annotate(f"+{int(jump)}", xy=((years[i]+years[i-1])/2, (points[i]+points[i-1])/2),
                         xytext=(0, 15), textcoords='offset points', ha='center',
                         color='#d93025' if jump < 0 else '#1e8e3e', fontweight='bold', arrowprops=dict(arrowstyle='->', color='gray'))

    # Styling
    plt.title(f"Highest 5-Year Career Progress: {top_player['Name']}", fontsize=18, fontweight='bold', color='#202124')
    plt.xlabel("Season Year", fontsize=13)
    plt.ylabel("Total Ranking Points", fontsize=13)
    plt.xticks(YEARS)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "most_progress_player.png")
    plt.savefig(save_path)
    print(f"\nANALYSIS COMPLETE")
    print(f"Top Player: {top_player['Name']} (ID: {top_player['ID']})")
    print(f"Total Progress: {top_player['Growth']} points increase")
    print(f"Chart saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    analyze_most_progress()