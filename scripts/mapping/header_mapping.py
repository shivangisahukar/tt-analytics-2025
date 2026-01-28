#Data Cleaning

import os
import pandas as pd
import re


# Configuration
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def clean_text(text):
    if pd.isna(text) or text == "": return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def run_mapping_pipeline():
    all_rows = []
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    
    for filename in csv_files:
        path = os.path.join(RAW_DIR, filename)
        year_match = re.search(r'20\d{2}', filename)
        year = year_match.group(0) if year_match else "Unknown"
        
        # Load raw CSV
        df_raw = pd.read_csv(path, header=None)
        
        locations = df_raw.iloc[0].fillna("").tolist()
        dates = df_raw.iloc[1].fillna("").tolist()
        headers = df_raw.iloc[2].fillna("").tolist()
        data_rows = df_raw.iloc[3:]

        # --- STEP 1: STRICT METADATA MAPPING ---
        col_map = {}
        tournament_indices = []
        
        for i, h in enumerate(headers):
            h_clean = clean_text(h).lower()
            
            # Explicit metadata checks - looking for the specific ID/Name columns
            if i < 4: # Player metadata is ALWAYS in the first few columns
                if 'id' in h_clean: col_map['ttfi_id'] = i
                elif 'name' in h_clean: col_map['player_name'] = i
                elif 'state' in h_clean or h_clean == 'inst.': col_map['state_inst'] = i
                continue

            # Summary/Total point columns
            if 'points' in h_clean and 'best' not in h_clean: 
                col_map['total_points'] = i
            elif 'position' in h_clean or 'rank' in h_clean: 
                col_map['rank_position'] = i
            
            # --- STEP 2: TOURNAMENT DETECTION ---
            # If it's not metadata and has points/tournament keywords, it's an event
            elif any(key in h_clean for key in ['ranking', 'institutional', 'national', 'championship']):
                tournament_indices.append(i)

        # --- STEP 3: DATA EXTRACTION ---
        for _, row in data_rows.iterrows():
            if pd.isna(row[col_map.get('player_name', 2)]): continue
                
            player_base = {
                'season_year': year,
                'ttfi_id': row[col_map.get('ttfi_id')],
                'player_name': clean_text(row[col_map.get('player_name')]),
                'state_institution': clean_text(row[col_map.get('state_inst')]),
                'total_seasonal_points': row[col_map.get('total_points')],
                'final_rank_position': row[col_map.get('rank_position')]
            }
            
            for t_idx in tournament_indices:
                points_val = row[t_idx]
                if pd.notna(points_val) and str(points_val).strip() != "":
                    entry = player_base.copy()
                    entry.update({
                        'tournament_name': clean_text(headers[t_idx]),
                        'a1_location': clean_text(locations[t_idx]),
                        'a2_date': clean_text(dates[t_idx]),
                        'points_earned': points_val
                    })
                    all_rows.append(entry)

    master_df = pd.DataFrame(all_rows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    master_df.to_csv(os.path.join(PROCESSED_DIR, "master_long_dataset.csv"), index=False)
    print(f"PIPELINE SUCCESS: Processed {len(master_df)} tournament entries.")

if __name__ == "__main__":
    run_mapping_pipeline()