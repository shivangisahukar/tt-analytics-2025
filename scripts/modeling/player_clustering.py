import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Configuration
INPUT_FILE = "data/processed/features_master.csv"
OUTPUT_DIR = "data/insights"

def run_player_clustering():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_features.py first!")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # Select features for clustering
    # We use: Momentum, Volatility (Risk), Pressure (Big Games), and Total Pts
    cluster_features = ['momentum_score', 'volatility_index', 'pressure_score', 'total_pts']
    
    # Handle any remaining NaNs (players with very little history)
    data = df[cluster_features].fillna(df[cluster_features].mean())

    # 2. Scale Features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 3. K-Means Clustering (Using K=4 as a standard sports archetype baseline)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)

    # 4. Archetype Labeling Logic
    # We analyze the averages of each cluster to assign human-readable names
    cluster_profiles = df.groupby('cluster')[cluster_features].mean()
    
    # Logic to map cluster numbers to names based on their 'DNA'
    # This is a simplified mapping; usually, we look at the 'pressure_score' and 'momentum'
    def label_archetype(row):
        if row['pressure_score'] > cluster_profiles['pressure_score'].mean() and row['total_pts'] > cluster_profiles['total_pts'].mean():
            return "Elite Core"
        elif row['momentum_score'] > cluster_profiles['momentum_score'].mean():
            return "Rising Star"
        elif row['volatility_index'] > cluster_profiles['volatility_index'].mean():
            return "Wildcard / Giant Killer"
        else:
            return "Steady Veteran"

    df['archetype'] = df.apply(label_archetype, axis=1)

    # 5. PCA for 2D Visualization
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)
    df['pca_1'] = pca_results[:, 0]
    df['pca_2'] = pca_results[:, 1]

    # 6. Generate Cluster Map
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='pca_1', y='pca_2', 
        hue='archetype', 
        style='archetype',
        data=df, 
        s=150, 
        palette='viridis',
        alpha=0.8
    )
    
    # Annotate Top 10 players for reference
    top_players = df.nlargest(10, 'total_pts')
    for i, row in top_players.iterrows():
        plt.text(row['pca_1']+0.1, row['pca_2'], row['player_name'], fontsize=9, alpha=0.7)

    plt.title('Table Tennis Player Archetypes (Unsupervised K-Means)', fontsize=15, fontweight='bold')
    plt.xlabel('Principal Component 1 (Performance Volume)')
    plt.ylabel('Principal Component 2 (Performance Style)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "player_archetype_clusters.png"))
    
    # 7. Save results
    df.to_csv(os.path.join(OUTPUT_DIR, "player_clusters_report.csv"), index=False)
    print("SUCCESS: Player clustering and PCA map generated.")
    print(df['archetype'].value_counts())

if __name__ == "__main__":
    run_player_clustering()