import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# --- ROBUST PATH MANAGEMENT ---
# This finds the 'scripts' folder and adds it to Python's search list
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# --- IMPORT MODELS ---
try:
    from feature_engineering.extract_features import run_advanced_feature_pipeline
    from feature_engineering.sliding_window import create_advanced_sliding_window
    from modeling.train_xgboost_ensemble import run_ensemble_scouting_report
    from modeling.player_clustering import run_player_clustering
    from modeling.elo_rating_system import run_elo_simulation
    from modeling.survival_analysis import run_survival_analysis
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure 'scripts/modeling/train_xgboost_ensemble.py' exists.")
    print("2. Ensure '__init__.py' exists in 'modeling' and 'feature_engineering' folders.")
    sys.exit(1)

def run_scouting_pipeline():
    print("\n🚀 GENERATING 2026 TABLE TENNIS SCOUTING REPORT\n" + "="*50)
    
    # 1. DATA REFRESH
    print("[1/3] Processing 5-Year Data & Volatility...")
    run_advanced_feature_pipeline()
    create_advanced_sliding_window()

    # 2. RUN MODELS
    print("\n[2/3] Running AI Ensemble, Skill Ratings, and Longevity Analysis...")
    run_ensemble_scouting_report() 
    run_player_clustering()        
    run_elo_simulation()           
    run_survival_analysis()        

    # 3. PDF COMPILATION
    print("\n[3/3] Saving PDF Report...")
    # Navigate to the root's data/insights folder
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    insights_dir = os.path.join(ROOT_DIR, "data", "insights")
    pdf_path = os.path.join(insights_dir, "Scouting_Report_2026.pdf")
    
    visuals = [
        "momentum_score.png",
        "consistency_index.png",
        "player_archetype_clusters.png",
        "career_survival_curve.png",
        "scouting_heatmap_top20.png"
    ]

    try:
        os.makedirs(insights_dir, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            # Report Cover
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.6, "2026 Table Tennis\nPerformance Forecast", 
                     fontsize=24, ha='center', fontweight='bold')
            plt.text(0.5, 0.45, f"Date: {datetime.now().strftime('%B %Y')}", 
                     fontsize=14, ha='center')
            plt.axis('off')
            pdf.savefig()
            plt.close()

            # Add Model Visuals
            for img_name in visuals:
                path = os.path.join(insights_dir, img_name)
                if os.path.exists(path):
                    img = plt.imread(path)
                    plt.figure(figsize=(11, 8.5))
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
                    print(f"  ✅ Added to PDF: {img_name}")

        print(f"\nFinal Report Saved: {pdf_path}\n" + "="*50)
        
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")

if __name__ == "__main__":
    run_scouting_pipeline()