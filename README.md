# 🏓 Table Tennis Intelligence: 2026 Season Performance Forecasting

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/ML-XGBoost%20|%20LightGBM-green.svg" alt="ML Models">
  <img src="https://img.shields.io/badge/Status-Production--Ready-orange.svg" alt="Status">
</div>

## 📌 Project Overview
This repository contains a production-grade sports analytics pipeline designed to predict the **2026 Indian Table Tennis National Season**. By leveraging five years of historical TTFI (Table Tennis Federation of India) data (2020–2024), we've built a multi-model engine that forecasts rankings, clusters player archetypes, and simulates match probabilities using Elo ratings.

---

## 🏗️ 1. Data Structuring & Feature Engineering
Before modeling, the raw data was transformed into a "Supervised Learning" format. We engineered four critical features to capture the "DNA" of a player's performance.

### **a) Decay-Weighted Momentum**
Unlike simple averages, we implement a **Time-Weighted Average (TWA)**. Recent performance in 2024 is weighted more heavily than historical data from 2020.
* **Math:** $Momentum = \sum (Points_{year} \times 0.8^{2024-year})$
* **Purpose:** Captures current "form" while respecting career trajectory.

### **b) Volatility Index (Consistency)**
We calculate the **Standard Deviation** of ranking points earned across the 5-year window.
* **Low Volatility:** Indicates a "Steady Professional" who consistently reaches late tournament stages.
* **High Volatility:** Identifies "Giant Killers" or "Clutch Players" who are prone to upsets but can defeat top seeds.

### **c) Tournament Tiering**
Not all points are equal. Our pipeline applies multipliers based on tournament prestige:
* **Senior Nationals:** 2.0x weight (High Pressure).
* **National Ranking:** 1.0x weight (Standard).
* **Inter-Institutional:** 1.0x weight.

### **d) Institutional Synergy Encoding**
We encode the `STATE/INST` column (e.g., **RBI, PSPB, RSPB**). Historically, these institutions provide elite coaching ecosystems, acting as a strong categorical predictor for rank longevity.



---

## 🤖 2. Advanced ML Models

### **A. Performance Forecasting (XGBoost Ensemble)**
* **Model:** Gradient Boosted Trees (XGBoost) with a **10-Round Consensus Ensemble**.
* **Methodology:** We use a **Sliding Window** approach. The model trains on years $N$ through $N+2$ to predict $N+3$. 
* **Objective:** Predict the Total Ranking Points for the **2026 Season**.
* **Why:** Tree-based models handle non-linear career spikes and missing seasons better than standard regression.

### **B. Player Clustering (Unsupervised Archetypes)**
* **Model:** K-Means Clustering + Principal Component Analysis (PCA).
* **Objective:** Group the Top 100 players into functional archetypes:
    * **The Elite Core:** High points, high pressure score, low volatility.
    * **The Rising Stars:** High momentum, medium points, high growth.
    * **The Steady Veterans:** Low volatility, consistent medium point accumulation.
    * **The Wildcards:** High volatility; unpredictable performance patterns.

### **C. Elo Rating System (Probabilistic Skill)**
* **Model:** Custom Elo Simulation.
* **Execution:** We simulate "virtual matches" by comparing how players finished relative to each other in the same tournament. 
* **Formula:** $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$
* **Result:** A dynamic "Skill Rating" that updates after every event, allowing for win-probability predictions between any two players.

### **D. Survival Analysis (Career Longevity)**
* **Model:** Kaplan-Meier Estimator & Cox Proportional Hazards.
* **Objective:** Predict the "Risk" of a player dropping out of the National Top 50.
* **Process:** Treats "Ranking Dropout" as the event. It analyzes which signals (like a sudden drop in Delhi Ranking points) precede the end of an elite-tier career.



---

## 🛠️ Project Structure
```text
tt-analytics-2026/
├── data/
│   ├── processed/          # master_long_dataset.csv
│   └── insights/           # Scouting_Report_2026.pdf, clusters.png
├── scripts/
│   ├── feature_engineering/# extract_features.py, sliding_window.py
│   └── modeling/           # train_xgboost.py, player_clustering.py, elo_rating.py
├── main.py                 # Master Controller to run the full pipeline
└── requirements.txt        # xgboost, lifelines, scikit-learn, seaborn