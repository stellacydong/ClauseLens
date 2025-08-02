#!/usr/bin/env python3
"""
04_generate_dashboard_data.py

Aggregates KPIs from:
1. MARL simulation results (from 03_run_simulation.py)
2. MarketLens ML outputs (acceptance & loss ratio predictions)
3. Fairness audit scores (optional)

Outputs:
    ../data/processed/dashboard_data.parquet
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

SIM_RESULTS_PATH = os.path.join(DATA_DIR, "simulation_results.parquet")
MARKETLENS_FEATURES_PATH = os.path.join(DATA_DIR, "marketlens_features.parquet")
MARKETLENS_LABELS_PATH = os.path.join(DATA_DIR, "marketlens_labels.parquet")
DASHBOARD_DATA_PATH = os.path.join(DATA_DIR, "dashboard_data.parquet")

FAIRNESS_METRICS_PATH = os.path.join(DATA_DIR, "fairness_metrics.parquet")  # optional


def load_or_empty(path, cols=None):
    """Helper to load a parquet file or return an empty DataFrame with given cols"""
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return pd.DataFrame(columns=cols if cols else [])


def aggregate_dashboard_data():
    # Load MARL simulation results
    sim_df = load_or_empty(SIM_RESULTS_PATH, ["round", "avg_profit", "win_rate", "cvar_95"])
    
    # Load MarketLens synthetic features and labels
    features_df = load_or_empty(MARKETLENS_FEATURES_PATH)
    labels_df = load_or_empty(MARKETLENS_LABELS_PATH)
    
    # Merge MarketLens data if available
    marketlens_df = pd.DataFrame()
    if not features_df.empty and not labels_df.empty:
        marketlens_df = features_df.copy()
        marketlens_df = marketlens_df.merge(labels_df, on="treaty_id", how="left")
    
    # Compute MarketLens KPIs if available
    marketlens_summary = {}
    if not marketlens_df.empty:
        marketlens_summary = {
            "avg_pred_acceptance": marketlens_df.get("pred_acceptance", pd.Series(dtype=float)).mean(),
            "avg_pred_loss_ratio": marketlens_df.get("pred_loss_ratio", pd.Series(dtype=float)).mean(),
            "portfolio_size": len(marketlens_df)
        }
    
    # Load fairness metrics (optional)
    fairness_df = load_or_empty(FAIRNESS_METRICS_PATH)
    fairness_score = fairness_df["fairness_score"].mean() if "fairness_score" in fairness_df else np.nan
    
    # Prepare final dashboard KPIs
    dashboard_data = {
        "timestamp": datetime.utcnow(),
        "avg_profit": sim_df["avg_profit"].mean() if not sim_df.empty else np.nan,
        "win_rate": sim_df["win_rate"].mean() if not sim_df.empty else np.nan,
        "cvar_95": sim_df["cvar_95"].mean() if not sim_df.empty else np.nan,
        "avg_pred_acceptance": marketlens_summary.get("avg_pred_acceptance", np.nan),
        "avg_pred_loss_ratio": marketlens_summary.get("avg_pred_loss_ratio", np.nan),
        "portfolio_size": marketlens_summary.get("portfolio_size", 0),
        "fairness_score": fairness_score,
    }

    dashboard_df = pd.DataFrame([dashboard_data])
    dashboard_df.to_parquet(DASHBOARD_DATA_PATH, index=False)
    
    print(f"[INFO] Dashboard data saved to {DASHBOARD_DATA_PATH}")
    return dashboard_df


if __name__ == "__main__":
    df = aggregate_dashboard_data()
    print(df)
