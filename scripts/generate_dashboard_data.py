import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd

from marl_engine.utils import compute_dashboard_kpis

# -----------------------------
# Config
# -----------------------------
PROCESSED_DIR = "../data/processed"
DASHBOARD_DIR = "../data/demo"

os.makedirs(DASHBOARD_DIR, exist_ok=True)

SIM_RUNS_PATH = os.path.join(PROCESSED_DIR, "simulation_runs.csv")
SIM_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "simulation_summary.csv")
SIM_STRESS_PATH = os.path.join(PROCESSED_DIR, "simulation_stressed.csv")

# Output paths
DASHBOARD_KPI_PATH = os.path.join(DASHBOARD_DIR, "dashboard_kpis.csv")
DASHBOARD_TREND_PATH = os.path.join(DASHBOARD_DIR, "dashboard_trends.csv")
DASHBOARD_BIDS_PATH = os.path.join(DASHBOARD_DIR, "dashboard_bids.csv")
DASHBOARD_STRESS_PATH = os.path.join(DASHBOARD_DIR, "dashboard_stress_summary.csv")

# -----------------------------
# 1. Load Simulation Data
# -----------------------------
if not os.path.exists(SIM_RUNS_PATH) or not os.path.exists(SIM_SUMMARY_PATH):
    raise FileNotFoundError("❌ Simulation results not found. Run run_simulation.py first!")

sim_df = pd.read_csv(SIM_RUNS_PATH)
summary_df = pd.read_csv(SIM_SUMMARY_PATH)

print(f"✅ Loaded simulation runs: {len(sim_df)} rows")
print(f"✅ Loaded episode summary: {len(summary_df)} episodes")

# -----------------------------
# 2. Load Stress Test Results (Optional)
# -----------------------------
stress_df = None
if os.path.exists(SIM_STRESS_PATH):
    stress_df = pd.read_csv(SIM_STRESS_PATH)
    print(f"✅ Loaded stressed simulation: {len(stress_df)} rows")

# -----------------------------
# 3. Compute Dashboard KPIs
# -----------------------------
# Ensure summary has necessary fields
if "avg_compliance" not in summary_df.columns:
    summary_df["avg_compliance"] = 0.85  # default for demo

kpi_df = compute_dashboard_kpis(summary_df)

# -----------------------------
# 4. Save Dashboard Files
# -----------------------------
# KPIs and trends
kpi_df.to_csv(DASHBOARD_KPI_PATH, index=False)
kpi_df.to_csv(DASHBOARD_TREND_PATH, index=False)

# Live bids: last 200
sim_df.sort_values(by=["episode"], inplace=True)
sim_df.tail(200).to_csv(DASHBOARD_BIDS_PATH, index=False)

print(f"✅ Saved dashboard KPIs to {DASHBOARD_KPI_PATH}")
print(f"✅ Saved dashboard trends to {DASHBOARD_TREND_PATH}")
print(f"✅ Saved dashboard bids (last 200) to {DASHBOARD_BIDS_PATH}")

# -----------------------------
# 5. Aggregate Stress Test Summary
# -----------------------------
if stress_df is not None:
    stress_summary = {
        "episodes": stress_df["episode"].nunique(),
        "mean_reward_post_cat": stress_df["reward_cat"].mean(),
        "mean_reward_downturn": stress_df["reward_downturn"].mean(),
        "mean_cvar_squeezed": stress_df["cvar_squeeze"].mean(),
        "mean_risk_adj_return": stress_df["risk_adj_return"].mean(),
        "max_cvar_squeezed": stress_df["cvar_squeeze"].max(),
    }
    pd.DataFrame([stress_summary]).to_csv(DASHBOARD_STRESS_PATH, index=False)
    print(f"✅ Saved dashboard stress summary to {DASHBOARD_STRESS_PATH}")

print("\n🎯 Dashboard data ready. Load these in demo_app.py for Streamlit visualization.")

