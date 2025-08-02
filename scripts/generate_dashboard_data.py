import os
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_DEMO = os.path.join(PROJECT_ROOT, "data", "demo")
os.makedirs(DATA_DEMO, exist_ok=True)

SIM_RUNS_PATH = os.path.join(DATA_PROCESSED, "simulation_runs.csv")
SIM_SUMMARY_PATH = os.path.join(DATA_PROCESSED, "simulation_summary.csv")
SIM_STRESS_PATH = os.path.join(DATA_PROCESSED, "simulation_stressed.csv")

DASHBOARD_KPIS = os.path.join(DATA_DEMO, "dashboard_kpis.csv")
DASHBOARD_TRENDS = os.path.join(DATA_DEMO, "dashboard_trends.csv")
DASHBOARD_BIDS = os.path.join(DATA_DEMO, "dashboard_bids.csv")
DASHBOARD_STRESS = os.path.join(DATA_DEMO, "dashboard_stress_summary.csv")

# -----------------------------
# Parameters
# -----------------------------
EPISODE_SIZE = 20  # Must match run_simulation.py for index fallback

# -----------------------------
# Load Simulation Data
# -----------------------------
if not os.path.exists(SIM_RUNS_PATH) or not os.path.exists(SIM_SUMMARY_PATH):
    raise FileNotFoundError("❌ Simulation data not found. Run run_simulation.py first!")

runs_df = pd.read_csv(SIM_RUNS_PATH)
summary_df = pd.read_csv(SIM_SUMMARY_PATH)
print(f"✅ Loaded simulation runs: {len(runs_df)} rows")
print(f"✅ Loaded episode summary: {len(summary_df)} episodes")

# -----------------------------
# Ensure Compliance Column
# -----------------------------
if "compliance" not in runs_df.columns:
    print("⚠️ 'compliance' column not found. Generating proxy compliance...")
    if "cvar" in runs_df.columns:
        runs_df["compliance"] = 1.0 - (runs_df["cvar"] / runs_df["cvar"].max() * 0.4)
        runs_df["compliance"] = runs_df["compliance"].clip(lower=0.6, upper=1.0)
    else:
        runs_df["compliance"] = 0.6 + 0.4 * np.random.rand(len(runs_df))

# Save updated runs_df back to processed
runs_df.to_csv(SIM_RUNS_PATH, index=False)

# -----------------------------
# 1. Compute Latest KPIs
# -----------------------------
latest_avg_profit = runs_df["profit"].mean() if "profit" in runs_df else 0
latest_avg_cvar = runs_df["cvar"].mean() if "cvar" in runs_df else 0
latest_avg_compliance = runs_df["compliance"].mean()  # float 0.6–1.0

kpi_df = pd.DataFrame([{
    "avg_profit": latest_avg_profit,
    "avg_cvar": latest_avg_cvar,
    "avg_compliance": latest_avg_compliance
}])
kpi_df.to_csv(DASHBOARD_KPIS, index=False)
print(f"✅ Saved dashboard KPIs to {DASHBOARD_KPIS}")

# -----------------------------
# 2. Episode Trends for Line Charts
# -----------------------------
trend_df = summary_df.copy()

# Detect grouping key: use episode if available, else derive from index
if "episode" in runs_df.columns:
    episode_groups = runs_df.groupby("episode")
else:
    runs_df["episode"] = runs_df.index // EPISODE_SIZE
    episode_groups = runs_df.groupby("episode")

# Compute avg_profit per episode if missing
if "avg_profit" not in trend_df.columns and "profit" in runs_df.columns:
    grouped_profit = episode_groups["profit"].mean()
    trend_df["avg_profit"] = grouped_profit.reindex(
        range(len(trend_df)), fill_value=grouped_profit.iloc[-1]
    ).values

# Compute avg_compliance per episode
if "avg_compliance" not in trend_df.columns:
    grouped_compliance = episode_groups["compliance"].mean()
    trend_df["avg_compliance"] = grouped_compliance.reindex(
        range(len(trend_df)), fill_value=grouped_compliance.iloc[-1]
    ).values

trend_df.to_csv(DASHBOARD_TRENDS, index=False)
print(f"✅ Saved dashboard trends to {DASHBOARD_TRENDS}")

# -----------------------------
# 3. Recent Bids Table
# -----------------------------
bids_df = runs_df.copy()

# Add timestamp if missing
if "timestamp" not in bids_df.columns:
    bids_df["timestamp"] = pd.date_range(
        start=pd.Timestamp.now() - pd.Timedelta(minutes=len(bids_df)),
        periods=len(bids_df),
        freq="min"
    )

bids_df.sort_values("timestamp", ascending=False).head(200).to_csv(DASHBOARD_BIDS, index=False)
print(f"✅ Saved dashboard bids to {DASHBOARD_BIDS}")

# -----------------------------
# 4. Optional: Stress Test Summary
# -----------------------------
if os.path.exists(SIM_STRESS_PATH):
    stress_df = pd.read_csv(SIM_STRESS_PATH)

    # Ensure compliance for stressed df as well
    if "compliance" not in stress_df.columns:
        if "cvar" in stress_df.columns:
            stress_df["compliance"] = 1.0 - (stress_df["cvar"] / stress_df["cvar"].max() * 0.4)
            stress_df["compliance"] = stress_df["compliance"].clip(lower=0.6, upper=1.0)
        else:
            stress_df["compliance"] = 0.6 + 0.4 * np.random.rand(len(stress_df))

    stress_summary = {
        "mean_reward_post_cat": stress_df.get("profit", pd.Series(dtype=float)).mean(),
        "mean_cvar": stress_df.get("cvar", pd.Series(dtype=float)).mean(),
        "mean_compliance": stress_df["compliance"].mean(),
        "episodes": stress_df.get("episode", pd.Series(dtype=float)).nunique(),
    }

    pd.DataFrame([stress_summary]).to_csv(DASHBOARD_STRESS, index=False)
    print(f"✅ Saved dashboard stress summary to {DASHBOARD_STRESS}")
else:
    print("ℹ️ No stressed simulation found. Skipping dashboard_stress_summary.csv")

# -----------------------------
# Final Message
# -----------------------------
print("\n🎯 Dashboard data ready. Load these in demo_app.py for Streamlit visualization.")
