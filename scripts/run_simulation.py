import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from datetime import datetime

from marl_engine.simulate_env import TreatyBiddingEnv, run_episode
from marl_engine.marl_agents import MAPPOAgent
from marl_engine.stress_tests import run_stress_tests, summarize_stress_results
from marl_engine.utils import compute_episode_summary, save_results, save_episode_summaries

# -----------------------------
# Config
# -----------------------------
DEMO_DATA_PATH = "../data/demo/sample_treaties.csv"
PROCESSED_DIR = "../data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

N_AGENTS = 5
EPISODES = 50
EPISODE_SIZE = 20

SIM_RUNS_PATH = os.path.join(PROCESSED_DIR, "simulation_runs.csv")
SIM_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "simulation_summary.csv")
SIM_STRESS_PATH = os.path.join(PROCESSED_DIR, "simulation_stressed.csv")

# -----------------------------
# 1. Load Demo Treaties
# -----------------------------
if not os.path.exists(DEMO_DATA_PATH):
    raise FileNotFoundError(
        "❌ Demo treaties not found. Run generate_synthetic_treaties.py first!"
    )

treaties_df = pd.read_csv(DEMO_DATA_PATH)
print(f"✅ Loaded {len(treaties_df)} demo treaties")

# -----------------------------
# 2. Initialize Environment & Agents
# -----------------------------
env = TreatyBiddingEnv(treaties_df, n_agents=N_AGENTS, episode_size=EPISODE_SIZE)
agents = [MAPPOAgent(f"A{i+1}", risk_aversion=0.2) for i in range(N_AGENTS)]

results = []
episode_summaries = []

# -----------------------------
# 3. Run Simulation Episodes
# -----------------------------
print(f"🎬 Running {EPISODES} MARL bidding episodes...")
for ep in range(EPISODES):
    env.reset()
    ep_df = run_episode(env, agents)
    results.append(ep_df)
    episode_summaries.append(compute_episode_summary(ep_df))

results_df = pd.concat(results, ignore_index=True)
print(f"✅ Completed simulation: {len(results_df)} total bids")

# -----------------------------
# 4. Generate Compliance Column
# -----------------------------
if "compliance" not in results_df.columns:
    if "cvar" in results_df.columns:
        # Compliance inversely related to CVaR (normalized to 0.6–1.0)
        results_df["compliance"] = 1.0 - (
            results_df["cvar"] / results_df["cvar"].max() * 0.4
        )
        results_df["compliance"] = results_df["compliance"].clip(lower=0.6, upper=1.0)
    else:
        # Random fallback for demo
        results_df["compliance"] = 0.6 + 0.4 * np.random.rand(len(results_df))

# -----------------------------
# 5. Save Raw Simulation Results
# -----------------------------
save_results(results_df, SIM_RUNS_PATH)
summary_df = save_episode_summaries(episode_summaries, SIM_SUMMARY_PATH)

# -----------------------------
# 6. Stress Test Simulation Results
# -----------------------------
print("⚡ Running stress tests...")
stressed_df = run_stress_tests(results_df)

# Ensure compliance also exists in stressed simulation
if "compliance" not in stressed_df.columns:
    if "cvar" in stressed_df.columns:
        stressed_df["compliance"] = 1.0 - (
            stressed_df["cvar"] / stressed_df["cvar"].max() * 0.4
        )
        stressed_df["compliance"] = stressed_df["compliance"].clip(lower=0.6, upper=1.0)
    else:
        stressed_df["compliance"] = 0.6 + 0.4 * np.random.rand(len(stressed_df))

save_results(stressed_df, SIM_STRESS_PATH)
stress_summary = summarize_stress_results(stressed_df)
print("✅ Stress Test Summary:", stress_summary)

# -----------------------------
# 7. Final Dashboard Message
# -----------------------------
print("\n🎯 Simulation complete! Outputs ready for dashboard:")
print(f"- Simulation runs:      {SIM_RUNS_PATH}")
print(f"- Episode summary:      {SIM_SUMMARY_PATH}")
print(f"- Stressed simulation:  {SIM_STRESS_PATH}")
print("✅ Compliance column included in outputs.")
