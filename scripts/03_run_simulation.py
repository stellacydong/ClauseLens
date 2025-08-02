#!/usr/bin/env python3
"""
03_run_simulation.py

Runs Multi-Agent PPO/MAPPO simulations using the TreatyBiddingEnv
and outputs KPIs for MarketLens + Streamlit integration.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add marl_engine to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, "..", "marl_engine")
sys.path.append(ENGINE_DIR)

from envs.treaty_env import TreatyBiddingEnv
from agents.mappo_agent import MAPPOAgent

# Output path
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "simulation_results.parquet")


# -----------------------------------------------------------------------------
# Run Simulation with MAPPO
# -----------------------------------------------------------------------------
def run_simulation(num_episodes: int = 10, train_epochs: int = 2, max_steps: int = 20):
    """
    Runs MAPPO training/simulation for num_episodes and saves results.
    """
    num_agents = 3
    obs_dim = 6
    action_dim = 1  # Each agent outputs a single bid
    env = TreatyBiddingEnv(num_agents=num_agents, obs_dim=obs_dim, max_steps=max_steps)

    # Initialize MAPPO agent
    agent = MAPPOAgent(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim, device="cpu")

    results = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            # MAPPO selects actions for all agents
            actions = agent.select_actions(obs)

            # Step in environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Store transition in MAPPO buffer
            agent.store_transition(obs, actions, rewards, next_obs, terminated)

            obs = next_obs

        # Perform MAPPO update after each episode (or batch)
        agent.update(epochs=train_epochs)

        # Collect KPIs at episode end
        results.append({
            "round": episode,
            "avg_profit": info.get("avg_profit", np.nan),
            "win_rate": info.get("win_rate", np.nan),
            "cvar_95": info.get("cvar_95", np.nan),
            "timestamp": datetime.utcnow()
        })

        print(f"[EP {episode}] Profit={info.get('avg_profit', 0):.2f} | WinRate={info.get('win_rate', 0):.2f}")

    df_results = pd.DataFrame(results)

    # Save to parquet for Streamlit dashboard
    df_results.to_parquet(OUTPUT_FILE, index=False)
    print(f"[INFO] Simulation results saved to {OUTPUT_FILE}")

    return df_results


if __name__ == "__main__":
    df = run_simulation(num_episodes=10, train_epochs=2, max_steps=20)
    print(df)
