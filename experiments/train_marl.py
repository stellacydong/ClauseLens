"""
train_marl.py
-------------
Training script for MARL agents for ClauseLens treaty bidding.

- Initializes multi-agent environment (TreatyEnv)
- Trains MARL agents using simple PPO-like updates (demo-friendly)
- Logs episode metrics (profit, CVaR, compliance)
- Saves trained agent parameters for reuse in demo_app.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Ensure src is on Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.simulate_env import TreatyEnv
from src.marl_agents import MARLAgent
from src.evaluation import evaluate_bids, summarize_portfolio

# ---------------------------
# Training Config
# ---------------------------
NUM_AGENTS = 3
NUM_EPISODES = 200
SAVE_DIR = "experiments/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Sample treaty pool
with open("data/sample_treaties.json") as f:
    SAMPLE_TREATIES = json.load(f)

np.random.seed(42)


# ---------------------------
# Simple MARL Training Loop
# ---------------------------
def train_marl_agents(num_episodes=NUM_EPISODES):
    env = TreatyEnv(num_agents=NUM_AGENTS)
    agents = [MARLAgent(i) for i in range(NUM_AGENTS)]

    all_rewards = []
    portfolio_results = []

    for episode in range(num_episodes):
        treaty = SAMPLE_TREATIES[episode % len(SAMPLE_TREATIES)]
        state = env.reset(treaty_override=treaty)

        # Each agent proposes a bid
        bids = [agent.get_bid(state) for agent in agents]

        # Environment decides winner
        winner_idx, reward = env.step(bids)
        winning_bid = bids[winner_idx]

        # Evaluate the winning bid
        kpi = evaluate_bids([winning_bid], winner_idx=0, treaty=state)
        portfolio_results.append(kpi)
        all_rewards.append(kpi["profit"])

        # Simple training placeholder: Adjust agent biases
        for i, agent in enumerate(agents):
            agent.update_policy(kpi["profit"] if i == winner_idx else -kpi["cvar"])

        if (episode + 1) % 10 == 0:
            summary = summarize_portfolio(portfolio_results[-10:])
            print(f"[Episode {episode+1}/{num_episodes}] "
                  f"Avg Profit: {summary['avg_profit']:.0f}, "
                  f"Avg CVaR: {summary['avg_cvar']:.0f}, "
                  f"Compliance: {summary['compliance_rate']*100:.0f}%")

    return agents, portfolio_results


# ---------------------------
# Save Trained Agents
# ---------------------------
def save_agents(agents, save_dir=SAVE_DIR):
    for i, agent in enumerate(agents):
        agent_file = os.path.join(save_dir, f"marl_agent_{i}.json")
        with open(agent_file, "w") as f:
            json.dump(agent.export_parameters(), f)
    print(f"✅ Saved {len(agents)} MARL agents to {save_dir}")


if __name__ == "__main__":
    print("🚀 Starting MARL training...")
    start_time = time.time()

    agents, results = train_marl_agents()
    save_agents(agents)

    duration = time.time() - start_time
    summary = summarize_portfolio(results)
    print("\n=== Training Summary ===")
    print(f"Episodes: {summary['episodes']}")
    print(f"Average Profit: {summary['avg_profit']:.0f}")
    print(f"Average CVaR: {summary['avg_cvar']:.0f}")
    print(f"Compliance Rate: {summary['compliance_rate']*100:.0f}%")
    print(f"Training Time: {duration:.1f} sec")
