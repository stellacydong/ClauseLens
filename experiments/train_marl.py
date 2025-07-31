"""
experiments/train_marl.py

Trains Multi-Agent Reinforcement Learning (MARL) agents
for reinsurance treaty bidding and evaluates performance.

Checkpoints are saved to experiments/checkpoints/
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulate_env import TreatyEnv
from src.marl_agents import MARLAgent
from src.data_loader import load_sample_treaties
from src.evaluation import evaluate_bids, summarize_portfolio
from src.utils import set_seed

# ---------------------------
# Configuration
# ---------------------------
NUM_AGENTS = 3
NUM_EPISODES = 200
CHECKPOINT_DIR = os.path.join("experiments", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

set_seed(42)

# ---------------------------
# Load Sample Treaties
# ---------------------------
treaties = load_sample_treaties()
if not treaties:
    raise ValueError("No treaties found. Ensure data/sample_treaties.json exists.")

print(f"Loaded {len(treaties)} sample treaties for training.")

# ---------------------------
# Initialize Environment & Agents
# ---------------------------
env = TreatyEnv(num_agents=NUM_AGENTS)
agents = [MARLAgent(i) for i in range(NUM_AGENTS)]

portfolio_results_marl = []

# ---------------------------
# Training Loop
# ---------------------------
print("🚀 Starting MARL training...")

for episode in range(1, NUM_EPISODES + 1):
    # Randomly select a treaty
    treaty = np.random.choice(treaties)
    env.reset(treaty_override=treaty)

    # Agents submit bids
    bids = [agent.get_bid(treaty) for agent in agents]

    # Select winner (highest profit minus CVaR)
    scores = [(b["premium"] - b["expected_loss"]) - 0.2 * b["tail_risk"] for b in bids]
    winner_idx = int(np.argmax(scores))

    # Evaluate KPIs for the winning bid
    kpi = evaluate_bids([bids[winner_idx]], winner_idx=0, treaty=treaty)
    portfolio_results_marl.append(kpi)

    # Simple reward signal: profit for winner, negative CVaR for losers
    for i, agent in enumerate(agents):
        reward = kpi["profit"] if i == winner_idx else -kpi["cvar"]
        agent.update_policy(reward)

    # Print progress every 10 episodes
    if episode % 10 == 0:
        summary = summarize_portfolio(portfolio_results_marl[-10:])
        print(f"[Episode {episode}/{NUM_EPISODES}] "
              f"Avg Profit: {summary['avg_profit']:,.0f}, "
              f"Avg CVaR: {summary['avg_cvar']:,.0f}, "
              f"Compliance: {summary['compliance_rate']*100:.0f}%")

# ---------------------------
# Save Checkpoints
# ---------------------------
for agent in agents:
    agent_path = os.path.join(CHECKPOINT_DIR, f"marl_agent_{agent.agent_id}.json")
    agent.save(agent_path)

print(f"✅ Saved {len(agents)} MARL agents to {CHECKPOINT_DIR}")

# ---------------------------
# Training Summary
# ---------------------------
training_summary = summarize_portfolio(portfolio_results_marl)
summary_json = {
    "timestamp": datetime.now().isoformat(),
    "episodes": NUM_EPISODES,
    "summary": training_summary
}

summary_file = os.path.join("experiments", "training_summary.json")
with open(summary_file, "w") as f:
    json.dump(summary_json, f, indent=2)

print("\n=== Training Summary ===")
print(f"Episodes: {NUM_EPISODES}")
print(f"Average Profit: {training_summary['avg_profit']:,.0f}")
print(f"Average CVaR: {training_summary['avg_cvar']:,.0f}")
print(f"Compliance Rate: {training_summary['compliance_rate']*100:.0f}%")
print(f"Training Time: {0.0:.1f} sec")  # Add timing if needed
