"""
policy_training.py
------------------
Core training loop for ClauseLens MARL agents:
- Runs simulated episodes
- Updates agent policies based on reward signals
- Saves checkpoints for later use in demos and dashboards
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from src import config
from src.data_loader import DataLoader
from src.simulate_env import TreatyEnv
from src.marl_agents import create_agents
from src.evaluation import evaluate_bids, summarize_portfolio
from src.logger import ClauseLensLogger


class PolicyTrainer:
    def __init__(self, num_agents: int = 3, num_episodes: int = 200, epsilon: float = 0.1):
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.epsilon = epsilon

        self.loader = DataLoader()
        self.logger = ClauseLensLogger()
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)

    # ---------------------------
    # Episode Simulation
    # ---------------------------
    def simulate_episode(self, agents, treaty: Dict) -> Tuple[int, Dict, List[Dict]]:
        """
        Simulate a single episode for a given treaty with multiple agents.
        Returns winner index, KPI, and all bids.
        """
        env = TreatyEnv(num_agents=self.num_agents)
        env.reset(treaty_override=treaty)

        bids = [agent.get_bid(treaty) for agent in agents]
        profits = [b["premium"] - b["expected_loss"] for b in bids]
        winner_idx = int(np.argmax(profits))

        kpi = evaluate_bids([bids[winner_idx]], winner_idx=0, treaty=treaty)
        return winner_idx, kpi, bids

    # ---------------------------
    # Training Loop
    # ---------------------------
    def train(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Run MARL training loop with reward-based updates.
        Returns final agent parameters and all KPI results.
        """
        treaties = self.loader.load_treaties()
        agents = create_agents(self.num_agents, mode="train", epsilon=self.epsilon)

        portfolio_results = []

        self.logger.info(f"🚀 Starting MARL Training for {self.num_episodes} episodes...")

        for ep in range(self.num_episodes):
            treaty = treaties[ep % len(treaties)]
            winner_idx, kpi, bids = self.simulate_episode(agents, treaty)
            portfolio_results.append(kpi)

            # Update agents based on reward (profit for winner, -cvar for losers)
            for i, agent in enumerate(agents):
                reward = kpi["profit"] if i == winner_idx else -kpi["cvar"]
                agent.update_policy(reward)

            # Logging
            if (ep + 1) % 10 == 0 or ep == 0:
                summary = summarize_portfolio(portfolio_results[-10:])
                self.logger.info(
                    f"[Episode {ep+1}/{self.num_episodes}] "
                    f"Avg Profit: {summary['avg_profit']:,.0f}, "
                    f"Avg CVaR: {summary['avg_cvar']:,.0f}, "
                    f"Compliance: {summary['compliance_rate']*100:.0f}%"
                )

        self.logger.info("✅ Training complete.")
        return agents, portfolio_results

    # ---------------------------
    # Checkpoint Save & Load
    # ---------------------------
    def save_agents(self, agents, filename: str = "marl_agents.json"):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        file_path = self.checkpoint_dir / filename

        agents_data = [
            {"agent_id": agent.agent_id, "base_quota": agent.base_quota, "base_margin": agent.base_margin}
            for agent in agents
        ]

        with open(file_path, "w") as f:
            json.dump(agents_data, f, indent=2)
        self.logger.info(f"💾 Saved {len(agents)} MARL agents to {file_path}")

    def load_agents(self, filename: str = "marl_agents.json"):
        file_path = self.checkpoint_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No saved agents found at {file_path}")

        with open(file_path, "r") as f:
            agents_data = json.load(f)

        from src.marl_agents import MARLAgent
        agents = []
        for data in agents_data:
            agent = MARLAgent(agent_id=data["agent_id"], mode="demo")
            agent.base_quota = data["base_quota"]
            agent.base_margin = data["base_margin"]
            agents.append(agent)

        self.logger.info(f"🔄 Loaded {len(agents)} MARL agents from {file_path}")
        return agents


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    trainer = PolicyTrainer(num_agents=3, num_episodes=50, epsilon=0.1)
    agents, results = trainer.train()
    trainer.save_agents(agents)
    summary = summarize_portfolio(results)
    print("\n=== Training Summary ===")
    print(summary)
