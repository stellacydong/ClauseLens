"""
simulate_env.py
---------------
Simulated treaty environment for ClauseLens MARL agents:
- Handles environment reset and bid evaluation
- Returns treaty info for demo and training
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from src.data_loader import DataLoader


class TreatyEnv:
    """
    Multi-agent treaty simulation environment.
    """
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.loader = DataLoader()
        self.current_treaty: Dict = None

    # ---------------------------
    # Reset Environment
    # ---------------------------
    def reset(self, treaty_override: Dict = None) -> Dict:
        """
        Reset environment for a new episode.
        :param treaty_override: Optional treaty dict to use
        :return: Selected treaty dict
        """
        treaties = self.loader.load_treaties()
        self.current_treaty = treaty_override or random.choice(treaties)
        return self.current_treaty

    # ---------------------------
    # Step: Evaluate Bids
    # ---------------------------
    def step(self, bids: List[Dict]) -> Tuple[int, float]:
        """
        Evaluate bids and return winning agent index and reward.
        Reward is simplified as (premium - expected_loss).
        """
        if not bids:
            return 0, 0.0

        profits = [b["premium"] - b["expected_loss"] for b in bids]
        winner_idx = int(np.argmax(profits))
        reward = profits[winner_idx]
        return winner_idx, reward

    # ---------------------------
    # Simulate One Episode
    # ---------------------------
    def simulate_episode(self, agents) -> Dict:
        """
        Simulate a full episode: reset → agents bid → evaluate → return results.
        """
        treaty = self.reset()
        bids = [agent.get_bid(treaty) for agent in agents]
        winner_idx, reward = self.step(bids)

        return {
            "treaty": treaty,
            "bids": bids,
            "winner_idx": winner_idx,
            "winning_bid": bids[winner_idx],
            "reward": reward,
        }


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    from src.marl_agents import create_agents

    env = TreatyEnv(num_agents=3)
    agents = create_agents(3, mode="demo")

    print("🚀 Simulating 3 episodes...")
    for ep in range(3):
        result = env.simulate_episode(agents)
        print(
            f"Episode {ep+1}: Winner={result['winner_idx']}, "
            f"Profit={result['winning_bid']['premium']-result['winning_bid']['expected_loss']:,.0f}"
        )
