"""
marl_agents.py
--------------
Multi-Agent Reinforcement Learning (MARL) agents for ClauseLens:
- Generates treaty bids (quota share, premium, tail risk)
- Supports demo (random) and training (epsilon-greedy) modes
- Includes simple reward-based policy updates for training
"""

import numpy as np
from typing import Dict


class MARLAgent:
    def __init__(self, agent_id: int, mode: str = "demo", epsilon: float = 0.1):
        """
        :param agent_id: Unique agent identifier
        :param mode: 'demo' (random) or 'train' (epsilon-greedy)
        :param epsilon: Exploration rate for training
        """
        self.agent_id = agent_id
        self.mode = mode
        self.epsilon = epsilon

        # Internal parameters for policy updates (very simple demo logic)
        self.base_quota = np.random.uniform(0.2, 0.5)
        self.base_margin = np.random.uniform(0.1, 0.3)

    # ---------------------------
    # Bid Generation
    # ---------------------------
    def get_bid(self, treaty: Dict) -> Dict:
        """
        Generate a bid for the given treaty.
        :param treaty: Dict with keys like exposure, quota_share_cap
        :return: Dict {agent_id, quota_share, premium, expected_loss, tail_risk}
        """
        exposure = treaty.get("exposure", 5_000_000)
        quota_cap = treaty.get("quota_share_cap", 0.5)

        # Expected loss as a function of exposure and randomness
        expected_loss = np.random.uniform(0.02, 0.06) * exposure

        # Quota share decision
        if self.mode == "train" and np.random.rand() < self.epsilon:
            # Explore: pick random within cap
            quota_share = np.random.uniform(0.1, quota_cap)
        else:
            # Exploit: biased towards base policy
            quota_share = min(quota_cap, max(0.05, np.random.normal(self.base_quota, 0.05)))

        # Premium: expected loss * margin
        margin_factor = np.random.uniform(1.15, 1.35)
        premium = expected_loss * margin_factor

        # Tail risk (CVaR component)
        tail_risk = expected_loss * np.random.uniform(0.2, 0.5)

        return {
            "agent_id": self.agent_id,
            "quota_share": float(quota_share),
            "premium": float(premium),
            "expected_loss": float(expected_loss),
            "tail_risk": float(tail_risk),
        }

    # ---------------------------
    # Training: Policy Update
    # ---------------------------
    def update_policy(self, reward: float):
        """
        Adjust the agent's base policy based on reward signal.
        Positive reward -> slightly increase quota/margin
        Negative reward -> slightly decrease risk appetite
        """
        if reward > 0:
            self.base_quota = min(0.6, self.base_quota + 0.01)
            self.base_margin = min(0.35, self.base_margin + 0.01)
        else:
            self.base_quota = max(0.05, self.base_quota - 0.01)
            self.base_margin = max(0.05, self.base_margin - 0.01)

    def __repr__(self):
        return f"MARLAgent(id={self.agent_id}, quota={self.base_quota:.2f}, margin={self.base_margin:.2f})"


# ---------------------------
# Multi-Agent Utility
# ---------------------------
def create_agents(num_agents: int, mode: str = "demo", epsilon: float = 0.1):
    """
    Helper function to create multiple MARL agents.
    """
    return [MARLAgent(agent_id=i, mode=mode, epsilon=epsilon) for i in range(num_agents)]


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    dummy_treaty = {
        "exposure": 5_000_000,
        "quota_share_cap": 0.5
    }
    agents = create_agents(3, mode="demo")
    print("Generated Bids:")
    for agent in agents:
        bid = agent.get_bid(dummy_treaty)
        print(bid)
        agent.update_policy(bid["premium"] - bid["expected_loss"])
