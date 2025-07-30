import numpy as np
import random

class TreatyEnv:
    def __init__(self, num_agents=3, seed=None):
        """
        Multi-agent reinsurance treaty bidding environment.

        Args:
            num_agents (int): Number of bidding agents.
            seed (int, optional): Random seed for reproducibility.
        """
        self.num_agents = num_agents
        self.current_treaty = None

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def reset(self, treaty_override=None):
        """
        Reset environment for a new episode.
        Returns the current treaty for agents to bid on.
        """
        if treaty_override is not None:
            self.current_treaty = treaty_override
        else:
            # Generate a simple random treaty if none is provided
            self.current_treaty = {
                "peril": random.choice(["Hurricane", "Flood", "Wildfire", "Earthquake", "Winter Storm"]),
                "region": random.choice(["Florida", "California", "US Midwest", "Canada"]),
                "exposure": int(np.random.uniform(3_000_000, 8_000_000)),
                "limit": round(np.random.uniform(0.3, 0.6), 2),
                "quota_share_cap": round(np.random.uniform(0.4, 0.6), 2),
                "notes": "Synthetic treaty generated for demo episode"
            }

        return self.current_treaty

    def step(self, bids):
        """
        Decide the winning bid based on highest profit.

        Args:
            bids (list of dict): Each bid dict requires:
                - 'premium'
                - 'expected_loss'
                - 'tail_risk'

        Returns:
            winner_idx (int): Index of winning agent
            reward (float): Profit of winning bid
        """
        if not bids or len(bids) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} bids, got {len(bids) if bids else 0}")

        # Compute profits
        profits = [b["premium"] - b["expected_loss"] for b in bids]
        winner_idx = int(np.argmax(profits))
        reward = profits[winner_idx]

        return winner_idx, reward
