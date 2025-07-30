import numpy as np

class MARLAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        # Demo parameters: could represent a bias in bidding
        self.bias = np.random.uniform(0.9, 1.1)

    def get_bid(self, treaty):
        """
        Generate a bid with quota share, premium, expected loss, and tail risk.
        """
        quota_share = np.random.uniform(0.1, treaty["quota_share_cap"])
        expected_loss = np.random.uniform(100_000, 500_000)
        premium = expected_loss * self.bias * np.random.uniform(1.1, 1.3)
        tail_risk = expected_loss * np.random.uniform(0.2, 0.5)

        return {
            "agent_id": self.agent_id,
            "quota_share": quota_share,
            "premium": premium,
            "expected_loss": expected_loss,
            "tail_risk": tail_risk,
        }

    def update_policy(self, reward):
        """
        Simple demo update: adjust bias slightly based on reward.
        Positive reward nudges bias up, negative reward nudges down.
        """
        step = 0.01
        self.bias += step * np.sign(reward)
        # Keep bias in a reasonable range
        self.bias = np.clip(self.bias, 0.8, 1.2)

    def export_parameters(self):
        """Export agent parameters for saving to JSON."""
        return {"agent_id": self.agent_id, "bias": float(self.bias)}

    def load_parameters(self, params):
        """Load parameters from JSON."""
        self.bias = params.get("bias", 1.0)
