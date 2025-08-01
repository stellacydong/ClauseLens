import numpy as np

class BiddingAgent:
    """
    Base bidding agent for treaty pricing simulation.
    Can act heuristically or randomly for demo purposes.
    """
    def __init__(self, agent_id: str, risk_aversion: float = 0.1):
        self.id = agent_id
        self.risk_aversion = risk_aversion
        self.policy_params = {"premium_multiplier": (0.8, 1.2)}  # action space

    def act(self, state_row):
        """
        Choose an action given the current state.
        For demo: return a random premium multiplier.
        """
        low, high = self.policy_params["premium_multiplier"]
        return np.random.uniform(low, high)

    def evaluate(self, reward, cvar_95):
        """
        Evaluate agent performance given reward and CVaR.
        Default: risk-adjusted reward.
        """
        return reward - self.risk_aversion * cvar_95


class MAPPOAgent(BiddingAgent):
    """
    Multi-Agent PPO agent skeleton for treaty bidding.
    Placeholder for PPO/MAPPO logic; can be extended with PyTorch.
    """
    def __init__(self, agent_id: str, risk_aversion: float = 0.1, lr: float = 1e-3):
        super().__init__(agent_id, risk_aversion)
        self.lr = lr
        self.replay_buffer = []

    def act(self, state_row):
        """
        PPO policy: for demo, still random. Replace with NN policy later.
        """
        return super().act(state_row)

    def store_experience(self, state, action, reward, cvar_95, done):
        """
        Store experience tuple for PPO updates.
        """
        self.replay_buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "cvar_95": cvar_95,
            "done": done
        })

    def compute_risk_adjusted_reward(self, reward, cvar_95):
        """
        CVaR-aware reward shaping: penalize high tail-risk exposures.
        """
        return reward - self.risk_aversion * cvar_95

    def update_policy(self):
        """
        Placeholder for PPO/MAPPO policy update.
        In a real implementation, this would:
        1. Compute advantages and returns
        2. Update actor and critic networks
        3. Apply CVaR-aware objective
        """
        if not self.replay_buffer:
            return
        # Here you could add PyTorch training steps
        self.replay_buffer = []  # Clear buffer after update


class RandomAgent(BiddingAgent):
    """
    Simple random agent, inherits from BiddingAgent.
    Useful for baselines in MARL simulations.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id, risk_aversion=0.0)

    def act(self, state_row):
        return super().act(state_row)
