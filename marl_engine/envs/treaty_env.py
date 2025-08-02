"""
treaty_env.py

Multi-Agent Treaty Bidding Environment for Transparent Market Platform.
- Compatible with Gymnasium / PettingZoo style
- Supports Centralized Training with Decentralized Execution (CTDE)
- Generates KPIs for CVaR-aware training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple


class TreatyBiddingEnv(gym.Env):
    """
    Multi-Agent Treaty Bidding Environment
    Agents: Reinsurers bidding on insurance treaties
    Observations: Market state, treaty features, agent-specific state
    Actions: Bid amounts (discrete or continuous)
    Rewards: Profit or CVaR-adjusted return
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_agents: int = 3,
        obs_dim: int = 6,
        action_space_type: str = "continuous",
        max_steps: int = 20,
        cvar_alpha: float = 0.95,
        random_seed: int = 42,
    ):
        super(TreatyBiddingEnv, self).__init__()

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.cvar_alpha = cvar_alpha
        self.rng = np.random.default_rng(random_seed)

        # Action Space
        if action_space_type == "continuous":
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        else:  # discrete bids (0-9 scaled)
            self.action_space = spaces.MultiDiscrete([10] * num_agents)

        # Observation Space: each agent sees same obs for simplicity (can be agent-specific)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_agents, obs_dim), dtype=np.float32
        )

        # Internal state
        self.current_step = 0
        self.agent_states = np.zeros((num_agents, obs_dim), dtype=np.float32)
        self.agent_profits = np.zeros(num_agents, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Core Gym Methods
    # -------------------------------------------------------------------------
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_profits = np.zeros(self.num_agents, dtype=np.float32)

        # Initialize market state (random treaty features)
        self.agent_states = self._sample_initial_state()

        obs = self._get_obs()
        info = {"step": self.current_step}
        return obs, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """
        Perform one bidding step.
        Args:
            actions (np.ndarray): Agent bids normalized to [0,1] or discrete

        Returns:
            obs, rewards, terminated, truncated, info
        """
        self.current_step += 1

        # Simulate treaty outcome
        rewards, info = self._simulate_market(actions)

        # Update profits
        self.agent_profits += rewards

        # Generate next observation
        self.agent_states = self._sample_next_state()
        obs = self._get_obs()

        # Check episode termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        return obs, rewards, terminated, truncated, info

    def render(self):
        """Optional: Print current step and profits"""
        print(f"Step {self.current_step} | Profits: {self.agent_profits}")

    def close(self):
        pass

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    def _sample_initial_state(self) -> np.ndarray:
        """Sample initial market/treaty states"""
        # Example: treaty size, risk score, cat factor, region encoding
        return self.rng.normal(0, 1, size=(self.num_agents, self.obs_dim)).astype(np.float32)

    def _sample_next_state(self) -> np.ndarray:
        """Simulate evolving market state"""
        drift = self.rng.normal(0, 0.1, size=(self.num_agents, self.obs_dim))
        return (self.agent_states + drift).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return observations for all agents"""
        return self.agent_states.copy()

    def _simulate_market(self, actions: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Simulate the effect of agent bids:
        - Higher bids increase win probability but reduce expected profit
        - Risk-adjusted return is considered for CVaR training
        """
        # Normalize if discrete
        if actions.dtype != np.float32:
            actions = actions.astype(np.float32) / 9.0

        # Simulate whether each agent wins (simple probability model)
        win_prob = np.clip(actions + self.rng.normal(0, 0.1, size=actions.shape), 0, 1)
        wins = self.rng.binomial(1, win_prob)

        # Compute profit = (premium - expected loss)
        premiums = actions * 1_000_000
        expected_losses = premiums * self.rng.uniform(0.5, 1.2, size=self.num_agents)
        rewards = (premiums - expected_losses) * wins

        # CVaR: compute downside risk metric
        cvar_penalty = np.percentile(rewards, (1 - self.cvar_alpha) * 100)
        rewards = rewards - abs(cvar_penalty) * 0.05  # penalize downside

        info = {
            "avg_profit": np.mean(rewards),
            "win_rate": np.mean(wins),
            "cvar_95": np.percentile(rewards, 5),
            "step": self.current_step
        }
        return rewards.astype(np.float32), info
