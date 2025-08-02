"""
mappo_agent.py

Multi-Agent PPO (MAPPO) agent for Treaty Bidding Environment
- Centralized critic (CTDE)
- Decentralized actors per agent
- Compatible with Gymnasium-style envs
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

# -----------------------------------------------------------------------------
# Neural Network Components
# -----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # outputs in [-1,1], scaled later
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int = 128):
        super().__init__()
        # Centralized critic sees all agents' observations
        self.net = nn.Sequential(
            nn.Linear(obs_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# MAPPO Agent Class
# -----------------------------------------------------------------------------
class MAPPOAgent:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        device: str = "cpu"
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.device = device

        # Decentralized actors
        self.actors = [Actor(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]

        # Centralized critic
        self.critic = Critic(obs_dim, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = []  # Store experiences for PPO updates

    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------
    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Select actions for each agent given their observations
        Args:
            obs: np.ndarray of shape (num_agents, obs_dim)
        Returns:
            actions: np.ndarray of shape (num_agents,)
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions = []
        for i, actor in enumerate(self.actors):
            action = actor(obs_tensor[i]).detach().cpu().numpy()
            # Map from [-1,1] to [0,1] for bidding
            action_scaled = (action + 1) / 2
            actions.append(action_scaled)
        return np.array(actions)

    # -------------------------------------------------------------------------
    # Experience Buffer Management
    # -------------------------------------------------------------------------
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        self.memory.append((obs, actions, rewards, next_obs, dones))

    def clear_memory(self):
        self.memory.clear()

    # -------------------------------------------------------------------------
    # PPO Update (simplified)
    # -------------------------------------------------------------------------
    def update(self, epochs: int = 4, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return

        # Convert memory to tensors
        obs, actions, rewards, next_obs, dones = zip(*self.memory)
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)

        # Compute advantages and returns
        returns, advantages = self.compute_advantages(obs, rewards, next_obs, dones)

        # Update actors
        for i, actor in enumerate(self.actors):
            # PPO surrogate loss
            logits = actor(obs[:, i, :])
            log_probs = -((actions[:, i] - logits) ** 2).mean()  # simplified continuous log-prob
            ratio = torch.exp(log_probs - log_probs.detach())
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            loss_actor = -torch.min(ratio * advantages, clipped).mean()

            self.actor_optimizers[i].zero_grad()
            loss_actor.backward()
            self.actor_optimizers[i].step()

        # Update centralized critic
        critic_input = obs.reshape(obs.shape[0], -1)
        values = self.critic(critic_input).squeeze()
        loss_critic = ((returns - values) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Clear buffer
        self.clear_memory()

    # -------------------------------------------------------------------------
    # Advantage Estimation (GAE-Lambda)
    # -------------------------------------------------------------------------
    def compute_advantages(self, obs, rewards, next_obs, dones):
        rewards = rewards.mean(axis=1)  # average reward across agents
        values = self.critic(obs.reshape(obs.shape[0], -1)).detach().cpu().numpy().squeeze()
        next_values = self.critic(next_obs.reshape(next_obs.shape[0], -1)).detach().cpu().numpy().squeeze()

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (0 if dones[t] else next_values[t]) - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + values
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
