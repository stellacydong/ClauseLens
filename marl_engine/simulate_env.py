import numpy as np
import pandas as pd

class TreatyBiddingEnv:
    """
    Multi-Agent Treaty Bidding Environment
    --------------------------------------
    Simulates multi-agent reinsurance treaty bidding episodes.
    
    Each episode:
    1. Selects a subset of treaty opportunities
    2. Agents submit bids (premium multipliers)
    3. Computes simulated rewards and CVaR (tail risk)
    """

    def __init__(self, treaties_df, n_agents=5, episode_size=20, cvar_threshold=0.95):
        self.treaties_df = treaties_df
        self.n_agents = n_agents
        self.episode_size = episode_size
        self.cvar_threshold = cvar_threshold
        self.current_episode = 0
        self.history = []

    def reset(self):
        """Reset environment for a new episode."""
        self.current_episode += 1
        self.history = []
        return self._get_state()

    def _get_state(self):
        """
        Returns the current episode's treaty opportunities.
        """
        return self.treaties_df.sample(self.episode_size).reset_index(drop=True)

    def step(self, actions):
        """
        Takes agent actions (premium multipliers) and simulates results.
        Returns:
            rewards: list of simulated profits per agent
            cvars: list of simulated CVaR 95% per agent
            done: bool (episode end)
        """
        rewards = []
        cvars = []

        for a in actions:
            # Reward: simulate profit proportional to premium multiplier
            reward = np.random.normal(5_000_000 * a, 2_000_000)
            # Tail risk: higher variance if aggressive pricing
            cvar_95 = abs(np.random.normal(3_000_000 * a, 1_500_000))
            
            rewards.append(reward)
            cvars.append(cvar_95)

        done = True  # Single-step episodes for demo
        return rewards, cvars, done


def run_episode(env, agents):
    """
    Run a single episode in the environment with the given agents.
    Returns a DataFrame of agent results.
    """
    # Step 1: get state (subset of treaties)
    state = env._get_state()

    # Step 2: agents act
    actions = [agent.act(state.iloc[i % len(state)]) for i, agent in enumerate(agents)]

    # Step 3: environment simulates step
    rewards, cvars, done = env.step(actions)

    # Step 4: store results
    episode_df = pd.DataFrame({
        "episode": [env.current_episode] * len(agents),
        "agent_id": [agent.id for agent in agents],
        "action": actions,
        "reward": rewards,
        "cvar_95": cvars
    })

    for i, agent in enumerate(agents):
        agent.store_experience(
            state=state.iloc[i % len(state)].to_dict(),
            action=actions[i],
            reward=rewards[i],
            cvar_95=cvars[i],
            done=done
        )

    return episode_df
