# src/demo_pipeline.py
import json
import random

from src.simulate_env import TreatyEnv
from src.marl_agents import MARLAgent
from src.clause_retrieval import ClauseRetriever
from src.explanation_generator import ClauseExplainer
from src import config


# Load all sample treaties
with open(config.SAMPLE_TREATIES_PATH, "r") as f:
    sample_treaties = json.load(f)


def run_demo(category_filter=None, jurisdiction_filter=None, random_seed=None):
    """
    Run a single ClauseLens + MARL demo episode.

    Args:
        category_filter (str, optional): Filter clauses by category (e.g., "Capital Adequacy")
        jurisdiction_filter (str, optional): Filter clauses by jurisdiction (e.g., "EU")
        random_seed (int, optional): Set random seed for reproducibility

    Returns:
        dict: Episode result containing:
              - treaty (dict)
              - winner (int)
              - winning_bid (dict)
              - clauses (list[dict])
              - explanation (str)
              - clauses_used (list[int])
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Initialize environment and components
    env = TreatyEnv(num_agents=3)
    retriever = ClauseRetriever(top_k=5)
    explainer = ClauseExplainer()

    # Pick a random treaty from sample list
    treaty = random.choice(sample_treaties)

    # Reset environment with selected treaty
    env.reset(treaty_override=treaty)

    # Initialize MARL agents
    agents = [MARLAgent(agent_id=i) for i in range(env.num_agents)]

    # Generate bids from all agents
    bids = [agent.get_bid(treaty) for agent in agents]

    # Simulate environment step to pick winner
    winner_idx, reward = env.step(bids)
    winning_bid = bids[winner_idx]

    # Retrieve clauses with optional filters
    clauses = retriever.retrieve(
        treaty,
        category_filter=category_filter,
        jurisdiction_filter=jurisdiction_filter
    )

    # Generate ClauseLens explanation
    explanation = explainer.generate_explanation(winning_bid, clauses)

    # Collect clause IDs for KPI tracking
    clause_ids = [c["id"] for c in clauses]

    return {
        "treaty": treaty,
        "winner": winner_idx,
        "winning_bid": winning_bid,
        "clauses": clauses,
        "clauses_used": clause_ids,
        "explanation": explanation
    }


# ---------------------------
# Multi-Episode Runner
# ---------------------------
# ---------------------------
# Multi-Episode Runner
# ---------------------------
def run_multi_episode(num_episodes=5, seed=None):
    """
    Run multiple demo episodes for investor dashboard.

    Args:
        num_episodes (int): Number of episodes to run
        seed (int, optional): Random seed for reproducibility

    Returns:
        list[dict]: List of episode results
    """
    import random
    if seed is not None:
        random.seed(seed)

    results = []
    treaties_pool = sample_treaties.copy()
    random.shuffle(treaties_pool)

    for ep in range(num_episodes):
        # Pick a new treaty each episode (cycle if needed)
        treaty = treaties_pool[ep % len(treaties_pool)]

        env = TreatyEnv(num_agents=3)
        env.reset(treaty_override=treaty)

        agents = [MARLAgent(agent_id=i) for i in range(env.num_agents)]
        bids = [agent.get_bid(treaty) for agent in agents]
        winner_idx, reward = env.step(bids)
        winning_bid = bids[winner_idx]

        retriever = ClauseRetriever(top_k=5)
        explainer = ClauseExplainer()

        clauses = retriever.retrieve(treaty)
        explanation = explainer.generate_explanation(winning_bid, clauses)

        results.append({
            "treaty": treaty,
            "winner": winner_idx,
            "winning_bid": winning_bid,
            "clauses": clauses,
            "clauses_used": [c["id"] for c in clauses],
            "explanation": explanation
        })

    return results


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    print("Running a single demo episode...\n")
    single_result = run_demo()
    print("Treaty:", single_result["treaty"])
    print("Winning Bid:", single_result["winning_bid"])
    print("Top Clauses:", [c["id"] for c in single_result["clauses"]])
    print("Explanation:\n", single_result["explanation"])

    print("\nRunning 3 episodes for multi-episode simulation...")
    multi_results = run_multi_episode(num_episodes=3)
    for i, r in enumerate(multi_results, start=1):
        print(f"\nEpisode {i}: Treaty={r['treaty']['cedent']}, Winner={r['winner']}, Clauses={r['clauses_used']}")
