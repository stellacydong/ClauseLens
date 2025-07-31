# src/demo_pipeline.py

import json, random
import numpy as np
from src.simulate_env import TreatyEnv
from src.marl_agents import MARLAgent
from src.evaluation import evaluate_bids
from src.clause_retrieval import ClauseRetriever
from src.explanation_generator import ClauseExplainer
from app.components.bidding_animation import simulate_bidding

# Load sample treaties
SAMPLE_TREATIES_PATH = "data/sample_treaties.json"
with open(SAMPLE_TREATIES_PATH, "r") as f:
    sample_treaties = json.load(f)

def run_multi_episode(num_episodes=3, steps_per_episode=30, show_animation=False):
    """Simulate multiple treaty bidding episodes and return exactly 3 objects."""
    
    env = TreatyEnv(num_agents=3)
    retriever = ClauseRetriever()
    explainer = ClauseExplainer()

    results = []
    portfolio_results_marl = []
    portfolio_results_baseline = []

    for ep in range(num_episodes):
        # 1. Select treaty for this episode
        treaty = random.choice(sample_treaties)
        env.reset(treaty_override=treaty)

        # 2. Create agents
        agents = [MARLAgent(i) for i in range(env.num_agents)]

        # 3. Simulate bidding
        bids = (
            simulate_bidding(agents, treaty, steps=steps_per_episode, delay=0.02)
            if show_animation
            else [agent.get_bid(treaty) for agent in agents]
        )

        # 4. Pick winner
        winner_idx = max(range(len(bids)), key=lambda i: bids[i]["premium"])
        winning_bid = bids[winner_idx]

        # 5. Compute KPIs
        marl_kpi = evaluate_bids([winning_bid], winner_idx=0, treaty=treaty)
        portfolio_results_marl.append(marl_kpi)

        exp_loss = np.random.uniform(100_000, 200_000)
        baseline_bid = {
            "agent_id": "Baseline",
            "quota_share": 0.3,
            "premium": exp_loss * 1.2,
            "expected_loss": exp_loss,
            "tail_risk": exp_loss * 0.3,
        }
        baseline_kpi = evaluate_bids([baseline_bid], winner_idx=0, treaty=treaty)
        portfolio_results_baseline.append(baseline_kpi)

        # 6. ClauseLens explanation
        clauses = retriever.retrieve(treaty)
        explanation = explainer.generate_explanation(winning_bid, clauses)

        results.append({
            "episode": ep + 1,
            "treaty": treaty,
            "winning_bid": winning_bid,
            "clauses": clauses,
            "explanation": explanation
        })

    return results, portfolio_results_marl, portfolio_results_baseline
