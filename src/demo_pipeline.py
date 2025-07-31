"""
demo_pipeline.py
----------------
Pipeline for running ClauseLens demo episodes:
- Loads treaties and clauses
- Runs MARL agents to generate bids
- Retrieves and explains clauses for investor-friendly output
"""

import random
from typing import Dict, List, Tuple

from src import config
from src.data_loader import DataLoader
from src.simulate_env import TreatyEnv
from src.marl_agents import MARLAgent
from src.clause_retrieval import ClauseRetriever
from src.explanation_generator import ClauseExplainer
from src.evaluation import evaluate_bids


# ---------------------------
# Initialize Components
# ---------------------------
loader = DataLoader()
retriever = ClauseRetriever()
explainer = ClauseExplainer()


# ---------------------------
# Single Episode
# ---------------------------
def run_demo(treaty_override: Dict = None) -> Dict:
    """
    Run a single demo episode with optional treaty override.
    Returns full episode results.
    """
    treaties = loader.load_treaties()
    treaty = treaty_override or random.choice(treaties)

    # Initialize environment and agents
    env = TreatyEnv(num_agents=config.NUM_AGENTS)
    env.reset(treaty_override=treaty)
    agents = [MARLAgent(agent_id=i) for i in range(config.NUM_AGENTS)]

    # Agents generate bids
    bids = [agent.get_bid(treaty) for agent in agents]

    # Determine winner (highest profit or random tiebreak)
    profits = [b["premium"] - b["expected_loss"] for b in bids]
    winner_idx = int(max(range(len(profits)), key=lambda i: profits[i]))
    winning_bid = bids[winner_idx]

    # Clause retrieval & explanation
    clauses = retriever.retrieve(treaty, top_k=config.TOP_K_CLAUSES)
    explanation = explainer.generate_explanation(winning_bid, clauses)

    # Compute KPIs
    kpi = evaluate_bids([winning_bid], winner_idx=0, treaty=treaty)
    kpi["explanation"] = explanation

    return {
        "treaty": treaty,
        "bids": bids,
        "winner_idx": winner_idx,
        "winning_bid": winning_bid,
        "clauses": clauses,
        "explanation": explanation,
        "kpi": kpi,
    }


# ---------------------------
# Multi-Episode Run
# ---------------------------
def run_multi_episode(num_episodes: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Runs multiple demo episodes and returns:
        results_list, portfolio_results_marl, portfolio_results_baseline
    Each element in results_list is a dict from run_demo().
    """
    results_list: List[Dict] = []
    portfolio_results_marl: List[Dict] = []
    portfolio_results_baseline: List[Dict] = []

    treaties = loader.load_treaties()
    for ep in range(num_episodes):
        treaty = treaties[ep % len(treaties)]
        result = run_demo(treaty_override=treaty)
        results_list.append(result)

        # MARL evaluation (just winning bid)
        marl_kpi = result["kpi"]
        portfolio_results_marl.append(marl_kpi)

        # Baseline evaluation (actuarial pricing ~ expected_loss * 1.2)
        exp_loss = result["winning_bid"]["expected_loss"]
        baseline_bid = {
            "agent_id": "Baseline",
            "quota_share": 0.3,
            "premium": exp_loss * 1.2,
            "expected_loss": exp_loss,
            "tail_risk": exp_loss * 0.3,
        }
        baseline_kpi = evaluate_bids([baseline_bid], winner_idx=0, treaty=treaty)
        portfolio_results_baseline.append(baseline_kpi)

    return results_list, portfolio_results_marl, portfolio_results_baseline


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    print("🚀 Running a single demo episode...")
    result = run_demo()
    print("Winner:", result["winner_idx"], "Profit:", result["kpi"]["profit"])

    print("\n🏃 Running 3 multi-episodes...")
    results, marl, base = run_multi_episode(num_episodes=3)
    print(f"Generated {len(results)} episodes, MARL avg profit: {sum([r['profit'] for r in marl])/len(marl):,.0f}")
