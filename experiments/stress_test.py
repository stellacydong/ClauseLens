"""
stress_test.py
--------------
Performs catastrophe + capital stress tests for ClauseLens MARL agents.

- Loads trained MARL agents (fallback to random if none)
- Applies stress scenarios: high-cat frequency, capital reduction, extreme tail risk
- Evaluates Profit, CVaR, and Compliance KPIs for each scenario
- Outputs a summary table and saves detailed results to JSON
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Ensure src is in Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.simulate_env import TreatyEnv
from src.marl_agents import MARLAgent
from src.evaluation import evaluate_bids, summarize_portfolio

# ---------------------------
# Config
# ---------------------------
NUM_AGENTS = 3
NUM_EPISODES = 50
CHECKPOINT_DIR = "experiments/checkpoints"
DATA_DIR = "data"

np.random.seed(42)

with open(os.path.join(DATA_DIR, "sample_treaties.json")) as f:
    SAMPLE_TREATIES = json.load(f)


# ---------------------------
# Load Trained Agents
# ---------------------------
def load_trained_agents(checkpoint_dir=CHECKPOINT_DIR, num_agents=NUM_AGENTS):
    agents = []
    for i in range(num_agents):
        agent = MARLAgent(i)
        ckpt_file = os.path.join(checkpoint_dir, f"marl_agent_{i}.json")
        if os.path.exists(ckpt_file):
            with open(ckpt_file) as f:
                params = json.load(f)
            agent.load_parameters(params)
        agents.append(agent)
    return agents


# ---------------------------
# Stress Scenarios
# ---------------------------
STRESS_SCENARIOS = [
    {
        "name": "High-Cat Frequency",
        "loss_multiplier": 1.5,     # 50% more losses
        "tail_risk_multiplier": 1.3,
        "capital_shock": 1.0,
    },
    {
        "name": "Capital Reduction 20%",
        "loss_multiplier": 1.0,
        "tail_risk_multiplier": 1.0,
        "capital_shock": 0.8,      # 20% less available capital
    },
    {
        "name": "Severe Tail Scenario",
        "loss_multiplier": 1.2,
        "tail_risk_multiplier": 2.0, # Double tail risk stress
        "capital_shock": 0.9,
    },
]


# ---------------------------
# Stress Test Execution
# ---------------------------
def run_stress_test(num_episodes=NUM_EPISODES):
    agents = load_trained_agents()
    env = TreatyEnv(num_agents=NUM_AGENTS)
    results_by_scenario = {}

    for scenario in STRESS_SCENARIOS:
        print(f"\n=== Running Stress Test: {scenario['name']} ===")
        scenario_results = []

        for ep in range(num_episodes):
            treaty = SAMPLE_TREATIES[ep % len(SAMPLE_TREATIES)]
            state = env.reset(treaty_override=treaty)

            # MARL Bids
            bids = []
            for agent in agents:
                bid = agent.get_bid(state)
                # Apply stress multipliers
                bid["expected_loss"] *= scenario["loss_multiplier"]
                bid["tail_risk"] *= scenario["tail_risk_multiplier"]
                bids.append(bid)

            # Determine winner and apply capital shock
            winner_idx, _ = env.step(bids)
            winning_bid = bids[winner_idx]

            # Adjust KPIs for capital shock (affects compliance)
            kpi = evaluate_bids([winning_bid], winner_idx=0, treaty=state)
            if scenario["capital_shock"] < 1.0:
                # Simulate stricter compliance under lower capital
                kpi["regulatory_flags"]["all_ok"] = (
                    kpi["regulatory_flags"]["all_ok"] and
                    np.random.rand() < scenario["capital_shock"]
                )

            scenario_results.append(kpi)

        summary = summarize_portfolio(scenario_results)
        results_by_scenario[scenario["name"]] = summary

        print(f"Scenario: {scenario['name']}")
        print(f"  Avg Profit: ${summary['avg_profit']:,.0f}")
        print(f"  Avg CVaR:   ${summary['avg_cvar']:,.0f}")
        print(f"  Compliance: {summary['compliance_rate']*100:.0f}%")

    return results_by_scenario


# ---------------------------
# Run and Save Results
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting MARL Stress Tests...")
    results = run_stress_test()

    output_file = "experiments/stress_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Stress test results saved to {output_file}")
