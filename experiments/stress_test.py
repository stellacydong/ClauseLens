"""
experiments/stress_test.py

Runs catastrophe + capital stress tests for MARL and baseline reinsurance strategies.
Outputs results to stress_test_results.json for investor dashboards and analysis.
"""

import os
import json
import numpy as np
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_sample_treaties
from src.evaluation import evaluate_bids, summarize_portfolio
from src.marl_agents import MARLAgent
from src.simulate_env import TreatyEnv
from src.utils import set_seed

# ---------------------------
# Configuration
# ---------------------------
NUM_AGENTS = 3
STRESS_SCENARIOS_FILE = os.path.join("data", "stress_scenarios.json")
OUTPUT_FILE = os.path.join("experiments", "stress_test_results.json")
NUM_EPISODES = 50

set_seed(42)

# ---------------------------
# Load Data
# ---------------------------
treaties = load_sample_treaties()

if os.path.exists(STRESS_SCENARIOS_FILE):
    with open(STRESS_SCENARIOS_FILE, "r") as f:
        stress_scenarios = json.load(f)
else:
    raise FileNotFoundError(f"Stress scenario file not found: {STRESS_SCENARIOS_FILE}")

print(f"Loaded {len(treaties)} sample treaties and {len(stress_scenarios)} stress scenarios.")

# ---------------------------
# Helper Functions
# ---------------------------
def baseline_bid(treaty):
    """Simple actuarial baseline: expected loss * 1.2 premium."""
    exp_loss = np.random.uniform(100_000, 200_000)
    return {
        "agent_id": "Baseline",
        "quota_share": min(0.3, treaty.get("quota_share_cap", 0.3)),
        "premium": exp_loss * 1.2,
        "expected_loss": exp_loss,
        "tail_risk": exp_loss * 0.3,
    }

def apply_stress(treaty, scenario):
    """Adjust treaty exposure and expected loss according to stress severity."""
    stressed_treaty = dict(treaty)
    severity = scenario.get("severity", 1.0)
    
    # Example: scale exposure and add stress flag
    stressed_treaty["exposure"] *= severity
    stressed_treaty["stress_scenario"] = scenario["name"]
    
    return stressed_treaty

# ---------------------------
# Stress Test Execution
# ---------------------------
env = TreatyEnv(num_agents=NUM_AGENTS)
agents = [MARLAgent(i) for i in range(NUM_AGENTS)]

results = []
portfolio_results_marl = []
portfolio_results_baseline = []

print("🚀 Running stress tests...")

for episode in range(NUM_EPISODES):
    treaty = np.random.choice(treaties)
    scenario = np.random.choice(stress_scenarios)
    
    # Apply stress
    stressed_treaty = apply_stress(treaty, scenario)
    env.reset(treaty_override=stressed_treaty)
    
    # Generate bids
    marl_bids = [agent.get_bid(stressed_treaty) for agent in agents]
    winner_idx = np.random.randint(len(agents))  # placeholder selection
    marl_kpi = evaluate_bids([marl_bids[winner_idx]], winner_idx=0, treaty=stressed_treaty)
    
    baseline = baseline_bid(stressed_treaty)
    baseline_kpi = evaluate_bids([baseline], winner_idx=0, treaty=stressed_treaty)
    
    # Store results
    results.append({
        "episode": episode + 1,
        "treaty_id": stressed_treaty["id"],
        "stress_scenario": scenario["name"],
        "marl_profit": marl_kpi["profit"],
        "marl_cvar": marl_kpi["cvar"],
        "marl_compliance": marl_kpi["regulatory_flags"]["all_ok"],
        "baseline_profit": baseline_kpi["profit"],
        "baseline_cvar": baseline_kpi["cvar"],
        "baseline_compliance": baseline_kpi["regulatory_flags"]["all_ok"]
    })
    
    portfolio_results_marl.append(marl_kpi)
    portfolio_results_baseline.append(baseline_kpi)

# ---------------------------
# Summary
# ---------------------------
summary_marl = summarize_portfolio(portfolio_results_marl)
summary_baseline = summarize_portfolio(portfolio_results_baseline)

summary = {
    "timestamp": datetime.now().isoformat(),
    "num_episodes": NUM_EPISODES,
    "marl_summary": summary_marl,
    "baseline_summary": summary_baseline,
    "results": results
}

# ---------------------------
# Save Results
# ---------------------------
os.makedirs("experiments", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Stress test complete. Results saved to {OUTPUT_FILE}")
