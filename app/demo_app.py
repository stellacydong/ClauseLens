import sys, os, json, time, unicodedata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fpdf import FPDF

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_bids, summarize_portfolio
from src.marl_agents import MARLAgent
from app.components.bidding_animation import simulate_bidding, live_bidding_summary
from app.components.clause_explainer import ClauseExplainerComponent
from app.components.kpi_dashboard import (
    display_portfolio_kpis,
    display_episode_table,
    display_profit_vs_cvar_chart,
)

# Stress test imports
from experiments.stress_test import run_stress_test, STRESS_SCENARIOS

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "data"
CHECKPOINT_DIR = "experiments/checkpoints"
RESULTS_FILE = "experiments/stress_test_results.json"

with open(os.path.join(DATA_DIR, "sample_treaties.json")) as f:
    SAMPLE_TREATIES = json.load(f)

st.set_page_config(page_title="ClauseLens Investor Demo", layout="wide")
st.title("🧠 ClauseLens + Multi-Agent Treaty Bidding Demo")

st.markdown("""
**Investor Demo Highlights:**
1. Live multi-agent MARL bidding
2. ClauseLens explanations with regulatory traceability
3. Portfolio KPIs: Profit, CVaR, and Compliance
4. Optional **Stress Test Dashboard** for capital resilience
""")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Simulation Settings")
episodes = st.sidebar.slider("Number of Episodes", 1, len(SAMPLE_TREATIES), 3)
steps = st.sidebar.slider("Bidding Animation Steps", 10, 100, 50)
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)
rerun_stress = st.sidebar.button("🔄 Rerun Stress Tests")
run_button = st.sidebar.button("Run Simulation 🚀")

# ---------------------------
# Utilities
# ---------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def load_trained_agents(checkpoint_dir=CHECKPOINT_DIR, num_agents=3):
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
# Main Tabs
# ---------------------------
tabs = st.tabs(["🏦 Live Demo", "📊 Portfolio Summary", "⚡ Stress Test Dashboard"])

# ---------------------------
# Tab 1: Live Demo
# ---------------------------
with tabs[0]:
    st.subheader("🎯 Live Multi-Agent Bidding with ClauseLens")

    if run_button:
        np.random.seed(seed)
        explainer = ClauseExplainerComponent(os.path.join(DATA_DIR, "clauses_corpus.json"), seed=seed)
        agents = load_trained_agents(CHECKPOINT_DIR, num_agents=3)

        results, kpis = [], []
        selected_treaties = SAMPLE_TREATIES[:episodes]

        for ep_idx, treaty in enumerate(selected_treaties):
            st.markdown(f"### Episode {ep_idx+1}: {treaty['cedent']}")
            bids = simulate_bidding(agents, treaty=treaty, steps=steps, delay=0.02)
            winner_idx = live_bidding_summary(bids)

            clauses, explanation = explainer.explain_treaty(treaty, bids[winner_idx])
            ep_result = {
                "treaty": treaty,
                "winner": winner_idx,
                "winning_bid": bids[winner_idx],
                "clauses": clauses,
                "explanation": explanation
            }
            results.append(ep_result)
            kpis.append(evaluate_bids([bids[winner_idx]], winner_idx=0, treaty=treaty))

            # Display explanation and clauses
            st.markdown("**ClauseLens Explanation:**")
            st.write(explanation)
            with st.expander("Retrieved Clauses"):
                for clause in clauses:
                    st.info(f"ID {clause['id']} ({clause.get('category','N/A')}): {clause['text']}")

        st.session_state["results"] = results
        st.session_state["kpis"] = kpis

# ---------------------------
# Tab 2: Portfolio Summary
# ---------------------------
with tabs[1]:
    st.subheader("📊 MARL vs Baseline Portfolio KPIs")

    if "results" in st.session_state and st.session_state["results"]:
        results = st.session_state["results"]
        kpis = st.session_state["kpis"]

        # Compute baseline KPIs
        def baseline_bid(treaty):
            exp_loss = np.random.uniform(100_000, 200_000)
            return {
                "agent_id": "Baseline",
                "quota_share": min(0.3, treaty["quota_share_cap"]),
                "premium": exp_loss * 1.2,
                "expected_loss": exp_loss,
                "tail_risk": exp_loss * 0.3,
            }

        baseline_kpis = []
        for ep in results:
            base_bid = baseline_bid(ep["treaty"])
            baseline_kpis.append(evaluate_bids([base_bid], winner_idx=0, treaty=ep["treaty"]))

        summary_marl = summarize_portfolio(kpis)
        summary_base = summarize_portfolio(baseline_kpis)

        display_portfolio_kpis(summary_marl, summary_base)
        display_episode_table(kpis, baseline_kpis)
        chart_path = "profit_vs_cvar.png"
        display_profit_vs_cvar_chart(kpis, baseline_kpis)
        plt.savefig(chart_path)

# ---------------------------
# Tab 3: Stress Test Dashboard
# ---------------------------
with tabs[2]:
    st.subheader("⚡ Capital & Catastrophe Stress Testing")

    if rerun_stress or not os.path.exists(RESULTS_FILE):
        st.info("Running fresh stress tests...")
        stress_results = run_stress_test(num_episodes=episodes)
        with open(RESULTS_FILE, "w") as f:
            json.dump(stress_results, f, indent=2)
    else:
        with open(RESULTS_FILE) as f:
            stress_results = json.load(f)

    summary_df = pd.DataFrame([
        {
            "Scenario": name,
            "Episodes": m["episodes"],
            "Avg Profit ($)": f"{m['avg_profit']:,.0f}",
            "Avg CVaR ($)": f"{m['avg_cvar']:,.0f}",
            "Compliance Rate": f"{m['compliance_rate']*100:.0f}%"
        }
        for name, m in stress_results.items()
    ])
    st.dataframe(summary_df, use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        compliance_rates = [v["compliance_rate"]*100 for v in stress_results.values()]
        plt.figure(figsize=(5,4))
        plt.bar(stress_results.keys(), compliance_rates, color=["orange","red","purple"])
        plt.ylabel("Compliance Rate (%)")
        plt.title("Compliance Rate Under Stress")
        plt.xticks(rotation=15)
        st.pyplot(plt.gcf())
    with col2:
        profits = [v["avg_profit"] for v in stress_results.values()]
        cvars = [v["avg_cvar"] for v in stress_results.values()]
        plt.figure(figsize=(5,4))
        plt.scatter(cvars, profits, s=120, c=["orange","red","purple"], alpha=0.8)
        for i, name in enumerate(stress_results.keys()):
            plt.annotate(name, (cvars[i], profits[i]+10000), fontsize=8, ha="center")
        plt.xlabel("Avg CVaR (Tail Risk $)")
        plt.ylabel("Avg Profit ($)")
        plt.title("Profit vs Tail-Risk (CVaR) by Scenario")
        st.pyplot(plt.gcf())

    st.markdown("### Scenario Definitions")
    st.table(pd.DataFrame(STRESS_SCENARIOS))
