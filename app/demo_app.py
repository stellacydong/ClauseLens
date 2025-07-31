import sys, os, time, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.demo_pipeline import run_demo, run_multi_episode
from src.evaluation import evaluate_bids, summarize_portfolio

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="ClauseLens Demo", layout="wide")

st.title("🧠 ClauseLens + Multi-Agent Treaty Bidding Demo")
st.markdown("""
Welcome to the **ClauseLens Investor Dashboard**.

This demo showcases:

1. **Intelligent Multi-Agent Bidding** with MARL  
2. **ClauseLens Explanations** grounded in regulatory clauses  
3. **Profitability, Tail-Risk (CVaR), and Compliance KPIs**
""")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Treaty Settings")
scenario = st.sidebar.selectbox(
    "Select Demo Scenario",
    ["Sample Treaty 1", "Sample Treaty 2", "Random"]
)
episodes = st.sidebar.slider(
    "Number of Episodes", min_value=1, max_value=20, value=1, step=1
)
run_button = st.sidebar.button("Run Simulation 🚀")

# ---------------------------
# Dashboard Layout Placeholders
# ---------------------------
st.markdown("### 📊 Portfolio Summary KPIs")

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Avg MARL Profit ($)", "—")
kpi_col2.metric("Avg MARL CVaR ($)", "—")
kpi_col3.metric("MARL Compliance Rate", "—")

left_col, right_col = st.columns([2, 3])

with left_col:
    st.markdown("#### Episode Results (Compact)")
    empty_df = pd.DataFrame(columns=[
        "Episode", "MARL Profit ($)", "MARL CVaR ($)", "MARL Comp",
        "Baseline Profit ($)", "Baseline CVaR ($)", "Baseline Comp"
    ])
    episode_table_placeholder = st.dataframe(empty_df, use_container_width=True, height=220)

with right_col:
    st.markdown("#### Profit vs Tail-Risk (CVaR)")
    fig, ax = plt.subplots()
    ax.set_xlabel("CVaR (Tail Risk $)")
    ax.set_ylabel("Profit ($)")
    ax.set_title("Profit vs Tail-Risk Comparison")
    chart_placeholder = st.pyplot(fig)

# ---------------------------
# Run Simulation
# ---------------------------
if run_button:
    st.subheader(f"🔄 Running {episodes} Episode{'s' if episodes>1 else ''}...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    results, portfolio_results_marl, portfolio_results_baseline = run_multi_episode(num_episodes=episodes)

    # Compute summaries
    summary_marl = summarize_portfolio(portfolio_results_marl)
    summary_baseline = summarize_portfolio(portfolio_results_baseline)

    # Update KPI Metrics
    kpi_col1.metric("Avg MARL Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
    kpi_col2.metric("Avg MARL CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
    kpi_col3.metric("MARL Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")

    # Build Episode Results Table
    episodes_df = pd.DataFrame([
        {
            "Episode": i + 1,
            "MARL Profit ($)": f"{marl['profit']:,.0f}",
            "MARL CVaR ($)": f"{marl['cvar']:,.0f}",
            "MARL Comp": "P" if marl['regulatory_flags']['all_ok'] else "F",
            "Baseline Profit ($)": f"{base['profit']:,.0f}",
            "Baseline CVaR ($)": f"{base['cvar']:,.0f}",
            "Baseline Comp": "P" if base['regulatory_flags']['all_ok'] else "F",
        }
        for i, (marl, base) in enumerate(zip(portfolio_results_marl, portfolio_results_baseline))
    ])

    # Update live table
    episode_table_placeholder.dataframe(episodes_df, use_container_width=True, height=220)

    # Build Profit vs CVaR Chart
    marl_profits = [res["profit"] for res in portfolio_results_marl]
    marl_cvar = [res["cvar"] for res in portfolio_results_marl]
    base_profits = [res["profit"] for res in portfolio_results_baseline]
    base_cvar = [res["cvar"] for res in portfolio_results_baseline]

    fig, ax = plt.subplots()
    ax.scatter(base_cvar, base_profits, color="red", label="Baseline")
    ax.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")
    ax.set_xlabel("CVaR (Tail Risk $)")
    ax.set_ylabel("Profit ($)")
    ax.set_title("Profit vs Tail-Risk Comparison")
    ax.legend()
    chart_placeholder.pyplot(fig)

    st.success("✅ Simulation Complete! Review the KPIs, table, and chart above.")
