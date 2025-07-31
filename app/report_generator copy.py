import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.stress_test import run_stress_test, STRESS_SCENARIOS

RESULTS_FILE = "experiments/stress_test_results.json"

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="ClauseLens Stress Test Dashboard", layout="wide")
st.title("⚡ ClauseLens Stress Test Dashboard")

st.markdown("""
This dashboard visualizes **catastrophe & capital stress tests** for ClauseLens MARL agents.

- **High‑Cat Frequency:** Simulates spike in catastrophe claims  
- **Capital Reduction:** 20% lower solvency buffer, stricter compliance  
- **Severe Tail Scenario:** Extreme tail risk and capital stress  

The dashboard highlights **Profit, CVaR, and Compliance** under each scenario.
""")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Stress Test Controls")
rerun = st.sidebar.button("🔄 Run New Stress Test")
episodes = st.sidebar.slider("Episodes per Scenario", 10, 100, 50)

# ---------------------------
# Run or Load Stress Test
# ---------------------------
if rerun or not os.path.exists(RESULTS_FILE):
    st.info("Running fresh stress tests...")
    results = run_stress_test(num_episodes=episodes)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
else:
    st.success(f"Loaded results from {RESULTS_FILE}")
    with open(RESULTS_FILE) as f:
        results = json.load(f)

# ---------------------------
# Display Results Table
# ---------------------------
st.markdown("## 📊 Stress Test Results Summary")

summary_data = []
for scenario_name, metrics in results.items():
    summary_data.append({
        "Scenario": scenario_name,
        "Episodes": metrics["episodes"],
        "Avg Profit ($)": f"{metrics['avg_profit']:,.0f}",
        "Avg CVaR ($)": f"{metrics['avg_cvar']:,.0f}",
        "Compliance Rate": f"{metrics['compliance_rate']*100:.0f}%",
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# ---------------------------
# Visual Charts
# ---------------------------
st.markdown("## 📈 Stress Test Visualizations")

col1, col2 = st.columns(2)

# --- Compliance Bar Chart ---
with col1:
    compliance_rates = [v["compliance_rate"]*100 for v in results.values()]
    plt.figure(figsize=(5,4))
    plt.bar(results.keys(), compliance_rates, color=["orange", "red", "purple"])
    plt.ylabel("Compliance Rate (%)")
    plt.title("Compliance Rate Under Stress")
    plt.xticks(rotation=15)
    st.pyplot(plt.gcf())

# --- Profit vs CVaR Scatter ---
with col2:
    profits = [v["avg_profit"] for v in results.values()]
    cvars = [v["avg_cvar"] for v in results.values()]

    plt.figure(figsize=(5,4))
    plt.scatter(cvars, profits, s=120, c=["orange","red","purple"], alpha=0.8)
    for i, name in enumerate(results.keys()):
        plt.annotate(name, (cvars[i], profits[i]+10000), fontsize=8, ha="center")
    plt.xlabel("Avg CVaR (Tail Risk $)")
    plt.ylabel("Avg Profit ($)")
    plt.title("Profit vs Tail-Risk (CVaR) by Scenario")
    st.pyplot(plt.gcf())

# ---------------------------
# Scenario Details
# ---------------------------
st.markdown("## 🧐 Scenario Definitions")
st.table(pd.DataFrame(STRESS_SCENARIOS))
