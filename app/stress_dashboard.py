# app/stress_dashboard.py

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="ClauseLens Stress Test Dashboard", layout="wide")
st.title("⚡ Stress Test Dashboard – ClauseLens Reinsurance Analytics")
st.markdown("""
This dashboard visualizes **catastrophe and capital stress test results**  
for MARL-driven treaty bidding vs Baseline actuarial pricing.
""")

# ---------------------------
# Load Results
# ---------------------------
RESULTS_FILE = os.path.join("experiments", "stress_test_results.json")

if not os.path.exists(RESULTS_FILE):
    st.error(f"Stress test results not found at `{RESULTS_FILE}`. Run `experiments/stress_test.py` first.")
    st.stop()

with open(RESULTS_FILE, "r") as f:
    results_data = json.load(f)

results_df = pd.DataFrame(results_data["results"])
summary_marl = results_data["marl_summary"]
summary_baseline = results_data["baseline_summary"]

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Filters")
selected_scenario = st.sidebar.selectbox(
    "Stress Scenario",
    ["All"] + sorted(results_df["stress_scenario"].unique())
)
selected_metric = st.sidebar.selectbox(
    "Metric to Visualize",
    ["Profit ($)", "CVaR ($)", "Compliance Rate (%)"]
)

# Apply filter
if selected_scenario != "All":
    filtered_df = results_df[results_df["stress_scenario"] == selected_scenario]
else:
    filtered_df = results_df.copy()

# ---------------------------
# KPI Summary
# ---------------------------
st.markdown("### 📊 Portfolio KPI Summary")

col1, col2 = st.columns(2)
with col1:
    st.subheader("MARL Agents")
    st.metric("Avg Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
    st.metric("Avg CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
    st.metric("Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")

with col2:
    st.subheader("Baseline Pricing")
    st.metric("Avg Profit ($)", f"{summary_baseline['avg_profit']:,.0f}")
    st.metric("Avg CVaR ($)", f"{summary_baseline['avg_cvar']:,.0f}")
    st.metric("Compliance Rate", f"{summary_baseline['compliance_rate']*100:.0f}%")

# ---------------------------
# Episode-Level Results Table
# ---------------------------
st.markdown("### 📄 Episode Results Table (Filtered)")
st.dataframe(filtered_df, use_container_width=True, height=300)

# ---------------------------
# Visualization
# ---------------------------
st.markdown("### 📈 Profit vs Tail-Risk (CVaR) Scatter Plot")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    filtered_df["marl_cvar"], filtered_df["marl_profit"],
    color="green", alpha=0.6, label="MARL Agent"
)
ax.scatter(
    filtered_df["baseline_cvar"], filtered_df["baseline_profit"],
    color="red", alpha=0.6, label="Baseline"
)
ax.set_xlabel("CVaR (Tail Risk $)")
ax.set_ylabel("Profit ($)")
ax.set_title(f"Profit vs CVaR – {selected_scenario if selected_scenario!='All' else 'All Scenarios'}")
ax.legend()
st.pyplot(fig)

# ---------------------------
# Scenario Performance Bar Chart
# ---------------------------
st.markdown("### 📊 Scenario Performance Comparison")

scenario_summary = (
    filtered_df.groupby("stress_scenario")
    .agg({
        "marl_profit": "mean",
        "marl_cvar": "mean",
        "baseline_profit": "mean",
        "baseline_cvar": "mean"
    })
    .reset_index()
)

fig2, ax2 = plt.subplots(figsize=(8, 4))
width = 0.35
x = range(len(scenario_summary))
ax2.bar(x, scenario_summary["marl_profit"], width, label="MARL Profit", color="green", alpha=0.7)
ax2.bar([p + width for p in x], scenario_summary["baseline_profit"], width, label="Baseline Profit", color="red", alpha=0.7)
ax2.set_xticks([p + width/2 for p in x])
ax2.set_xticklabels(scenario_summary["stress_scenario"], rotation=45, ha="right")
ax2.set_ylabel("Profit ($)")
ax2.set_title("Average Profit by Stress Scenario")
ax2.legend()
st.pyplot(fig2)

# ---------------------------
# Export Filtered Results
# ---------------------------
st.markdown("### 📥 Export Filtered Results")
st.download_button(
    "Download Filtered Results as CSV",
    filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="stress_test_filtered_results.csv",
    mime="text/csv"
)
