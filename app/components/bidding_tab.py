import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Self-contained PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "demo")


def load_dashboard_data():
    files = {
        "kpis": "dashboard_kpis.csv",
        "trends": "dashboard_trends.csv",
        "bids": "dashboard_bids.csv"
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(DATA_DEMO_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    return data


def animate_metric(placeholder, label, start_value, end_value, steps=30, delay=0.05):
    values = np.linspace(start_value, end_value, steps)
    is_increase = end_value >= start_value
    delta_color = "normal" if is_increase else "inverse"

    for val in values:
        arrow = "▲" if is_increase else "▼"
        placeholder.metric(
            label,
            f"{val:,.2f}",
            f"{arrow} {abs(end_value-start_value):,.2f}",
            delta_color=delta_color
        )
        time.sleep(delay)


def render_bidding_tab():
    st.subheader("Live Multi-Agent Treaty Bidding Simulation")
    st.info("Click to animate the latest KPIs and compliance gauge.")

    if st.button("⚡ Animate Live Market KPIs"):
        data = load_dashboard_data()
        if "kpis" in data and "bids" in data:
            kpi_row = data["kpis"].tail(1).iloc[0]

            col1, col2, col3 = st.columns(3)
            animate_metric(col1, "Avg MARL Profit ($)", 0, kpi_row["avg_profit"])
            animate_metric(col2, "Portfolio CVaR 95%", 0, kpi_row["avg_cvar"])
            animate_metric(col3, "Clause Compliance (%)", 0, kpi_row["avg_compliance"] * 100)

            if "trends" in data:
                st.markdown("### KPI Trends")
                fig = px.line(data["trends"], x="episode", y=["avg_profit", "avg_compliance"])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Recent Bids")
            st.dataframe(data["bids"].tail(20), use_container_width=True)
        else:
            st.warning("Run `run_simulation.py` and `generate_dashboard_data.py` first.")
