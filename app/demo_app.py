# kpi_dashboard.py

import time
import streamlit as st
import numpy as np
import pandas as pd
from plotly import graph_objects as go

# Local imports from your project
from simulate_env import run_simulation
from portfolio_summary import compute_portfolio_metrics
from report_generator import generate_report_pdf

st.set_page_config(page_title="ClauseLens Demo Dashboard", layout="wide")

st.title("ClauseLens Demo Dashboard")
st.write("This dashboard animates MARL metrics for treaty pricing simulation.")

# --- Animated Metric Function ---
def animate_metric(container, start_value, end_value, label, fmt="{:,.2f}", duration=1.5):
    steps = 20
    for i in range(steps + 1):
        interpolated = start_value + (end_value - start_value) * (i / steps)
        container.metric(label, fmt.format(interpolated))
        time.sleep(duration / steps)

# --- Gauge Chart for Clause Compliance ---
def compliance_gauge(value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Clause Compliance (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=20, b=0, l=0, r=0))
    return fig

# --- Initialize columns ---
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

if "prev_metrics" not in st.session_state:
    st.session_state.prev_metrics = {
        "profit": 100_000,
        "cvar": 0.25,
        "compliance": 85.0
    }

# --- Button to run simulation ---
if st.button("Run Simulation & Update Metrics"):

    # 1. Run your MARL simulation
    sim_results = run_simulation(episodes=100)
    
    # 2. Compute portfolio metrics
    metrics = compute_portfolio_metrics(sim_results)
    new_profit = metrics["avg_profit"]
    new_cvar = metrics["portfolio_cvar"]
    new_compliance = metrics["clause_compliance"]

    # 3. Animate metrics
    animate_metric(kpi_col1, st.session_state.prev_metrics["profit"], new_profit, "Avg MARL Profit ($)")
    animate_metric(kpi_col2, st.session_state.prev_metrics["cvar"], new_cvar, "Portfolio CVaR", "{:.2f}")
    
    # Compliance metric with trend arrow
    delta = new_compliance - st.session_state.prev_metrics["compliance"]
    arrow = "▲" if delta >= 0 else "▼"
    color = "green" if delta >= 0 else "red"
    kpi_col3.markdown(f"<h3>Clause Compliance (%)</h3><h1 style='color:{color}'>{new_compliance:.2f} {arrow}</h1>", unsafe_allow_html=True)
    
    # 4. Gauge chart
    st.plotly_chart(compliance_gauge(new_compliance), use_container_width=True)

    # 5. Save state
    st.session_state.prev_metrics = {
        "profit": new_profit,
        "cvar": new_cvar,
        "compliance": new_compliance
    }

    # 6. Generate PDF report
    pdf_path = generate_report_pdf(metrics)
    st.download_button("Download Report", data=open(pdf_path, "rb"), file_name="ClauseLens_Report.pdf")

