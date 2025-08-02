import os
import streamlit as st
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Import tab components
from components import clause_tab, marketlens_tab, bidding_tab, governance_tab

# -----------------------------------------------------------------------------
# Paths & Asset Setup
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(APP_DIR, "assets")

favicon_path = os.path.join(ASSETS_DIR, "favicon.ico")
logo_path = os.path.join(ASSETS_DIR, "logo.png")
background_path = os.path.join(ASSETS_DIR, "background.jpg")
hero_banner_path = os.path.join(ASSETS_DIR, "hero_banner.jpg")

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transparent Market Platform",
    page_icon=favicon_path,
    layout="wide"
)

# -----------------------------------------------------------------------------
# Auto-refresh every 5 seconds for live KPIs
# -----------------------------------------------------------------------------
refresh_interval_sec = 5
st_autorefresh(interval=refresh_interval_sec * 1000, key="market_refresh")

# -----------------------------------------------------------------------------
# Sidebar Branding
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)  # <-- fixed here
    else:
        st.write("🏦 Transparent Market Platform")

    st.markdown("---")
    st.subheader("Navigation")
    st.write("Use the tabs above to explore the platform features.")
    st.markdown("---")
    st.caption("© 2025 Reinsurance Analytics · YC Demo")

# -----------------------------------------------------------------------------
# Hero Section (Banner + Background)
# -----------------------------------------------------------------------------


## Hero Banner (Full-width)
#if os.path.exists(hero_banner_path):
#    st.image(hero_banner_path, use_container_width=True)  # <-- fixed here
#st.markdown("---")


# -----------------------------------------------------------------------------
# Initialize Persistent KPI Simulation
# -----------------------------------------------------------------------------
if "kpi_values" not in st.session_state:
    st.session_state.kpi_values = {
        "Net Profit ($)": 1_000_000.0,
        "CVaR (95%)": 2_500_000.0,
        "Fairness Score": 0.85,
        "Bid Win Rate (%)": 55.0
    }

higher_is_better = {
    "Net Profit ($)": True,
    "CVaR (95%)": False,
    "Fairness Score": True,
    "Bid Win Rate (%)": True
}

metrics = {}
for label, value in st.session_state.kpi_values.items():
    drift = np.random.normal(0, value * 0.002)  # 0.2% drift per refresh
    new_value = max(0, value + drift)
    st.session_state.kpi_values[label] = new_value
    metrics[label] = new_value

# -----------------------------------------------------------------------------
# KPI Cards with Color Coding and Trend Arrows
# -----------------------------------------------------------------------------
st.subheader("📊 Live Market KPIs")
cols = st.columns(len(metrics))

for (label, live_value), col in zip(metrics.items(), cols):
    last_value = live_value / (1 + np.random.normal(0, 0.002))
    drift = live_value - last_value
    trend_up = drift >= 0
    is_good = (trend_up and higher_is_better[label]) or (not trend_up and not higher_is_better[label])
    arrow = "⬆️" if trend_up else "⬇️"
    color = "green" if is_good else "red"

    if "($" in label:
        display_value = f"${live_value:,.0f}"
    elif "%" in label:
        display_value = f"{live_value:.1f}%"
    else:
        display_value = f"{live_value:.2f}"

    col.markdown(
        f"""
        <div style="background-color:#f9f9f9;border-radius:8px;padding:12px;text-align:center;
                    box-shadow:0 1px 3px rgba(0,0,0,0.1);">
            <div style="font-size:1.1em;font-weight:600;">{label}</div>
            <div style="font-size:1.5em;color:{color};margin:4px 0;">{arrow} {display_value}</div>
            <div style="font-size:0.9em;color:gray;">Updated {datetime.now().strftime('%H:%M:%S')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Multi-Tab Navigation
# -----------------------------------------------------------------------------
tabs = st.tabs(["📄 ClauseLens", "📈 MarketLens", "🤖 Bidding", "⚖️ Governance"])

with tabs[0]:
    clause_tab.render()

with tabs[1]:
    marketlens_tab.render()

with tabs[2]:
    bidding_tab.render()

with tabs[3]:
    governance_tab.render()
