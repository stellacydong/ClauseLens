import os
import sys
import importlib
import base64
import streamlit as st

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Transparent Market Platform – YC Demo", layout="wide")

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

COMPONENTS_DIR = os.path.join(PROJECT_ROOT, "app", "components")
if COMPONENTS_DIR not in sys.path:
    sys.path.insert(0, COMPONENTS_DIR)

ASSETS_DIR = os.path.join(PROJECT_ROOT, "app", "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

# -----------------------------
# Load & Display Professional Header
# -----------------------------
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_data = f.read()
    encoded_logo = base64.b64encode(logo_data).decode()
    logo_html = f'<img src="data:image/png;base64,{encoded_logo}" style="height:80px;" alt="Logo">'
else:
    logo_html = '<span style="font-size:72px;">🌐</span>'

st.markdown(f"""
    <div style="text-align:center; padding-top: 1rem; padding-bottom: 1rem;">
        {logo_html}
        <h1 style="margin-bottom:0.2rem;">Transparent Market Platform</h1>
        <p style="font-size:1.1rem; color:gray;">
            Clause-grounded • Risk-constrained • YC-ready Reinsurance Intelligence
        </p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Dynamic Tab Imports (Hot Reload)
# -----------------------------
import components.bidding_tab as bidding_tab
import components.clause_tab as clause_tab
import components.marketlens_tab as marketlens_tab
import components.governance_tab as governance_tab

def hot_reload_module(module):
    """Reloads a tab module dynamically for live editing."""
    try:
        return importlib.reload(module)
    except Exception as e:
        st.error(f"Failed to reload module {module.__name__}: {e}")
        return module

# -----------------------------
# Streamlit Tabs
# -----------------------------
tabs = st.tabs(["Live Market", "ClauseLens", "MarketLens", "Governance"])

with tabs[0]:
    bidding_tab = hot_reload_module(bidding_tab)
    bidding_tab.render_bidding_tab()

with tabs[1]:
    clause_tab = hot_reload_module(clause_tab)
    clause_tab.render_clause_tab()

with tabs[2]:
    marketlens_tab = hot_reload_module(marketlens_tab)
    marketlens_tab.render_marketlens_tab()

with tabs[3]:
    governance_tab = hot_reload_module(governance_tab)
    governance_tab.render_governance_tab()

# -----------------------------
# Demo Mode Footer
# -----------------------------
st.markdown(
    """
    <hr style="margin-top:2rem; margin-bottom:0.5rem;">
    <p style="text-align:center; font-size:0.9rem; color:gray;">
        🔄 Hot‑reload enabled – edit any module in <code>app/components/</code> and refresh to see updates instantly.
    </p>
    """,
    unsafe_allow_html=True
)
