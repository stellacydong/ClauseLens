import streamlit as st
from components.governance_tab import render_governance_tab

st.set_page_config(page_title="Governance Tools", page_icon="⚖️", layout="wide")

st.title("⚖️ Governance & Oversight Tools")
st.write("Audit, override, and visualize decision traces for human-in-the-loop control.")

# Render Governance Tools tab (no heavy data needed)
render_governance_tab()
