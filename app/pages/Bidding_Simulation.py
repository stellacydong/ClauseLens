import streamlit as st
from utils.load_data import load_kpis
from components.bidding_tab import render_bidding_tab

st.set_page_config(page_title="Bidding Simulation", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Treaty Bidding Simulation")
st.write("Run PPO/MAPPO simulations and visualize performance KPIs.")

# Load MARL simulation KPIs
kpis_df = load_kpis()

# Render Bidding Tab component
render_bidding_tab(kpis_df)
