import streamlit as st
from utils.load_data import load_dashboard_data
from components.marketlens_tab import render_marketlens_tab

st.set_page_config(page_title="Market Overview", page_icon="📊", layout="wide")

st.title("📊 Market Overview")
st.write("This page provides KPIs and MarketLens benchmarking insights.")

# Load data
data = load_dashboard_data()
kpis_df = data["kpis"]
features_df = data["marketlens_features"]
labels_df = data["marketlens_labels"]

# Render the MarketLens tab (reusing component)
render_marketlens_tab(kpis_df, features_df, labels_df)
