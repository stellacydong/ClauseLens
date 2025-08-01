import os
import pandas as pd
import streamlit as st

# Self-contained PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "demo")
MARKETLENS_DIR = os.path.join(PROJECT_ROOT, "marketlens", "models")


def render_marketlens_tab():
    st.subheader("Market Benchmarking & Fairness Audits")
    st.info("Click to load MarketLens demo data.")

    if st.button("⚡ Load MarketLens"):
        features_path = os.path.join(DATA_DEMO_DIR, "sample_marketlens.parquet")
        labels_path = os.path.join(DATA_DEMO_DIR, "sample_marketlens_labels.parquet")
        shap_plot = os.path.join(MARKETLENS_DIR, "shap_summary.png")
        fairness_csv = os.path.join(MARKETLENS_DIR, "fairness_audit.csv")

        # Sample features & labels
        if os.path.exists(features_path) and os.path.exists(labels_path):
            st.markdown("### Sample Features")
            st.dataframe(pd.read_parquet(features_path).head(10), use_container_width=True)
            st.markdown("### Sample Labels")
            st.dataframe(pd.read_parquet(labels_path).head(10), use_container_width=True)
        else:
            st.warning("Run `marketlens/preprocess.py` to generate demo data.")

        # SHAP Summary Plot
        if os.path.exists(shap_plot):
            st.markdown("### SHAP Feature Importance")
            st.image(shap_plot, use_column_width=True)
        else:
            st.info("Run `marketlens/fairness_audit.py` to generate SHAP summary.")

        # Fairness Audit Table
        if os.path.exists(fairness_csv):
            st.markdown("### Fairness Audit Results")
            fairness_df = pd.read_csv(fairness_csv)
            st.dataframe(fairness_df, use_container_width=True)
        else:
            st.info("Run `marketlens/fairness_audit.py` to generate fairness metrics.")
