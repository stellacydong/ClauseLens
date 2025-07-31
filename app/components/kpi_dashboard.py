# app/components/kpi_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

class KPIDashboard:
    def __init__(self):
        """
        Simple component for displaying KPIs and episode results in a clean dashboard.
        """
        pass

    def display_summary(self, summary_marl, summary_baseline):
        """
        Display portfolio-level KPIs for MARL and Baseline agents.
        """
        st.markdown("### 📊 Portfolio KPIs Overview")

        tab1, tab2 = st.tabs(["MARL Agents", "Baseline Pricing"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
            col2.metric("Avg CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
            col3.metric("Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")

        with tab2:
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Profit ($)", f"{summary_baseline['avg_profit']:,.0f}")
            col2.metric("Avg CVaR ($)", f"{summary_baseline['avg_cvar']:,.0f}")
            col3.metric("Compliance Rate", f"{summary_baseline['compliance_rate']*100:.0f}%")

    def display_episode_table(self, episodes_df):
        """
        Display episode-level results as a compact table.
        """
        st.markdown("### 📄 Episode Results Table (MARL vs Baseline)")
        st.dataframe(episodes_df, use_container_width=True, height=250)

    def display_profit_vs_cvar(self, portfolio_results_marl, portfolio_results_baseline):
        """
        Plot Profit vs CVaR (Tail Risk) comparison.
        """
        st.markdown("### 📈 Profit vs Tail-Risk (CVaR)")

        marl_profits = [res["profit"] for res in portfolio_results_marl]
        marl_cvar = [res["cvar"] for res in portfolio_results_marl]
        base_profits = [res["profit"] for res in portfolio_results_baseline]
        base_cvar = [res["cvar"] for res in portfolio_results_baseline]

        fig, ax = plt.subplots()
        ax.scatter(base_cvar, base_profits, color="red", label="Baseline")
        ax.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")
        ax.set_xlabel("CVaR (Tail Risk $)")
        ax.set_ylabel("Profit ($)")
        ax.set_title("Profit vs Tail-Risk Comparison")
        ax.legend()
        st.pyplot(fig)

    def render_dashboard(self, summary_marl, summary_baseline, episodes_df,
                         portfolio_results_marl, portfolio_results_baseline):
        """
        High-level method to render the full KPI dashboard in tabs.
        """
        self.display_summary(summary_marl, summary_baseline)
        self.display_episode_table(episodes_df)
        self.display_profit_vs_cvar(portfolio_results_marl, portfolio_results_baseline)
