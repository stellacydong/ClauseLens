# app/components/portfolio_summary.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


class PortfolioSummary:
    """
    Component to render portfolio-level summaries including KPIs,
    multi-episode performance charts, and compliance stats.
    """

    def __init__(self):
        pass

    def render_kpis(self, summary_marl, summary_baseline):
        """
        Render top-level KPIs for MARL agents and Baseline pricing side-by-side.
        """
        st.markdown("### 📊 Portfolio Summary KPIs")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MARL Agents")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
            c2.metric("Avg CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
            c3.metric("Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")

        with col2:
            st.subheader("Baseline Pricing")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Profit ($)", f"{summary_baseline['avg_profit']:,.0f}")
            c2.metric("Avg CVaR ($)", f"{summary_baseline['avg_cvar']:,.0f}")
            c3.metric("Compliance Rate", f"{summary_baseline['compliance_rate']*100:.0f}%")

    def render_episode_table(self, episodes_df: pd.DataFrame):
        """
        Render a compact table of episode-level results.
        """
        st.markdown("### 📄 Episode Results Table")
        st.dataframe(episodes_df, use_container_width=True, height=260)

    def render_profit_vs_cvar(self, portfolio_results_marl, portfolio_results_baseline):
        """
        Render a Profit vs CVaR (Tail Risk) scatter plot.
        """
        st.markdown("### 📈 Profit vs Tail-Risk (CVaR)")

        marl_profits = [res["profit"] for res in portfolio_results_marl]
        marl_cvar = [res["cvar"] for res in portfolio_results_marl]
        base_profits = [res["profit"] for res in portfolio_results_baseline]
        base_cvar = [res["cvar"] for res in portfolio_results_baseline]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(base_cvar, base_profits, color="red", label="Baseline")
        ax.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")
        ax.set_xlabel("CVaR (Tail Risk $)")
        ax.set_ylabel("Profit ($)")
        ax.set_title("Profit vs Tail-Risk Comparison")
        ax.legend()
        st.pyplot(fig)

    def render_full_summary(
        self,
        summary_marl,
        summary_baseline,
        episodes_df: pd.DataFrame,
        portfolio_results_marl,
        portfolio_results_baseline,
    ):
        """
        High-level method to render the entire portfolio summary section.
        """
        self.render_kpis(summary_marl, summary_baseline)
        self.render_episode_table(episodes_df)
        self.render_profit_vs_cvar(portfolio_results_marl, portfolio_results_baseline)
