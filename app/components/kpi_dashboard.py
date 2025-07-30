import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def display_portfolio_kpis(summary_marl, summary_baseline=None, title="📊 Portfolio Summary KPIs"):
    """
    Display portfolio-level KPIs for MARL (and optionally Baseline).
    """
    st.markdown(f"## {title}")

    if summary_baseline:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("MARL Avg Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
        col2.metric("MARL Avg CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
        col3.metric("MARL Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")
        col4.metric("Base Avg Profit ($)", f"{summary_baseline['avg_profit']:,.0f}")
        col5.metric("Base Avg CVaR ($)", f"{summary_baseline['avg_cvar']:,.0f}")
        col6.metric("Base Compliance Rate", f"{summary_baseline['compliance_rate']*100:.0f}%")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Profit ($)", f"{summary_marl['avg_profit']:,.0f}")
        col2.metric("Avg CVaR ($)", f"{summary_marl['avg_cvar']:,.0f}")
        col3.metric("Compliance Rate", f"{summary_marl['compliance_rate']*100:.0f}%")


def display_episode_table(portfolio_results_marl, portfolio_results_baseline=None):
    """
    Show a compact episode results table for all episodes.
    """
    episodes_data = []
    for i, marl_kpi in enumerate(portfolio_results_marl, 1):
        row = {
            "Episode": i,
            "MARL Profit ($)": f"{marl_kpi['profit']:,.0f}",
            "MARL CVaR ($)": f"{marl_kpi['cvar']:,.0f}",
            "MARL Comp": "P" if marl_kpi['regulatory_flags']['all_ok'] else "F",
        }
        if portfolio_results_baseline:
            base_kpi = portfolio_results_baseline[i-1]
            row.update({
                "Base Profit ($)": f"{base_kpi['profit']:,.0f}",
                "Base CVaR ($)": f"{base_kpi['cvar']:,.0f}",
                "Base Comp": "P" if base_kpi['regulatory_flags']['all_ok'] else "F",
            })
        episodes_data.append(row)

    df = pd.DataFrame(episodes_data)
    st.markdown("### Episode Results (Compact)")
    st.dataframe(df, use_container_width=True, height=220)


def display_profit_vs_cvar_chart(portfolio_results_marl, portfolio_results_baseline=None):
    """
    Display Profit vs CVaR chart for MARL and optionally Baseline.
    """
    marl_profits = [res["profit"] for res in portfolio_results_marl]
    marl_cvar = [res["cvar"] for res in portfolio_results_marl]

    plt.figure(figsize=(6, 4))
    plt.scatter(marl_cvar, marl_profits, color="green", label="MARL Agent")

    if portfolio_results_baseline:
        base_profits = [res["profit"] for res in portfolio_results_baseline]
        base_cvar = [res["cvar"] for res in portfolio_results_baseline]
        plt.scatter(base_cvar, base_profits, color="red", label="Baseline")

    plt.xlabel("CVaR (Tail Risk $)")
    plt.ylabel("Profit ($)")
    plt.title("Profit vs Tail-Risk (CVaR)")
    plt.legend()
    st.pyplot(plt.gcf())

