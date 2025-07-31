# app/report_generator.py

import os
import unicodedata
from fpdf import FPDF

class ReportGenerator:
    """
    Generates a full PDF report for ClauseLens investor demo:
    - Treaty overview
    - Winning bid details
    - KPIs and compliance flags
    - ClauseLens explanation and retrieved clauses
    - Episode table summary
    - Portfolio summary chart
    """

    def __init__(self, output_file="ClauseLens_Demo_Report.pdf"):
        self.output_file = output_file

    @staticmethod
    def normalize_text(text: str) -> str:
        """Convert text to ASCII-safe for FPDF"""
        if not text:
            return ""
        return (
            unicodedata.normalize("NFKD", str(text))
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    def generate(
        self,
        treaty,
        winning_bid,
        kpi,
        clauses,
        episodes_data=None,
        portfolio_summary=None,
        chart_path=None
    ):
        """
        Create the full PDF report.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ---------------------------
        # Title
        # ---------------------------
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(200, 12, txt="ClauseLens Investor Report", ln=True, align='C')
        pdf.ln(5)

        # ---------------------------
        # I. Treaty Overview
        # ---------------------------
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="I. Treaty Overview", ln=True)
        pdf.set_font("Helvetica", size=12)
        treaty_text = (
            f"Cedent: {treaty.get('cedent', 'N/A')}\n"
            f"Peril: {treaty.get('peril', 'N/A')}\n"
            f"Region: {treaty.get('region', 'N/A')}\n"
            f"Line of Business: {treaty.get('line_of_business', 'N/A')}\n"
            f"Exposure: ${treaty.get('exposure', 0):,.0f}\n"
            f"Limit: {treaty.get('limit', 0):.0%}\n"
            f"Quota Share Cap: {treaty.get('quota_share_cap', 0):.0%}\n"
            f"Notes: {treaty.get('notes', 'None')}"
        )
        pdf.multi_cell(0, 8, self.normalize_text(treaty_text))
        pdf.ln(3)

        # ---------------------------
        # II. Winning Bid
        # ---------------------------
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="II. Winning Bid (MARL Agent)", ln=True)
        pdf.set_font("Helvetica", size=12)
        bid_text = (
            f"Quota Share: {winning_bid.get('quota_share',0):.0%}\n"
            f"Premium: ${winning_bid.get('premium',0):,.0f}\n"
            f"Expected Loss: ${winning_bid.get('expected_loss',0):,.0f}\n"
            f"Tail Risk (CVaR): ${winning_bid.get('tail_risk',0):,.0f}"
        )
        pdf.multi_cell(0, 8, self.normalize_text(bid_text))
        pdf.ln(3)

        # ---------------------------
        # III. Key Performance Indicators
        # ---------------------------
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="III. Key Performance Indicators", ln=True)
        pdf.set_font("Helvetica", size=12)
        compliance = kpi.get("regulatory_flags", {})
        kpi_text = (
            f"Profit: ${kpi.get('profit',0):,.0f}\n"
            f"CVaR (Tail Risk): ${kpi.get('cvar',0):,.0f}\n"
            f"Regulatory Compliance: {'Pass' if compliance.get('all_ok', False) else 'Fail'}\n"
            f" - Quota Share OK: {compliance.get('quota_share_ok', False)}\n"
            f" - Premium OK: {compliance.get('premium_ok', False)}\n"
            f" - Tail Risk OK: {compliance.get('tail_risk_ok', False)}"
        )
        pdf.multi_cell(0, 8, self.normalize_text(kpi_text))
        pdf.ln(3)

        # ---------------------------
        # IV. ClauseLens Explanation
        # ---------------------------
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="IV. ClauseLens Explanation", ln=True)
        pdf.set_font("Helvetica", size=12)
        explanation_text = kpi.get("explanation", "This treaty aligns with key regulatory and risk requirements.")
        pdf.multi_cell(0, 8, self.normalize_text(explanation_text))
        pdf.ln(3)

        # ---------------------------
        # V. Retrieved Clauses
        # ---------------------------
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="V. Retrieved Clauses", ln=True)
        pdf.set_font("Helvetica", size=12)
        if clauses:
            for i, clause in enumerate(clauses, start=1):
                clause_line = f"{i}. [{clause['category']}] {clause['text']} (Jurisdiction: {clause['jurisdiction']})"
                pdf.multi_cell(0, 8, self.normalize_text(clause_line))
                pdf.ln(1)
        else:
            pdf.multi_cell(0, 8, "No clauses retrieved.")
        pdf.ln(3)

        # ---------------------------
        # VI. Portfolio Summary
        # ---------------------------
        if portfolio_summary:
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(200, 10, txt="VI. Portfolio Summary", ln=True)
            pdf.set_font("Helvetica", size=12)
            summary_text = (
                f"Episodes: {portfolio_summary.get('episodes', 1)}\n"
                f"Average Profit: ${portfolio_summary.get('avg_profit',0):,.0f}\n"
                f"Average CVaR: ${portfolio_summary.get('avg_cvar',0):,.0f}\n"
                f"Compliance Rate: {portfolio_summary.get('compliance_rate',0)*100:.0f}%"
            )
            pdf.multi_cell(0, 8, self.normalize_text(summary_text))
            pdf.ln(3)

        # ---------------------------
        # VII. Episode Table (Optional)
        # ---------------------------
        if episodes_data and len(episodes_data) > 1:
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(200, 10, txt="VII. Episode Results Summary", ln=True)
            pdf.set_font("Helvetica", size=10)

            headers = [
                "Ep", "MARL Profit", "MARL CVaR", "M Comp",
                "Baseline Profit", "Baseline CVaR", "B Comp"
            ]
            widths = [8, 28, 28, 15, 28, 28, 15]
            for h, w in zip(headers, widths):
                pdf.cell(w, 8, h, border=1)
            pdf.ln()

            for i, row in enumerate(episodes_data, start=1):
                marl_ep, base_ep = row
                table_row = [
                    str(i),
                    f"{marl_ep['profit']:,.0f}",
                    f"{marl_ep['cvar']:,.0f}",
                    "P" if marl_ep['regulatory_flags']['all_ok'] else "F",
                    f"{base_ep['profit']:,.0f}",
                    f"{base_ep['cvar']:,.0f}",
                    "P" if base_ep['regulatory_flags']['all_ok'] else "F",
                ]
                for val, w in zip(table_row, widths):
                    pdf.cell(w, 8, val, border=1)
                pdf.ln()

        # ---------------------------
        # VIII. Profit vs CVaR Chart
        # ---------------------------
        if chart_path and os.path.exists(chart_path):
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(200, 10, txt="Profit vs Tail-Risk (CVaR)", ln=True, align='C')
            pdf.image(chart_path, x=10, y=30, w=180)

        # ---------------------------
        # Save File
        # ---------------------------
        pdf.output(self.output_file)
        return self.output_file
