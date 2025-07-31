"""
reporting.py
------------
Handles PDF and reporting for ClauseLens investor demos:
- Generates detailed PDF reports with clauses, KPIs, and charts
- Supports multi-episode summaries for audit-ready documentation
"""

import os
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from fpdf import FPDF
from src import config


class ReportGenerator:
    def __init__(self):
        self.output_dir = Path(config.REPORTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Text Utilities
    # ---------------------------
    def normalize_text(self, text: str) -> str:
        """Convert Unicode text to ASCII-safe for FPDF."""
        if not text:
            return ""
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    # ---------------------------
    # PDF Export
    # ---------------------------
    def export_pdf(
        self,
        results_list: List[Dict],
        portfolio_summary: Dict,
        portfolio_results: List[Dict],
        chart_path: Optional[str] = None,
        output_filename: str = None,
    ) -> str:
        """
        Generate a full PDF report for all episodes.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title Page
        pdf.set_font("Helvetica", style="B", size=18)
        pdf.cell(200, 12, txt="ClauseLens Investor Demo Report", ln=True, align='C')
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)

        # Portfolio Summary
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="Portfolio Summary", ln=True)
        pdf.set_font("Helvetica", size=12)
        summary_text = (
            f"Episodes: {portfolio_summary.get('episodes', len(results_list))}\n"
            f"Average Profit: ${portfolio_summary.get('avg_profit',0):,.0f}\n"
            f"Average CVaR: ${portfolio_summary.get('avg_cvar',0):,.0f}\n"
            f"Compliance Rate: {portfolio_summary.get('compliance_rate',0)*100:.0f}%"
        )
        pdf.multi_cell(0, 8, self.normalize_text(summary_text))
        pdf.ln(5)

        # Chart
        if chart_path and os.path.exists(chart_path):
            pdf.image(chart_path, x=15, w=180)
            pdf.ln(10)

        # ---------------------------
        # Per-Episode Sections
        # ---------------------------
        for idx, ep in enumerate(results_list, start=1):
            treaty = ep.get("treaty", {})
            winning_bid = ep.get("winning_bid", {})
            kpi = ep.get("kpi", {})
            clauses = ep.get("clauses", [])

            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 10, f"Episode {idx}: {treaty.get('cedent','Unknown')}", ln=True)
            pdf.set_font("Helvetica", size=11)

            # Treaty Overview
            treaty_text = (
                f"Cedent: {treaty.get('cedent','N/A')}\n"
                f"Peril: {treaty.get('peril','N/A')} | LOB: {treaty.get('line_of_business','N/A')} | Region: {treaty.get('region','N/A')}\n"
                f"Exposure: ${treaty.get('exposure',0):,.0f} | Limit: {treaty.get('limit',0):.0%} | QS Cap: {treaty.get('quota_share_cap',0):.0%}\n"
                f"Notes: {treaty.get('notes','None')}\n"
            )
            pdf.multi_cell(0, 7, self.normalize_text(treaty_text))
            pdf.ln(2)

            # Winning Bid + KPIs
            bid_text = (
                f"Winning Bid:\n"
                f"- Quota Share: {winning_bid.get('quota_share',0):.0%}\n"
                f"- Premium: ${winning_bid.get('premium',0):,.0f}\n"
                f"- Expected Loss: ${winning_bid.get('expected_loss',0):,.0f}\n"
                f"- Tail Risk (CVaR): ${winning_bid.get('tail_risk',0):,.0f}\n\n"
                f"KPIs:\n"
                f"- Profit: ${kpi.get('profit',0):,.0f}\n"
                f"- CVaR: ${kpi.get('cvar',0):,.0f}\n"
                f"- Compliance: {'Pass' if kpi.get('regulatory_flags',{}).get('all_ok',False) else 'Fail'}"
            )
            pdf.multi_cell(0, 7, self.normalize_text(bid_text))
            pdf.ln(2)

            # ClauseLens Explanation
            explanation = ep.get("explanation", "No explanation generated.")
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 8, "ClauseLens Explanation:", ln=True)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 7, self.normalize_text(explanation))
            pdf.ln(2)

            # Retrieved Clauses
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 8, "Retrieved Clauses:", ln=True)
            pdf.set_font("Helvetica", size=11)
            if clauses:
                for clause in clauses:
                    clause_text = (
                        f"[Clause {clause.get('id','?')}] ({clause.get('category','General')}) "
                        f"{clause.get('text','')}"
                    )
                    pdf.multi_cell(0, 6, self.normalize_text(clause_text))
            else:
                pdf.multi_cell(0, 6, "No clauses retrieved.")
            pdf.ln(3)

        # ---------------------------
        # Save PDF
        # ---------------------------
        output_filename = output_filename or config.DEFAULT_PDF_NAME
        output_file = self.output_dir / output_filename
        pdf.output(str(output_file))
        return str(output_file)


# ---------------------------
# Quick Test
# ---------------------------
if __name__ == "__main__":
    # Minimal mock for testing
    sample_results = [
        {
            "treaty": {
                "cedent": "ABC Insurance",
                "peril": "Hurricane",
                "line_of_business": "Property Cat XL",
                "region": "Florida",
                "exposure": 5000000,
                "limit": 0.4,
                "quota_share_cap": 0.5,
                "notes": "High-risk Florida portfolio"
            },
            "winning_bid": {
                "quota_share": 0.4,
                "premium": 200000,
                "expected_loss": 150000,
                "tail_risk": 60000
            },
            "kpi": {"profit": 50000, "cvar": 60000, "regulatory_flags": {"all_ok": True}},
            "clauses": [
                {"id": 1, "text": "Solvency II Article 101 requires capital coverage.", "category": "Capital Requirements"}
            ],
            "explanation": "This bid complies with capital requirements and provides margin."
        }
    ]
    portfolio_summary = {"episodes": 1, "avg_profit": 50000, "avg_cvar": 60000, "compliance_rate": 1.0}
    rg = ReportGenerator()
    file_path = rg.export_pdf(sample_results, portfolio_summary, [r["kpi"] for r in sample_results])
    print(f"✅ PDF generated: {file_path}")
