# ClauseLens: Clause-Grounded, Risk-Constrained Quoting

This repository contains the LaTeX source, code, and supporting materials for:

**ClauseLens: Clause-Grounded, Risk-Constrained Quoting for Trustworthy Reinsurance AI**

Stella C. Dong* and James R. Finlay

---

## 📄 Paper

- [PDF](paper/v3.pdf)
- [LaTeX Source](paper/v3.tex)

## 🏗 Repository Contents

- `paper/` – LaTeX source and figures for the ICAIF 2025 submission.
- `src/` – Core Python code for:
  - Clause retrieval and embedding
  - RA-CMDP reinforcement learning agent
  - Natural language explanation generation
  - Training and evaluation pipelines
- `data/` – Sample legal clauses and synthetic treaty environments.
- `experiments/` – Configs, logs, and results for reproducibility.

## ⚙️ Setup

```bash
git clone https://github.com/<username>/ClauseLens.git
cd ClauseLens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


ClauseLens/
├── README.md
├── LICENSE
├── .gitignore
├── paper/
│   ├── v3.pdf                # Final compiled PDF
│   ├── v3.tex                # Main LaTeX source
│   ├── sections/             # (Optional) Split LaTeX sections for readability
│   ├── figures/              # All figures in .pdf/.png/.tikz
│   ├── tables/               # (Optional) CSVs or LaTeX table files
│   ├── references.bib        # BibTeX file
│   └── macros.tex            # (Optional) custom LaTeX macros
├── src/
│   ├── retrieval_pipeline.py  # Clause retrieval and embedding
│   ├── ra_cmdp_agent.py       # RA-CMDP reinforcement learning agent
│   ├── explanation_generator.py # Natural language rationale module
│   ├── training_pipeline.py   # Full training loop with dual-projected PPO
│   └── evaluation.py          # Evaluation metrics: CVaR, BLEU, clause recall
├── data/
│   ├── clauses/               # Legal/regulatory clause corpus (if allowed)
│   ├── sample_envs/           # Simulated cedent & treaty JSONs
│   └── README.md
├── experiments/
│   ├── configs/               # YAML/JSON config files for runs
│   ├── logs/                  # Training logs
│   └── results/               # Evaluation outputs
└── requirements.txt           # Python dependencies
