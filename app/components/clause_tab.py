import streamlit as st
import pandas as pd
import os

LEGAL_CORPUS_PATH = os.path.join("..", "clauselens", "legal_corpus", "clauses.csv")

def render():
    st.header("📄 ClauseLens: Clause-Grounded Insights")

    st.write(
        """
        ClauseLens retrieves and analyzes treaty clauses for interpretability and governance.
        Use the search to explore your legal corpus or visualize embeddings.
        """
    )

    # Search Box
    query = st.text_input("🔍 Search clauses by keyword")
    
    # Load clause corpus (sample for demo)
    if os.path.exists(LEGAL_CORPUS_PATH):
        clauses = pd.read_csv(LEGAL_CORPUS_PATH).head(10)
    else:
        clauses = pd.DataFrame({
            "clause_id": [1, 2, 3],
            "text": ["Loss payable clause...", "Catastrophe event clause...", "Exclusion clause..."]
        })

    # Show results
    if query:
        results = clauses[clauses['text'].str.contains(query, case=False, na=False)]
        st.subheader("Search Results")
        st.dataframe(results)
    else:
        st.info("Enter a keyword to search clauses.")

    # Upload Option for Live Demos
    st.file_uploader("📂 Upload new clause CSV for testing", type=["csv"])

