import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_clause_embeddings(csv_path: str, output_path: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Generate semantic embeddings for ClauseLens clause_texts.
    Saves embeddings as embeddings.npy in the same order as clauses.csv.
    """
    # Load clauses
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clause CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "clause_text" not in df.columns:
        raise ValueError("clauses.csv must have a 'clause_text' column.")
    
    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = model.encode(df["clause_text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    
    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings
    np.save(output_path, embeddings)
    print(f"✅ Saved {embeddings.shape[0]} embeddings with dim {embeddings.shape[1]} to {output_path}")


if __name__ == "__main__":
    CSV_PATH = "../clauselens/legal_corpus/clauses.csv"
    OUTPUT_PATH = "../clauselens/legal_corpus/embeddings.npy"
    
    generate_clause_embeddings(CSV_PATH, OUTPUT_PATH)
