import os
import numpy as np
import faiss

def build_faiss_index(embedding_path: str, output_path: str, metric: str = "cosine"):
    """
    Build a FAISS index for ClauseLens embeddings.
    Args:
        embedding_path: Path to embeddings.npy
        output_path: Where to save faiss_index.bin
        metric: "cosine" or "l2"
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embeddings not found at {embedding_path}")
    
    embeddings = np.load(embedding_path).astype('float32')
    num_vectors, dim = embeddings.shape

    if metric == "cosine":
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine
    else:
        index = faiss.IndexFlatL2(dim)  # Euclidean distance

    # Add embeddings to index
    index.add(embeddings)

    # Save index
    faiss.write_index(index, output_path)
    print(f"✅ FAISS index built with {num_vectors} vectors (dim={dim}) and saved to {output_path}")


if __name__ == "__main__":
    EMBEDDING_PATH = "../clauselens/legal_corpus/embeddings.npy"
    OUTPUT_PATH = "../clauselens/legal_corpus/faiss_index.bin"

    build_faiss_index(EMBEDDING_PATH, OUTPUT_PATH, metric="cosine")
