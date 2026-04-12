import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    """Lazily loads the sentence-transformers model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def retrieve(query: str, index, chunks, top_k: int = 4) -> list[dict]:
    """Retrieves top_k chunks for the query."""
    if index is None or not chunks:
        return []
        
    model = get_model()
    
    # Encode query
    q_vec = model.encode([query])
    
    # L2 normalize
    faiss.normalize_L2(q_vec)
    
    # Search index
    distances, indices = index.search(q_vec, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk_data = chunks[idx].copy()
        chunk_data["similarity"] = float(dist)
        results.append(chunk_data)
        
    return results
