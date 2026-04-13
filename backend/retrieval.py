import faiss
from document_processor.embedder import get_model, embed_texts

def retrieve(query: str, index, chunks, top_k: int = 4) -> list[dict]:
    """Retrieves top_k chunks for the query."""
    if index is None or not chunks:
        return []

    model = get_model()

    q_vec = embed_texts(model, [f"query: {query}"])

    faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk_data = chunks[idx].copy()
        chunk_data["similarity"] = float(dist)
        results.append(chunk_data)

    return results
