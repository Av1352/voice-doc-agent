import os
import pickle
import gc
import faiss
import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Lazy load the model globally to avoid reloading on each retrieve call
_model = None

def clear_model():
    """Frees the embedding model to reduce RSS on low-memory hosts."""
    global _model
    _model = None
    gc.collect()

def get_model():
    global _model
    if _model is None:
        # Render (and similar) often logs GPU probing; disable it and keep ORT minimal.
        _model = TextEmbedding(
            MODEL_NAME,
            providers=["CPUExecutionProvider"],
            threads=1,
            cuda=False,
            lazy_load=True,
        )
    return _model

def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Runs fastembed and returns a contiguous float32 matrix (n, dim)."""
    vectors = list(model.embed(texts))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors).astype(np.float32, copy=False)

def build_index(chunks):
    """
    Encodes the given chunks using fastembed, builds a FAISS IndexFlatIP
    index with L2-normalized vectors, and saves both the index and the chunks to disk
    as data/index.faiss and data/chunks.pkl.
    """
    if not chunks:
        print("Warning: No chunks provided to build_index.")
        return None

    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    model = get_model()

    print(f"Encoding {len(chunks)} chunks...")
    # BGE retrieval: passage prefix for indexed documents
    texts = [f"passage: {chunk.get('content', '')}" for chunk in chunks]

    embeddings = embed_texts(model, texts)

    # L2 normalize vectors inplace for cosine similarity tracking via inner product
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save the index and chunks
    index_path = os.path.join(DATA_DIR, 'index.faiss')
    chunks_path = os.path.join(DATA_DIR, 'chunks.pkl')

    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)

    print(f"Saving chunks to {chunks_path}...")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

    print("Index built and saved successfully.")
    # Release the model after indexing to keep headroom for STT/LLM/TTS.
    clear_model()
    return index

def retrieve(query, index, chunks, top_k=4):
    """
    Encodes the query, normalizes it, searches the FAISS index, and returns
    the top-k matching chunks as a list of dicts.
    """
    model = get_model()

    # BGE retrieval: query prefix for search
    query_embedding = embed_texts(model, [f"query: {query}"])

    # L2 normalize the query vector in place
    faiss.normalize_L2(query_embedding)

    # Search the index for the nearest neighbors
    distances, indices = index.search(query_embedding, top_k)

    results = []
    # Both distances and indices are 2D arrays. We take [0] for the single query.
    for dist, i in zip(distances[0], indices[0]):
        # faiss might return -1 if there are fewer than top_k elements in the index
        if 0 <= i < len(chunks):
            # Create a copy so we do not mutate the original chunk dict
            match = dict(chunks[i])
            match['similarity_score'] = float(dist)
            results.append(match)

    return results

if __name__ == "__main__":
    # Test stub
    pass
