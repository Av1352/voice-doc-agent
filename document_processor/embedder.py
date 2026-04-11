import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Lazy load the model globally to avoid reloading on each retrieve call
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def build_index(chunks):
    """
    Encodes the given chunks using sentence-transformers, builds a FAISS IndexFlatIP 
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
    # Extract only the text content from the chunk dictionaries
    texts = [chunk.get('content', '') for chunk in chunks]
    
    # encode() returns a numpy array of embeddings (float32 normally)
    embeddings = model.encode(texts)
    
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
    return index

def retrieve(query, index, chunks, top_k=4):
    """
    Encodes the query, normalizes it, searches the FAISS index, and returns 
    the top-k matching chunks as a list of dicts.
    """
    model = get_model()
    
    # Encode the query string into a 2D array [1, dimension]
    query_embedding = model.encode([query])
    
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
