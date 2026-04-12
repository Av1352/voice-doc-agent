import os
import tempfile
from document_processor import chunker, embedder

def process_document(file_bytes: bytes, filename: str):
    """Processes a document to chunks and vectors."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        print(f"Processing PDF: {filename}")
        chunks = chunker.process_pdf(temp_path)
        
        print(f"Building FAISS index for {len(chunks)} chunks")
        index = embedder.build_index(chunks)
        
        return index, chunks
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
