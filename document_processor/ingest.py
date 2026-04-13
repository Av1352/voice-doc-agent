import os
import tempfile
from document_processor import chunker, embedder


def process_document_path(pdf_path: str, filename: str):
    """
    Processes a PDF on disk to chunks and vectors.
    Caller owns pdf_path (create/delete outside this function).
    """
    print(f"Processing PDF: {filename}")
    chunks = chunker.process_pdf(pdf_path)

    print(f"Building FAISS index for {len(chunks)} chunks")
    index = embedder.build_index(chunks)

    return index, chunks


def process_document(file_bytes: bytes, filename: str):
    """Processes in-memory PDF bytes (legacy); prefer process_document_path + streaming read."""
    temp_dir = tempfile.gettempdir()
    safe = os.path.basename(filename) or "document.pdf"
    temp_path = os.path.join(temp_dir, safe)

    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        return process_document_path(temp_path, filename)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
