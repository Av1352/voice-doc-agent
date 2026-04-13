import os
import sys
import json
import pickle
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load .env files
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from backend import pipeline
from backend.memory import mem_event
from document_processor import chunker
from document_processor import embedder
from document_processor import ingest

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Global state
global_state = {
    "index": None,
    "chunks": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load FAISS index and chunks on startup
    index_path = os.path.join(DATA_DIR, 'index.faiss')
    chunks_path = os.path.join(DATA_DIR, 'chunks.pkl')
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print("Loading existing FAISS index and chunk boundaries from disk memory...")
        try:
            mmap_flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            global_state["index"] = faiss.read_index(index_path, mmap_flags)
            with open(chunks_path, 'rb') as f:
                global_state["chunks"] = pickle.load(f)
            print(f"Persisted successfully: Loaded {len(global_state['chunks'])} document chunks.")
        except Exception as e:
            print(f"Read bounds failed checking indices structures natively: {e}")
            global_state["index"], global_state["chunks"] = None, None
    else:
        print("No existing FAISS index located - active mode setup pending document POST ingress pipeline.")
        
    yield
    
    # Secure clear-out
    global_state["index"] = None
    global_state["chunks"] = None


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.head("/health")
def health_check_head():
    return Response(status_code=200)

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Parses PDF, vectors chunks, and sets global state."""
    try:
        print(mem_event("upload_document:start", filename=file.filename))
        content = await file.read()
        print(mem_event("upload_document:read_complete", bytes=len(content)))
        index, chunks = ingest.process_document(content, file.filename)
        print(mem_event("upload_document:process_complete", chunks=len(chunks) if chunks else 0))
        # Keep memory low after index build (model can reload on first query).
        embedder.clear_model()
        
        # Update global state
        global_state["index"] = index
        global_state["chunks"] = chunks
        print(mem_event("upload_document:state_set"))
        
        return {
            "status": "indexed",
            "filename": file.filename,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles bi-directional WebSocket voice pipeline."""
    await websocket.accept()
    print("WebSocket connected")
    print(mem_event("ws:accepted"))
    
    try:
        while True:
            # Receive audio
            audio_bytes = await websocket.receive_bytes()
            print(mem_event("ws:received_audio", bytes=len(audio_bytes)))
            
            if global_state["index"] is None:
                await websocket.send_text(json.dumps({'error': 'No document loaded. Please upload a PDF first.'}))
                continue
            
            # Stream query pipeline (no RAM accumulation)
            try:
                async for event in pipeline.process_voice_query(
                    audio_bytes,
                    global_state["index"],
                    global_state["chunks"],
                ):
                    if event.get("type") == "audio":
                        await websocket.send_bytes(event["data"])
                    elif event.get("type") == "final":
                        print(mem_event("ws:final_ready"))
                        await websocket.send_text(json.dumps({
                            "query": event.get("query", ""),
                            "timings": event.get("timings", {}),
                            "response_text": event.get("response_text", ""),
                            "error": event.get("error", ""),
                        }))
                    else:
                        # Ignore unknown event types to keep the socket robust
                        continue
            except WebSocketDisconnect:
                # Client toggled mic / closed socket mid-stream.
                print("WebSocket closed during stream")
                break
            except Exception as e:
                # Any send_bytes/send_text failure should stop the stream immediately.
                print(f"WebSocket stream send error: {e}")
                break
            
    except WebSocketDisconnect:
        print("WebSocket closed cleanly")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            # Send error via websocket
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)