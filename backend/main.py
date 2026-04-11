import os
import sys
import json
import pickle
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Make absolutely sure that our imports work dynamically spanning the underlying project structure hierarchy
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load root .env files prior to loading SDK clients natively traversing globals
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from backend import pipeline
from document_processor import chunker
from document_processor import embedder

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Maintain active inference structure bounds globally
global_state = {
    "index": None,
    "chunks": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retrieve indexed structure matrices asynchronously on fastAPI boot
    index_path = os.path.join(DATA_DIR, 'index.faiss')
    chunks_path = os.path.join(DATA_DIR, 'chunks.pkl')
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print("Loading existing FAISS index and chunk boundaries from disk memory...")
        try:
            global_state["index"] = faiss.read_index(index_path)
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


# Configure central application execution contexts
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

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Parses pdf contents, aligns table extractions/short text sequences seamlessly,
    ingests content to vectors, and binds to active fastAPI inference global states.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_pdf_path = os.path.join(DATA_DIR, file.filename)
    
    try:
        content = await file.read()
        with open(temp_pdf_path, "wb") as f:
            f.write(content)
            
        print(f"Ingesting Uploaded PDF Structure Context: {file.filename}...")
        chunks = chunker.process_pdf(temp_pdf_path)
        
        print(f"Creating Semantic Extrapolations -> Build FAISS ({len(chunks)} chunks)...")
        index = embedder.build_index(chunks)
        
        # Load the result payload context into inference memory hooks dynamically
        global_state["index"] = index
        global_state["chunks"] = chunks
        
        return {
            "status": "indexed",
            "filename": file.filename,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bi-directional event loop WebSocket managing concurrent Whisper STT audio ingrains, 
    FAISS Retrieval inferences, sequential Claude LLM bounds, and ElevenLabs TTS relays.
    """
    await websocket.accept()
    print("Bi-directional Socket Session active bounds initialized...")
    
    try:
        while True:
            # Halt passively waiting for audio ingress bounds dynamically across connection
            audio_bytes = await websocket.receive_bytes()
            
            # Thread query resolution pipeline concurrently natively handling STT->Retrieval->LLM->TTS
            result = await pipeline.process_voice_query(
                audio_bytes, 
                global_state["index"], 
                global_state["chunks"]
            )
            
            # Immediately burst concurrent audio payload chunks completely joined out
            await websocket.send_bytes(result["audio"])
            
            # Propagate system processing metric states context trailing behind media byte transfers mapped
            telemetry_message = {
                "query": result["query"],
                "timings": result["timings"]
            }
            await websocket.send_text(json.dumps(telemetry_message))
            
    except WebSocketDisconnect:
        print("Bi-directional socket closed cleanly.")
    except Exception as e:
        print(f"Socket resolution error mapping broken bounds: {e}")
        try:
             # Ensure failures transmit closure status context cleanly 
             await websocket.send_text(json.dumps({"error": str(e)}))
             await websocket.close()
        except:
             pass
