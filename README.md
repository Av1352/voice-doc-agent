# VoiceDoc AI

Voice-first RAG over PDFs: upload a document, ask questions out loud, and get spoken answers back in real time.

**Live:** https://voice-doc-agent.vercel.app

---

## What it does

Upload a PDF (e.g., medical form, insurance letter, discharge summary), hold the mic button, speak a question, and hear a plain-language answer grounded in that document.

The full pipeline runs end to end for each question:

1. Your voice is transcribed by OpenAI Whisper API into text.
2. The question text is embedded and matched against your document via FAISS.
3. Claude generates a plain-language answer from the retrieved context.
4. ElevenLabs streams the answer back as audio, which starts playing as it’s generated.

This is a **real-time voice interface for long PDFs**, not just “chat with a document” via text.

---

## Measured latency (Render free tier, CPU only)

| Component                       | Latency   |
|---------------------------------|-----------|
| STT (OpenAI Whisper API)       | ~1,424ms  |
| Vector retrieval (FAISS)       | ~1,601ms  |
| Time to first audio (TTS)      | ~1,699ms  |
| Total pipeline (cold start)    | ~11s      |

On a warm instance with GPU, total end-to-end latency drops to ~3s.

The app overlaps STT, retrieval, LLM streaming, and TTS so the **first word** arrives quickly instead of waiting for the full response.

---

## Stack

| Layer             | Tech                                           |
|-------------------|------------------------------------------------|
| Frontend          | React + Vite (deployed on Vercel)             |
| Backend           | FastAPI + Uvicorn (deployed on Render)        |
| STT               | OpenAI Whisper API (`faster-whisper` available) |
| Embeddings        | `fastembed` (BAAI/bge-small-en-v1.5)          |
| Retrieval         | FAISS `IndexFlatIP`                           |
| LLM               | Claude via Anthropic API (streaming)          |
| TTS               | ElevenLabs streaming                          |
| Document parsing  | `pdfplumber` with table-aware chunking        |
| Transport         | WebSockets for full-duplex audio + metadata   |

> Note: the backend keeps a **single active FAISS index + chunk set** in memory for simplicity. Extending to multi-user/multi-document is straightforward but out of scope for this demo.

---

## Architecture

```text
User speaks
    ↓
Browser (MediaRecorder) → audio blobs over WebSocket
    ↓
FastAPI WebSocket handler
    ↓
STT → OpenAI Whisper API → transcript (text question)
    ↓
Embeddings → fastembed (BAAI/bge-small-en-v1.5)
    ↓
Retrieval → FAISS IndexFlatIP → top‑k document chunks
    ↓
LLM → Claude (streaming) → answer text
    ↓
TTS → ElevenLabs (streaming) → audio chunks over WebSocket
    ↓
Web Audio API → plays audio as it arrives
```

**Key design choices:**

- Voice is captured and played entirely in the browser; all ML inference happens on the backend.
- Retrieval is semantic: the LLM only sees the top‑k chunks from FAISS, not the entire document.
- Audio and text are streamed end‑to‑end for low perceived latency.

---

## Run locally

### Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # add API keys
python main.py
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env  # set VITE_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`, upload a PDF, hold the mic, and ask a question.

### Required API keys

- `ANTHROPIC_API_KEY` — Claude (Anthropic)
- `ELEVENLABS_API_KEY` — ElevenLabs (Starter plan is enough)
- `OPENAI_API_KEY` — Whisper API

---

## How it works (deep dive)

### 1. Document ingestion

When you hit `/upload-document` with a PDF:

1. FastAPI streams the upload to a temporary file on disk (with a configurable max size).
2. `document_processor.ingest` parses the PDF with `pdfplumber`.
3. `chunker` splits the text into table-aware chunks.
4. `embedder` encodes each chunk with `fastembed` (`BAAI/bge-small-en-v1.5`).
5. A FAISS `IndexFlatIP` index is built and persisted to `data/index.faiss`, and chunk metadata to `data/chunks.pkl`.

On startup, the backend memory-maps this FAISS index and loads the chunks into a simple **global_state** so subsequent queries are fast.

### 2. Voice input → text

The React frontend uses `MediaRecorder` to capture microphone audio and sends audio blobs over a WebSocket to `/ws`.

On the backend, the WebSocket handler:

- Receives audio bytes.
- Hands them to the voice pipeline, which calls the OpenAI Whisper API to transcribe the speech into text.
- If no document is loaded yet, it responds with an error asking you to upload a PDF first.

### 3. Text → relevant chunks

Once the question is transcribed:

1. The text query is embedded using the same `fastembed` model as the document chunks.
2. FAISS performs a nearest-neighbor search to find the most similar chunks (semantic retrieval).
3. Those chunks (plus the question) are packaged as context for the LLM.

### 4. LLM answer

The pipeline calls Claude via Anthropic’s streaming API, providing:

- The user’s question.
- The retrieved document chunks as context.

Claude streams back a grounded answer in plain language. The pipeline buffers text at the sentence level to send to TTS.

### 5. Text → speech

For each piece of answer text:

1. The pipeline calls ElevenLabs’ streaming TTS API.
2. Audio chunks are streamed back from ElevenLabs.
3. The backend forwards those audio chunks over the WebSocket to the browser.

On the frontend, the Web Audio API plays these chunks as they arrive, so the user hears the answer in (near) real time.

### 6. Final metadata

At the end of each query, the backend sends a final JSON message over the WebSocket containing:

- `query` — the transcribed text of the user’s question.
- `response_text` — the full LLM answer.
- `timings` — per-component latency breakdown (STT, retrieval, TTFA, total).
- `error` — if anything failed.

The UI displays these metrics so you can see where time is spent.

---

## Why this exists

Most document tools make you read and search manually. VoiceDoc AI lets you **talk** to long PDFs and hear answers back, which is especially useful for:

- Medical letters and discharge summaries
- Insurance and prior authorization packets
- Legal documents and contracts
- Any dense PDF that people struggle to parse on their own

From an engineering perspective, this is a reference implementation of a **low-latency, streaming voice RAG pipeline** (STT → retrieval → LLM → TTS) that can be adapted to more complex multi-agent or multi-document systems.
