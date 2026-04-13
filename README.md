# VoiceDoc AI

Ask questions about any PDF document — by voice. Get spoken answers back.

**Live:** https://voice-doc-agent.vercel.app

---

## What it does

Upload a PDF. Hold the mic button. Speak a question. Hear a plain-language answer.

The full pipeline runs end to end:

1. Your voice is transcribed by OpenAI Whisper API
2. The question is embedded and matched against your document via FAISS
3. Claude generates a plain-language answer from the retrieved context
4. ElevenLabs speaks the answer back sentence by sentence

---

## Measured latency (Render free tier, CPU only)

| Component | Latency |
|-----------|---------|
| STT (OpenAI Whisper API) | ~1,424ms |
| Vector retrieval (FAISS) | ~1,601ms |
| Time to first audio (ElevenLabs) | ~1,699ms |
| Total pipeline | ~11s (includes cold start) |

On a warm instance with GPU, total drops to ~3s.

---

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React + Vite, Vercel |
| Backend | FastAPI, Render |
| STT | OpenAI Whisper API |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5) |
| Retrieval | FAISS IndexFlatIP |
| LLM | Claude (Anthropic API), streaming |
| TTS | ElevenLabs streaming |
| Document parsing | pdfplumber with table-aware chunking |

---

## Architecture

```
User speaks
    ↓
Browser (MediaRecorder) → audio blob over WebSocket
    ↓
FastAPI WebSocket handler
    ↓
OpenAI Whisper API → transcript
    ↓
fastembed + FAISS → top-k document chunks
    ↓
Claude (streaming) → sentence-by-sentence response
    ↓
ElevenLabs (streaming) → audio chunks sent over WebSocket
    ↓
Web Audio API → plays audio as it arrives
```

---

## Run Locally

**Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
cp .env.example .env        # add API keys
python main.py
```

**Frontend**
```bash
cd frontend
npm install
cp .env.example .env        # set VITE_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`, upload a PDF, hold the mic and ask something.

**Required API keys:**
- `ANTHROPIC_API_KEY` — claude.ai
- `ELEVENLABS_API_KEY` — elevenlabs.io (Starter plan)
- `OPENAI_API_KEY` — platform.openai.com

---

## Why this exists

Most document tools make you read and search manually. This one listens to your question and answers out loud — useful for medical letters, insurance forms, legal documents, or anything dense that people struggle to parse on their own.

The engineering challenge is latency: overlapping STT, retrieval, LLM streaming, and TTS so the first word of audio arrives as fast as possible rather than waiting for the full response.

---

Built by [Anju Vilashni Nandhakumar](https://vxanju.com)