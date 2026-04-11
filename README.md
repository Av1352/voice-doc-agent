# Voice Document Agent

Voice Document Agent is an ultra-low latency, voice-to-voice Retrieval-Augmented Generation (RAG) assistant. It allows users to upload PDF documents, intelligently index them offline, and interact dynamically by holding a clean push-to-talk button to ask questions and hearing contextual, voice-synthesized answers seamlessly in return. 

## System Architecture

The core of the system relies on executing STT, semantic document retrieval, and overlapping LLM/TTS streams concurrently to achieve near real-time interaction bounds. 

```text
[ User Speaks ] 
      │ 
      ▼ (WebSocket / WebM Bytes)
[ FFmpeg In-Memory Pipe ]
      │ 
      ▼ (PCM 16k Float32)
[ OpenAI Whisper 'Base' ] (STT)
      │
      ▼ (Query String)
[ FAISS Vector DB ] (Semantic Document Retrieval)
      │
      ▼ (Context Chunks)
[ Anthropic Claude Stream ] (LLM logic bounded by 20-rule words)
      │
      ▼ (Streaming Tokens -> Splitting cleanly at '.' '!' '?')
[ ElevenLabs Turbo TTS ] (Text -> Audio Stream)
      │
      ▼ (Audio Array Buffers)
[ Client Web Audio API ] (Playback instantly)
```

## Latency Budget Breakdown

| Component | Expected Latency | Notes |
|:---|:---:|:---|
| **STT (Whisper Base)** | 150ms - 400ms | Extremely dependent on local compute hardware |
| **Retrieval (FAISS)** | 10ms - 30ms | Inner-product flat searches are extremely fast loaded in memory |
| **LLM (Claude Sonnet)** | 300ms - 500ms | Measured strictly against Time To First Sentence (TTFS)* |
| **TTS (ElevenLabs)** | 150ms - 250ms | Time To First Audio Chunk using `eleven_turbo_v2` |
| **Roundtrip TTFA** | **~800ms - 1.2s** | Audio starts playing back on the client device actively! |

## Tech Stack

| Domain | Technology / Library | Function |
|:---|:---|:---|
| Frontend | Vanilla HTML / JS | No-framework, MediaRecorder audio mapping |
| Backend API | FastAPI & Uvicorn | Async WebSocket and HTTP controller |
| STT Engine | openai-whisper | Local transcription over PCM byte bindings |
| Context Reader| pdfplumber | Fast layout extraction and text chunking |
| Embedder | sentence-transformers | `all-MiniLM-L6-v2` structural encodings |
| Vector DB | faiss-cpu | High speed `IndexFlatIP` retrieval architectures |
| Agent LLM | anthropic | Streaming contextual comprehension via Claude |
| TTS Engine | elevenlabs | Ultra-fast Synthetic voice generation ('Rachel') |

## Local Setup

### 1. FFmpeg Installation (Required)
The application handles web browser `webm/opus` audio natively by piping byte streams through FFmpeg logic. This must be installed on your operating system.

- **Windows**: `winget install ffmpeg` (or download and add to System PATH)
- **Mac**: `brew install ffmpeg`
- **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install ffmpeg`

### 2. Environment Variables & API Keys
Create a `.env` file securely inside your `/backend` directory mapping your API keys:
```text
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```
- **Anthropic**: Fetch keys directly from the [Anthropic Console](https://console.anthropic.com/).
- **ElevenLabs**: Fetch keys from your [ElevenLabs Dashboard](https://elevenlabs.io/).

### 3. Running the Backend Server
The backend utilizes Python 3.9+. 
```bash
cd backend
python -m venv venv
source venv/bin/activate       # (Mac/Linux)
# OR: .\venv\Scripts\activate  # (Windows)

pip install -r requirements.txt
uvicorn main:app --reload
```
*Note: Booting Whisper or Sentence Transformers for the very first time will safely download the required model weights into your HuggingFace cache locally which could take a moment.*

### 4. Booting the Frontend
Because we execute directly using completely Vanilla CSS and JS:
1. Double click `frontend/index.html` to open it locally via standard `file://` bounds.
2. OR (to remove local guarded CORS constraints on very specific browser engines), spin up a fast local web server targeting the UI explicitly:
```bash
cd frontend
python -m http.server 3000
```
Navigate your browser to `http://localhost:3000`.

## Engineering Details: The Overlapping LLM & TTS Sequence Challenge

A standard serial voice pipeline creates brutal runtime latency: `(User Stops) -> STT -> Vector Retrieval -> Fully Generate LLM Text -> Fully Generate TTS Audio -> (Bot Answers)`. This sequence can easily take 4 to 8+ seconds!

To achieve true sub-second Time To First Audio (TTFA), we completely overlap the largest bottlenecks! 

Instead of waiting for the LLM to complete its entire output, `voice-doc-agent`'s structural pipeline dictates that it must:
1. Traps Claude's streamed text generation natively traversing token-by-token continuously.
2. Check aggressively via Regex boundary hooks for strictly structured punctuation sequence terminations (`.`, `!`, `?` mapped explicitly across whitespace separations).
3. The absolute instant a single complete short sentence validates securely inside the chunking buffer, we pull it entirely and push it actively towards ElevenLabs while Claude continues predicting and generating the subsequent responses asynchronously! 

By explicitly forcing our System Prompt instruction context to restrict Claude's semantic structures exceptionally short (under 20 words bounded strictly per string output), it mitigates any sequential lockup and begins pushing live audio data back to your client device natively when the LLM is essentially a fraction of the way fully completed predicting exactly what it intends to say.
