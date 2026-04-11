import os
import sys
import time
import asyncio

# Ensure project root is fully registered under module scopes dynamically
# Resolves cross-import bindings dynamically so python module paths succeed universally
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend import stt
from backend import llm
from backend import tts
from backend.latency_tracker import log_latency
from document_processor import embedder as retrieval

async def process_voice_query(audio_bytes: bytes, index, chunks) -> dict:
    """
    Orchestrates the entire voice end-to-end pipeline handling conversion telemetry tracking:
    1. STT inference
    2. Document vector retrieval
    3. Pipelined stream inference connecting the Claude Response directly across ElevenLabs TTS generators
    4. Aggregates data sequences seamlessly returning combined playback bindings alongside telemetry logs.
    """
    total_start = time.perf_counter()
    timings = {}

    # 1. Speech to Text Analysis
    # Await via to_thread to keep asynchronous execution robust while the IO-bound whisper runs seamlessly
    stt_result = await asyncio.to_thread(stt.transcribe, audio_bytes)
    timings["stt_ms"] = stt_result.get("latency_ms", 0.0)
    query = stt_result.get("text", "")

    # 2. Vector Matrix Retrieval Execution
    retrieval_start = time.perf_counter()
    context_chunks = await asyncio.to_thread(retrieval.retrieve, query, index, chunks)
    timings["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000.0

    # 3. LLM and TTS Streaming Event Loop Integration
    first_sentence_ms = None
    all_audio_bytes = bytearray()
    
    llm_start = time.perf_counter()
    
    for sentence in llm.stream_response(query, context_chunks):
        
        # Tag precisely the point first sentence structure hits 
        if first_sentence_ms is None:
            first_sentence_ms = (time.perf_counter() - llm_start) * 1000.0
            timings["first_sentence_ms"] = first_sentence_ms
            
        # Relay sentence chunks sequentially outwards to ElevenLabs TTS execution 
        for audio_chunk in tts.stream_audio(sentence):
            all_audio_bytes.extend(audio_chunk)

    if first_sentence_ms is None:
        timings["first_sentence_ms"] = 0.0
        
    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    
    # Log timings safely externally executing latency_tracker writes 
    await asyncio.to_thread(log_latency, query, timings)

    return {
        "query": query,
        "audio": bytes(all_audio_bytes),
        "timings": timings
    }
