import os
import sys
import time
import asyncio
import threading
from typing import AsyncGenerator, Any, Dict, Optional

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

async def process_voice_query(audio_bytes: bytes, index, chunks) -> AsyncGenerator[Dict[str, Any], None]:
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
    #
    # IMPORTANT: llm.stream_response() and tts.stream_audio() are blocking generators.
    # Running them on the event loop can stall ping/pong and get the connection killed
    # by proxies. We run the blocking stream in a background thread and bridge results
    # via an asyncio.Queue.
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=32)
    loop = asyncio.get_running_loop()

    def put_event(evt: Dict[str, Any]) -> None:
        # Backpressure-safe: block in the worker thread until the event is enqueued.
        asyncio.run_coroutine_threadsafe(queue.put(evt), loop).result()

    def worker() -> None:
        first_sentence_ms_local: Optional[float] = None
        full_response_parts_local: list[str] = []
        llm_start = time.perf_counter()

        try:
            for sentence in llm.stream_response(query, context_chunks):
                full_response_parts_local.append(sentence)

                if first_sentence_ms_local is None:
                    first_sentence_ms_local = (time.perf_counter() - llm_start) * 1000.0

                for audio_chunk in tts.stream_audio(sentence):
                    put_event({"type": "audio", "data": audio_chunk})

            put_event(
                {
                    "type": "final",
                    "query": query,
                    "response_text": " ".join(full_response_parts_local).strip(),
                    "first_sentence_ms": float(first_sentence_ms_local or 0.0),
                }
            )
        except Exception as e:
            put_event({"type": "error", "error": str(e)})

    thread = threading.Thread(target=worker, name="llm_tts_stream", daemon=True)
    thread.start()

    while True:
        event = await queue.get()

        if event.get("type") == "audio":
            yield event
            continue

        if event.get("type") == "error":
            # Surface a final-ish event so the WS handler can respond.
            timings["first_sentence_ms"] = timings.get("first_sentence_ms", 0.0)
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            await asyncio.to_thread(log_latency, query, timings)
            yield {
                "type": "final",
                "query": query,
                "response_text": "",
                "timings": timings,
                "error": event.get("error", "Unknown error"),
            }
            break

        if event.get("type") == "final":
            timings["first_sentence_ms"] = float(event.get("first_sentence_ms", 0.0))
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            await asyncio.to_thread(log_latency, query, timings)
            yield {
                "type": "final",
                "query": event.get("query", query),
                "response_text": event.get("response_text", ""),
                "timings": timings,
            }
            break

        # Ignore unknown event types to keep the stream resilient.
        continue
    
    
