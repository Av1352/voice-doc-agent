import gc
import io
import os
import time
from typing import Any, Dict

import numpy as np
import ffmpeg


def use_local_whisper() -> bool:
    """
    Local faster-whisper is memory-heavy on small hosts.
    - USE_LOCAL_WHISPER=true|1|yes  -> always local
    - USE_LOCAL_WHISPER=false|0|no -> always remote (OpenAI)
    - unset on Render (RENDER=true) -> remote by default
    - unset locally -> local by default
    """
    v = os.environ.get("USE_LOCAL_WHISPER", "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return os.environ.get("RENDER", "").lower() != "true"


_model = None


def clear_model():
    """Drop the local Whisper model to free RSS."""
    global _model
    _model = None
    gc.collect()


def _get_local_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        print("Loading Whisper 'tiny' model (faster-whisper). This might take a moment on the first run...")
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model


def _transcribe_local(audio_bytes: bytes, sample_rate: int) -> Dict[str, Any]:
    start_time = time.perf_counter()
    try:
        out, _ = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
            .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg decoding failure: {e.stderr.decode()}")
        return {"text": "", "latency_ms": 0.0}

    audio_float32 = np.frombuffer(out, np.float32)
    model = _get_local_model()
    segments, _info = model.transcribe(audio_float32, language="en")
    text = "".join(segment.text for segment in segments).strip()
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {"text": text, "latency_ms": latency_ms}


def _transcribe_openai(audio_bytes: bytes) -> Dict[str, Any]:
    """OpenAI Whisper API — avoids loading faster-whisper on Render."""
    start_time = time.perf_counter()
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "text": "",
            "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            "error": "Set OPENAI_API_KEY for cloud STT, or set USE_LOCAL_WHISPER=true to use local faster-whisper.",
        }
    try:
        from openai import OpenAI

        client = OpenAI()
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.webm"
        tr = client.audio.transcriptions.create(
            model=os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"),
            file=buf,
            language="en",
        )
        text = (getattr(tr, "text", None) or "").strip()
        return {"text": text, "latency_ms": (time.perf_counter() - start_time) * 1000.0}
    except Exception as e:
        print(f"OpenAI transcription error: {e}")
        return {
            "text": "",
            "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            "error": str(e),
        }


def transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> dict:
    """Converts audio and transcribes (local faster-whisper or OpenAI API)."""
    if use_local_whisper():
        return _transcribe_local(audio_bytes, sample_rate)
    return _transcribe_openai(audio_bytes)


if __name__ == "__main__":
    pass
