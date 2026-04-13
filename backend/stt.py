import time
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg

# Lazy-load the Whisper model so it only initializes once upon the first request
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper 'tiny' model (faster-whisper). This might take a moment on the first run...")
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model

def transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> dict:
    """Converts audio to float32 and transcribes using Whisper."""
    
    start_time = time.perf_counter()
    
    # Convert audio via ffmpeg
    try:
        out, _ = (
            ffmpeg.input('pipe:0')
            .output('pipe:1', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
            .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg decoding failure: {e.stderr.decode()}")
        # Return empty on failure
        return {"text": "", "latency_ms": 0.0}

    # Map output to numpy array
    audio_float32 = np.frombuffer(out, np.float32)
    
    # Transcribe audio
    model = get_model()
    
    # Force English logic
    segments, _info = model.transcribe(audio_float32, language="en")
    text = "".join(segment.text for segment in segments).strip()
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    # Return text and latency
    return {
        "text": text,
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    # Test stub
    pass
