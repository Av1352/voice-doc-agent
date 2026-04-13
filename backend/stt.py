import time
import numpy as np
import whisper
import ffmpeg

# Lazy-load the Whisper model so it only initializes once upon the first request
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper 'tiny' model. This might take a moment on the first run...")
        _model = whisper.load_model("tiny")
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
    result = model.transcribe(audio_float32, language='en')
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    # Return text and latency
    return {
        "text": result.get("text", "").strip(),
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    # Test stub
    pass
