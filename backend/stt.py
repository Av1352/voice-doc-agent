import time
import numpy as np
import whisper
import ffmpeg

# Lazy-load the Whisper model so it only initializes once upon the first request
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper 'base' model. This might take a moment on the first run...")
        _model = whisper.load_model("base")
    return _model

def transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> dict:
    """
    Converts incoming webm/opus audio bytes from the browser to 16kHz mono PCM float32 
    in memory using ffmpeg-python, then passes it to OpenAI's Whisper model for transcription.
    """
    start_time = time.perf_counter()
    
    # 1. Convert webm/opus raw bytes via ffmpeg-python in memory (pipe)
    try:
        out, _ = (
            ffmpeg.input('pipe:0')
            .output('pipe:1', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
            .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg decoding failure: {e.stderr.decode()}")
        # Gracefully back out of the transaction returning empty structures
        return {"text": "", "latency_ms": 0.0}

    # The pipe safely mapped output directly out to f32le structure. We map right to numpy.
    audio_float32 = np.frombuffer(out, np.float32)
    
    # 2. Transcribe the natively formatted audio
    model = get_model()
    
    # Whisper forces transcription to en 
    result = model.transcribe(audio_float32, language='en')
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    # 3. Return the transcription outcome and tracking latency
    return {
        "text": result.get("text", "").strip(),
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    # Test stub
    pass
