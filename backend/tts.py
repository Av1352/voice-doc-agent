import os
import time
from typing import Generator
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Lazy-load the ElevenLabs client globally
_client = None

def get_client() -> ElevenLabs:
    global _client
    if _client is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            print("Warning: ELEVENLABS_API_KEY environment variable is missing!")
            
        # The new ElevenLabs SDK uses the standard ElevenLabs class wrapper for clients
        _client = ElevenLabs(api_key=api_key)
    return _client

def stream_audio(text: str) -> Generator[bytes, None, None]:
    """
    Calls the ElevenLabs SDK text_to_speech.convert_as_stream to synthesize audio
    dynamically and yields incoming byte chunks for playback. Tracks latency telemetry.
    """
    client = get_client()
    
    start_time = time.perf_counter()
    first_chunk_received = False
    
    audio_stream = client.text_to_speech.convert_as_stream(
        text=text,
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75
        )
    )
    
    for chunk in audio_stream:
        # ElevenLabs might sometimes yield empty chunk bytes in buffering depending on connection
        if chunk:
            if not first_chunk_received:
                # Track and print the Time to First Byte (TTFB/Latency to first chunk)
                first_chunk_latency = (time.perf_counter() - start_time) * 1000.0
                print(f"TTS Latency to first chunk: {first_chunk_latency:.2f} ms")
                first_chunk_received = True

            yield chunk

if __name__ == "__main__":
    # Test stub
    pass
