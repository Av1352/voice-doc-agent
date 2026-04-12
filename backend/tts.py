import os
import time
from typing import Generator
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from dotenv import load_dotenv

load_dotenv()

def stream_audio(text: str) -> Generator[bytes, None, None]:
    """Synthesizes speech using ElevenLabs TTS as a stream."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set.")
        return

    client = ElevenLabs(api_key=api_key)
    
    start_time = time.perf_counter()
    first_chunk_received = False
    
    try:
        audio_stream = client.text_to_speech.stream(
            voice_id='21m00Tcm4TlvDq8ikWAM',
            model_id='eleven_turbo_v2',
            text=text,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
        )
        
        for chunk in audio_stream:
            if chunk:
                if not first_chunk_received:
                    latency = (time.perf_counter() - start_time) * 1000.0
                    print(f"TTS Time to First Chunk Latency (ElevenLabs): {latency:.2f} ms")
                    first_chunk_received = True
                yield chunk

    except Exception as e:
        print(f"ElevenLabs TTS error: {e}")

if __name__ == "__main__":
    pass