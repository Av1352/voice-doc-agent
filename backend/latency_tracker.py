import os
import json
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def log_latency(query: str, timings: dict):
    """
    Appends the given query and its execution timings as a JSON line 
    to data/latency_log.jsonl.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    log_file = os.path.join(DATA_DIR, "latency_log.jsonl")
    
    record = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp": time.time(),
        "query": query,
    }
    
    # Merge the exact timing fields safely mapped sequentially
    record.update(timings)
    
    with open(log_file, "a", encoding="utf-8") as f:
        # Write exactly one JSON line tracking all data payloads safely.
        f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    pass
