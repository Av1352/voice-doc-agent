import os
from typing import Any, Dict


def get_rss_bytes() -> int:
    """
    Best-effort resident set size (RSS) in bytes.
    Works on Linux via /proc, otherwise returns 0.
    """
    try:
        with open(f"/proc/{os.getpid()}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Example: "VmRSS:\t  123456 kB\n"
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return kb * 1024
    except Exception:
        pass
    return 0


def mem_event(stage: str, **extra: Any) -> Dict[str, Any]:
    rss = get_rss_bytes()
    evt: Dict[str, Any] = {"stage": stage, "rss_mb": round(rss / (1024 * 1024), 1) if rss else None}
    evt.update(extra)
    return evt

