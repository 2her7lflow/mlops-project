from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict


_lock = threading.Lock()
_request_counts: Dict[str, int] = defaultdict(int)
_latency_ms_sum: Dict[str, float] = defaultdict(float)
_latency_ms_max: Dict[str, float] = defaultdict(float)


def record(path: str, method: str, latency_ms: float) -> None:
    key = f"{method} {path}"
    with _lock:
        _request_counts[key] += 1
        _latency_ms_sum[key] += latency_ms
        if latency_ms > _latency_ms_max[key]:
            _latency_ms_max[key] = latency_ms


def snapshot() -> dict:
    with _lock:
        rows = []
        for key, count in sorted(_request_counts.items()):
            avg = (_latency_ms_sum[key] / count) if count else 0.0
            rows.append(
                {
                    "endpoint": key,
                    "count": count,
                    "avg_latency_ms": round(avg, 2),
                    "max_latency_ms": round(_latency_ms_max[key], 2),
                }
            )
        return {
            "generated_at": time.time(),
            "endpoints": rows,
        }
