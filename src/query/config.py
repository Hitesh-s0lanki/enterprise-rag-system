"""Semantic reuse policy, separate from Mem0 connection settings."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryCacheSettings:
    similarity_threshold: float = 0.3
    equivalence_threshold: float = 0.85
    ttl_seconds: int = 900
    top_k: int = 5
    max_payload_bytes: int = 256000

    def __post_init__(self):
        for value in (self.similarity_threshold, self.equivalence_threshold):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Cache thresholds must be finite and in (0, 1]")
        for name, maximum in (("ttl_seconds", 86400), ("top_k", 100), ("max_payload_bytes", 1000000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}")

    @classmethod
    def from_env(cls):
        return cls(
            similarity_threshold=float(os.getenv("MEM0_CACHE_SIMILARITY_THRESHOLD", "0.3")),
            equivalence_threshold=float(os.getenv("MEM0_CACHE_EQUIVALENCE_THRESHOLD", "0.85")),
            ttl_seconds=int(os.getenv("MEM0_CACHE_TTL_SECONDS", "900")),
            top_k=int(os.getenv("MEM0_CACHE_TOP_K", "5")),
            max_payload_bytes=int(os.getenv("MEM0_CACHE_MAX_PAYLOAD_BYTES", "256000")),
        )
