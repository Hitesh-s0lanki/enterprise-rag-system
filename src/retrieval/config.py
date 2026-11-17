"""Bounded retrieval settings, independent of ingestion chunk sizes."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5
    candidate_k: int = 20
    dense_weight: float = 0.7
    rrf_k: int = 60
    timeout_seconds: float = 30
    max_query_characters: int = 4000

    def __post_init__(self):
        if not 1 <= self.top_k <= self.candidate_k <= 200:
            raise ValueError("Require 1 <= top_k <= candidate_k <= 200")
        if not math.isfinite(self.dense_weight) or not 0 < self.dense_weight < 1:
            raise ValueError("dense_weight must be strictly between 0 and 1 for hybrid search")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not 1 <= self.max_query_characters <= 4000:
            raise ValueError("max_query_characters must be between 1 and 4000")

    @classmethod
    def from_env(cls):
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            candidate_k=int(os.getenv("RETRIEVAL_CANDIDATE_K", "20")),
            dense_weight=float(os.getenv("RETRIEVAL_DENSE_WEIGHT", "0.7")),
            rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            timeout_seconds=float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "30")),
            max_query_characters=int(os.getenv("RETRIEVAL_MAX_QUERY_CHARACTERS", "4000")),
        )
