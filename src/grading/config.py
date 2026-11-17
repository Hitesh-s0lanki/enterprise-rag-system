"""Application grading thresholds, separate from Jev connection settings."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GradingSettings:
    relevance_threshold: float = 0.6
    evidence_threshold: float = 0.7
    max_documents: int = 20
    max_document_characters: int = 20000
    max_query_characters: int = 4000

    def __post_init__(self):
        for value in (self.relevance_threshold, self.evidence_threshold):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Grading thresholds must be finite and between 0 and 1")
        for value, maximum in ((self.max_documents, 200), (self.max_document_characters, 100000),
                               (self.max_query_characters, 4000)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"Grading limit must be an integer between 1 and {maximum}")

    @classmethod
    def from_env(cls):
        return cls(
            relevance_threshold=float(os.getenv("GRADING_RELEVANCE_THRESHOLD", "0.6")),
            evidence_threshold=float(os.getenv("GRADING_EVIDENCE_THRESHOLD", "0.7")),
            max_documents=int(os.getenv("GRADING_MAX_DOCUMENTS", "20")),
            max_document_characters=int(os.getenv("GRADING_MAX_DOCUMENT_CHARACTERS", "20000")),
            max_query_characters=int(os.getenv("GRADING_MAX_QUERY_CHARACTERS", "4000")),
        )
