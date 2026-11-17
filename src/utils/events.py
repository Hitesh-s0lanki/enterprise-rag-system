"""Serializable events shared by synchronous, asynchronous and SSE consumers."""

import json
from dataclasses import asdict, dataclass
from typing import Literal

EventType = Literal[
    "run_started", "model_started", "text_delta", "tool_call_delta", "model_completed",
    "usage", "tool_started", "tool_completed", "tool_error", "tool_cancelled", "custom",
    "run_completed", "run_error", "run_cancelled",
]


class AgentRunError(RuntimeError):
    """Safe application error shared by all agent run/stream consumers."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    run_id: str
    sequence: int
    timestamp: str
    data: dict
    source_run_id: str | None = None
    parent_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def to_sse(self) -> str:
        return f"id: {self.run_id}:{self.sequence}\nevent: {self.type}\ndata: {self.to_json()}\n\n"
