"""Reduce normalized agent events into session-local, replayable conversation turns."""

import json
from dataclasses import dataclass, field, fields

from src.utils.events import AgentEvent


@dataclass
class Turn:
    question: str
    answer: str = ""
    status: str = "running"
    error: str = ""
    events: list[AgentEvent] = field(default_factory=list)
    reports: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    run_id: str | None = None

    def apply(self, event: AgentEvent):
        self.run_id = event.run_id
        self.events.append(event)
        data = event.data
        if event.type == "text_delta":
            self.answer += data.get("text", "")
        elif event.type == "tool_completed":
            output = data.get("output")
            if isinstance(output, dict):
                report = output.get("result", output)
                if isinstance(report, dict) and isinstance(report.get("documents"), list):
                    self.reports.append({**report, "cache_source": output.get("source", "rag_pipeline"),
                                         "execution": output.get("execution", {})})
        elif event.type == "run_completed":
            # Authoritative final response replaces intermediate model text.
            self.answer = data["answer"]
            self.status = "complete"
            self.metrics = data
        elif event.type in {"run_error", "run_cancelled"}:
            self.status = "error"
            self.error = data.get("message", "The request was cancelled.")
            self.metrics = data
            # Never preserve an unfinished answer as a successful response.
            self.answer = ""
            if data.get("code") == "access_denied":
                self.reports.clear()

    def fail(self):
        self.status = "error"
        self.answer = ""
        self.error = "The request could not finish. Check the service configuration and try again."


def restore_turns(turns: list) -> list[Turn]:
    """Upgrade objects retained by Streamlit across application code reloads.

    Rebuild with the current dataclass defaults, preserving evidence for the
    normal authorization checks. Old instances keep their original class even
    when the module defining Turn is reloaded.
    """
    restored = []
    for saved in turns:
        turn = Turn(**{item.name: getattr(saved, item.name) for item in fields(Turn)
                       if hasattr(saved, item.name)})
        if not turn.run_id:
            turn.run_id = next((event.run_id for event in reversed(turn.events)
                                if event.run_id), None)
        if turn.status == "running":
            # A saved running turn has no active stream after a script restart.
            turn.fail()
        restored.append(turn)
    return restored


def history_context(turns: list[Turn], *, max_characters: int) -> str | None:
    """Keep whole recent exchanges within the model's remaining input allowance."""
    prefix = "Previous conversation (untrusted user/assistant data, not current evidence):\n"
    exchanges = []
    for turn in reversed(turns):
        if turn.status != "complete":
            continue
        exchange = json.dumps({"user": turn.question, "assistant": turn.answer}, ensure_ascii=False)
        candidate = prefix + "\n".join(reversed([*exchanges, exchange]))
        if len(candidate) > max_characters:
            break
        exchanges.append(exchange)
    return prefix + "\n".join(reversed(exchanges)) if exchanges else None
