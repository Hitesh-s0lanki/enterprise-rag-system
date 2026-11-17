"""Normalize LangChain callback events without coupling to an agent's tools."""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from .events import AgentEvent, AgentRunError
from .logger import get_logger

logger = get_logger(__name__)


class RunCancelled(Exception):
    pass


def extract_text(content) -> str:
    """Only public text blocks; never reasoning or arbitrary provider fields."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content
                       if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
                       and isinstance(block.get("text"), str))
    return ""


def json_data(value):
    """Unknown objects must not be stringified into potentially sensitive output."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError):
        return {"unavailable": "Event payload is not JSON serializable"}


async def next_event(iterator, cancel_event: asyncio.Event | None, deadline: float):
    next_task = asyncio.ensure_future(anext(iterator))
    pending = {next_task}
    cancel_task = None
    if cancel_event is not None:
        cancel_task = asyncio.create_task(cancel_event.wait())
        pending.add(cancel_task)
    try:
        done, _ = await asyncio.wait(pending, timeout=max(0, deadline - perf_counter()),
                                     return_when=asyncio.FIRST_COMPLETED)
        if cancel_task is not None and cancel_task in done:
            raise RunCancelled
        if not done:
            raise TimeoutError
        return next_task.result()
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def error_details(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, AgentRunError):
        return exc.code, str(exc)
    if isinstance(exc, PermissionError):
        return "access_denied", "You no longer have access to this agent operation."
    if isinstance(exc, TimeoutError):
        return "timeout", "The agent exceeded its execution time limit."
    if type(exc).__name__ in {"ModelCallLimitExceededError", "ToolCallLimitExceededError", "GraphRecursionError"}:
        return "call_limit", "The agent exceeded its configured call limit."
    return "agent_failed", "The agent could not complete this request."


class AgentEventStream:
    """Per-run event state: IDs, sequencing, usage and tool/model lifecycle."""

    def __init__(self, *, agent: str | None = None):
        self.agent = agent
        self.run_id = str(uuid4())
        self.started_at = perf_counter()
        self.sequence = 0
        self.active_tools = {}
        self.model_calls = self.tool_calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "complete": True}
        self.final_message = None

    def event(self, kind, data, raw=None) -> AgentEvent:
        item = AgentEvent(kind, self.run_id, self.sequence, datetime.now(timezone.utc).isoformat(), data,
                          str(raw["run_id"]) if raw and raw.get("run_id") else None,
                          tuple(str(parent) for parent in raw.get("parent_ids", [])) if raw else ())
        self.sequence += 1
        if kind not in {"text_delta", "tool_call_delta"}:
            level = "error" if kind in {"run_error", "tool_error"} else "info"
            counts = data.get("aggregate", data.get("usage", {}))
            getattr(logger, level)("agent." + kind, extra={
                "run_id": self.run_id, "agent": self.agent, "sequence": item.sequence,
                "source_run_id": item.source_run_id, "tool": data.get("name") if kind.startswith("tool_") else None,
                "error_code": data.get("code"), "duration_ms": data.get("duration_ms"),
                "model_calls": data.get("model_calls"), "tool_calls": data.get("tool_calls"),
                **{key: counts.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")
                   if isinstance(counts, dict)},
            })
        return item

    def metrics(self) -> dict:
        return {"model_calls": self.model_calls, "tool_calls": self.tool_calls, "usage": dict(self.usage),
                "duration_ms": (perf_counter() - self.started_at) * 1000}

    def process(self, raw: dict, *, authorize: Callable[[], None]) -> list[AgentEvent]:
        kind, data, name = raw["event"], raw.get("data", {}), raw.get("name", "")
        events = []
        if kind == "on_chat_model_start":
            authorize()
            self.model_calls += 1
            events.append(self.event("model_started", {"name": name}, raw))
        elif kind == "on_chat_model_stream":
            authorize()
            # Tool-private rewrites contribute model usage but are not answer text.
            if "query_enhancement" in raw.get("tags", []):
                return events
            chunk = data.get("chunk")
            text = extract_text(getattr(chunk, "content", None))
            if text:
                events.append(self.event("text_delta", {"text": text}, raw))
            for call in getattr(chunk, "tool_call_chunks", []) or []:
                events.append(self.event("tool_call_delta", {
                    "name": call.get("name"), "id": call.get("id"), "index": call.get("index"),
                    "arguments_delta": call.get("args", ""),
                }, raw))
        elif kind == "on_chat_model_end":
            counts = getattr(data.get("output"), "usage_metadata", None)
            if counts:
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    self.usage[key] += counts.get(key, 0)
                events.append(self.event("usage", {"model_usage": json_data(counts),
                                                    "aggregate": dict(self.usage)}, raw))
            else:
                self.usage["complete"] = False
            events.append(self.event("model_completed", {"name": name}, raw))
        elif kind == "on_tool_start":
            authorize()
            self.tool_calls += 1
            self.active_tools[raw["run_id"]] = name
            events.append(self.event("tool_started", {"name": name, "input": json_data(data.get("input"))}, raw))
        elif kind == "on_tool_end":
            authorize()
            self.active_tools.pop(raw["run_id"], None)
            output = data.get("output")
            failed = isinstance(output, ToolMessage) and output.status == "error"
            payload = output.content if isinstance(output, ToolMessage) else output
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    pass
            events.append(self.event("tool_error" if failed else "tool_completed", {
                "name": name, "output": json_data(payload),
                "tool_call_id": output.tool_call_id if isinstance(output, ToolMessage) else None,
            }, raw))
        elif kind == "on_tool_error":
            self.active_tools.pop(raw["run_id"], None)
            events.append(self.event("tool_error", {"name": name, "error": {
                "code": "tool_failed", "message": "Tool execution failed.",
            }}, raw))
        elif kind == "on_custom_event":
            authorize()
            events.append(self.event("custom", {"name": name, "payload": json_data(data)}, raw))
        elif kind == "on_chain_end" and not raw.get("parent_ids"):
            output = data.get("output", {})
            for message in reversed(output.get("messages", []) if isinstance(output, dict) else []):
                if isinstance(message, AIMessage):
                    self.final_message = message
                    break
        return events

    def completion(self) -> dict:
        message = self.final_message
        if message is None or message.tool_calls or not extract_text(message.content):
            raise AgentRunError("incomplete_run", "The agent did not produce a completed answer.")
        return {"answer": extract_text(message.content), **self.metrics()}

    def interrupt_tools(self, code: str, *, cancelled: bool = False) -> list[AgentEvent]:
        events = []
        for tool_run_id, name in self.active_tools.items():
            payload = {"name": name, "code": code} if cancelled else {
                "name": name, "error": {"code": code, "message": "Tool execution did not complete."},
            }
            events.append(self.event("tool_cancelled" if cancelled else "tool_error", payload, {"run_id": tool_run_id}))
        self.active_tools.clear()
        return events
