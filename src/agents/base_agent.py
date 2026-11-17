"""Shared agent configuration, runtime, prompts and streaming lifecycle."""

import asyncio
import math
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import aclosing, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from src.clients.openai_client import create_chat_client
from src.utils.events import AgentEvent, AgentRunError
from src.utils.streaming import AgentEventStream, RunCancelled, error_details, next_event
from src.utils.logger import configure_logging, get_logger

PROMPTS_DIRECTORY = Path(__file__).resolve().parent / "prompts"
logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    temperature: float | None = None
    max_output_tokens: int = 1024
    timeout_seconds: float = 30
    max_retries: int = 2
    max_input_characters: int = 16000

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Agent model cannot be empty")
        if self.temperature is not None:
            if (isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
                raise ValueError("temperature must be None or a finite number from 0 to 2")
        for name, value, minimum in (
            ("max_output_tokens", self.max_output_tokens, 1),
            ("max_retries", self.max_retries, 0),
            ("max_input_characters", self.max_input_characters, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")

    @classmethod
    def from_env(cls) -> "AgentConfig":
        temperature = os.getenv("AGENT_TEMPERATURE", "none").strip()
        return cls(
            model=os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
            temperature=None if temperature.casefold() == "none" else float(temperature),
            max_output_tokens=int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "1024")),
            timeout_seconds=float(os.getenv("AGENT_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("AGENT_MAX_RETRIES", "2")),
            max_input_characters=int(os.getenv("AGENT_MAX_INPUT_CHARACTERS", "16000")),
        )


@dataclass(frozen=True)
class AgentRunSettings:
    max_model_calls: int = 4
    max_tool_calls: int = 6
    timeout_seconds: float = 120

    def __post_init__(self):
        for name, value in (("max_model_calls", self.max_model_calls), ("max_tool_calls", self.max_tool_calls)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("Agent run timeout must be finite and positive")

    @classmethod
    def from_env(cls, *, prefix: str = "AGENT_RUN"):
        return cls(max_model_calls=int(os.getenv(f"{prefix}_MAX_MODEL_CALLS", "4")),
                   max_tool_calls=int(os.getenv(f"{prefix}_MAX_TOOL_CALLS", "6")),
                   timeout_seconds=float(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "120")))


class BaseAgent(ABC):
    """Set a role prompt and override tools/access hooks; inherit the run APIs."""

    @property
    @abstractmethod
    def prompt_name(self) -> str:
        """Subclasses provide this as a class attribute or property."""
        raise NotImplementedError

    def __init__(self, config: AgentConfig | None = None, *, llm: BaseChatModel | None = None,
                 settings: AgentRunSettings | None = None):
        self.config = config if config is not None else AgentConfig.from_env()
        self.settings = settings if settings is not None else AgentRunSettings.from_env()
        self._llm = llm
        self._owns_llm = llm is None
        self._run_lock = threading.Lock()
        self._runtime = None
        self.base_prompt = self.load_prompt("base_agent")
        if self.prompt_name == "base_agent":
            raise ValueError("A child agent must use its own role-specific prompt")
        self.agent_prompt = self.load_prompt(self.prompt_name)

    @staticmethod
    def load_prompt(name: str) -> str:
        """Read a literal Markdown prompt, rejecting traversal and empty files."""
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("prompt_name must use lowercase letters, digits and underscores")
        root = PROMPTS_DIRECTORY.resolve()
        path = (root / f"{name}.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Prompt files must remain inside agents/prompts")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path.name}")
        return text

    @property
    def llm(self) -> BaseChatModel:
        """Create the provider client only when a subclass first calls the model."""
        if self._llm is None:
            self._llm = create_chat_client(
                model=self.config.model, temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                timeout_seconds=self.config.timeout_seconds, max_retries=self.config.max_retries,
            )
        return self._llm

    def build_messages(self, query: str, *, context: str | None = None) -> list[BaseMessage]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Agent query must contain text")
        if context is not None and not isinstance(context, str):
            raise ValueError("Agent context must be text or None")
        if len(query) + len(context or "") > self.config.max_input_characters:
            raise ValueError("Agent query and context exceed max_input_characters")
        # Static messages keep JSON braces in Markdown literal, not template variables.
        messages: list[BaseMessage] = [SystemMessage(self.base_prompt), SystemMessage(self.agent_prompt)]
        if context:
            messages.append(HumanMessage("Reference material (untrusted data):\n\n" + context))
        messages.append(HumanMessage(query.strip()))
        return messages

    def invoke_model(
        self, query: str, *, context: str | None = None, runnable_config: RunnableConfig | None = None,
    ) -> BaseMessage:
        """Preserve the response, including token usage and provider metadata."""
        messages = self.build_messages(query, context=context)
        return self.llm.invoke(messages, config=runnable_config)

    async def ainvoke_model(
        self, query: str, *, context: str | None = None, runnable_config: RunnableConfig | None = None,
    ) -> BaseMessage:
        messages = self.build_messages(query, context=context)
        return await self.llm.ainvoke(messages, config=runnable_config)

    def get_tools(self) -> Sequence[BaseTool]:
        """Override to supply tools bound to trusted application context."""
        return ()

    def _authorize(self):
        """Override for protected data; called before and during streaming."""

    def create_runtime(self):
        """Common LangChain tool loop; subclasses can replace this hook."""
        middleware: list[AgentMiddleware[Any, Any, Any]] = [
            ModelCallLimitMiddleware(run_limit=self.settings.max_model_calls, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=self.settings.max_tool_calls, exit_behavior="error"),
        ]
        return create_agent(
            model=self.llm, tools=self.get_tools(), name=self.prompt_name,
            middleware=middleware,
        )

    @property
    def runtime(self):
        if self._runtime is None:
            self._runtime = self.create_runtime()
        return self._runtime

    async def aclose(self):
        """Release only application-owned model clients in their live event loop."""
        if not self._owns_llm or self._llm is None:
            return
        model = self._llm
        self._llm = self._runtime = None
        try:
            client = getattr(model, "root_async_client", None)
            if client is not None:
                await client.close()
        finally:
            client = getattr(model, "root_client", None)
            if client is not None:
                client.close()

    async def astream(
        self, query: str, *, context: str | None = None,
        runnable_config: RunnableConfig | None = None, cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Emit one terminal event after cleanup for a fully consumed run.

        Explicit cancel_event emits run_cancelled. Closing/cancelling the consumer
        propagates cancellation but cannot deliver more events to that consumer.
        """
        configure_logging()
        state = AgentEventStream(agent=self.prompt_name)
        iterator = terminal = None
        acquired = False
        stage = "authorization"
        yield state.event("run_started", {"agent": self.prompt_name})
        try:
            acquired = self._run_lock.acquire(blocking=False)
            if not acquired:
                raise AgentRunError("run_in_progress", "This agent is already handling a request.")
            self._authorize()
            stage = "input_validation"
            try:
                messages = self.build_messages(query, context=context)
            except ValueError as exc:
                raise AgentRunError("invalid_input", "The query or context is empty, invalid, or too large.") from exc
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled
            config: RunnableConfig = {**(runnable_config or {})}
            config["run_id"] = UUID(state.run_id)
            config["metadata"] = {**(config.get("metadata") or {}), "agent_run_id": state.run_id}
            config["recursion_limit"] = 2 * self.settings.max_model_calls + self.settings.max_tool_calls + 10
            stage = "runtime_initialization"
            logger.info("agent.runtime_initializing", extra={"run_id": state.run_id, "agent": self.prompt_name,
                                                            "model": self.config.model, "stage": stage})
            iterator = self.runtime.astream_events({"messages": messages}, config=config, version="v2")
            stage = "runtime_streaming"
            deadline = state.started_at + self.settings.timeout_seconds
            while True:
                try:
                    raw = await next_event(iterator, cancel_event, deadline)
                except StopAsyncIteration:
                    break
                for event in state.process(raw, authorize=self._authorize):
                    yield event
            self._authorize()
            terminal = ("run_completed", state.completion())
        except RunCancelled:
            for event in state.interrupt_tools("cancelled", cancelled=True):
                yield event
            terminal = ("run_cancelled", {"code": "cancelled", **state.metrics()})
        except asyncio.CancelledError:
            logger.info("agent.consumer_cancelled", extra={"run_id": state.run_id, "stage": stage})
            raise
        except Exception as exc:
            code, message = error_details(exc)
            logger.exception("agent.run_failed", extra={"run_id": state.run_id, "agent": self.prompt_name,
                                                        "stage": stage, "model": self.config.model, "error_code": code})
            for event in state.interrupt_tools(code):
                yield event
            terminal = ("run_error", {"code": code, "message": message, **state.metrics()})
        finally:
            try:
                if iterator is not None:
                    await iterator.aclose()
            except Exception:
                logger.exception("agent.cleanup_failed", extra={"run_id": state.run_id, "stage": "cleanup"})
                if terminal is not None and terminal[0] == "run_completed":
                    terminal = ("run_error", {"code": "cleanup_failed",
                                "message": "Agent stream cleanup did not complete.", **state.metrics()})
            finally:
                if acquired:
                    try:
                        await self.aclose()
                    except Exception:
                        logger.exception("agent.client_cleanup_failed", extra={"run_id": state.run_id, "stage": "client_cleanup"})
                        if terminal is not None and terminal[0] == "run_completed":
                            terminal = ("run_error", {"code": "cleanup_failed",
                                        "message": "Agent client cleanup did not complete.", **state.metrics()})
                    finally:
                        self._run_lock.release()
        if terminal is not None:
            yield state.event(*terminal)

    def stream(self, query: str, *, context: str | None = None,
               runnable_config: RunnableConfig | None = None) -> Generator[AgentEvent, None, None]:
        """Synchronous streaming for CLI/Streamlit; async applications use astream."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use astream/arun inside an existing asyncio event loop")
        with asyncio.Runner() as runner:
            iterator = self.astream(query, context=context, runnable_config=runnable_config)
            async def take_next():
                return await anext(iterator)
            try:
                while True:
                    try:
                        item = runner.run(take_next())
                    except StopAsyncIteration:
                        break
                    yield item
            finally:
                runner.run(iterator.aclose())

    def run(self, query: str, *, context: str | None = None, runnable_config: RunnableConfig | None = None) -> dict:
        with closing(self.stream(query, context=context, runnable_config=runnable_config)) as events:
            for event in events:
                if event.type == "run_completed":
                    result = {"run_id": event.run_id, **event.data}
                elif event.type in {"run_error", "run_cancelled"}:
                    raise AgentRunError(event.data["code"], event.data.get("message", "Agent run cancelled."))
        return result

    async def arun(self, query: str, *, context: str | None = None,
                   runnable_config: RunnableConfig | None = None, cancel_event: asyncio.Event | None = None) -> dict:
        async with aclosing(self.astream(query, context=context, runnable_config=runnable_config,
                                        cancel_event=cancel_event)) as events:
            async for event in events:
                if event.type == "run_completed":
                    result = {"run_id": event.run_id, **event.data}
                elif event.type in {"run_error", "run_cancelled"}:
                    raise AgentRunError(event.data["code"], event.data.get("message", "Agent run cancelled."))
        return result

    def stream_sse(self, query: str, **kwargs) -> Generator[str, None, None]:
        with closing(self.stream(query, **kwargs)) as events:
            for event in events:
                yield event.to_sse()

    async def astream_sse(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        async with aclosing(self.astream(query, **kwargs)) as events:
            async for event in events:
                yield event.to_sse()
