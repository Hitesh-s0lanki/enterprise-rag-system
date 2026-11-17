"""Firecrawl Streamable HTTP MCP client with bounded reconnect retries."""

import asyncio
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamable_http_client
from src.utils.logger import get_logger, current_run_id

logger = get_logger(__name__)

DEFAULT_MCP_URL = "https://mcp.firecrawl.dev/v2/mcp"
READ_TOOLS = frozenset({"firecrawl_search", "firecrawl_scrape", "firecrawl_map"})


@dataclass(frozen=True)
class FirecrawlSettings:
    url: str = DEFAULT_MCP_URL
    timeout_seconds: float = 30
    max_retries: int = 2
    retry_delay_seconds: float = 1

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Firecrawl MCP URL must be HTTPS without credentials, query or fragment")
        for name in ("timeout_seconds", "retry_delay_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be an integer from 0 to 5")

    @classmethod
    def from_env(cls):
        return cls(
            url=os.getenv("FIRECRAWL_MCP_URL", DEFAULT_MCP_URL),
            timeout_seconds=float(os.getenv("FIRECRAWL_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("FIRECRAWL_MAX_RETRIES", "2")),
            retry_delay_seconds=float(os.getenv("FIRECRAWL_RETRY_DELAY_SECONDS", "1")),
        )


class FirecrawlError(RuntimeError):
    """Public error; provider bodies and headers are never included."""


def _retryable(error: BaseException) -> bool:
    # SDK transport task groups can wrap HTTP errors in ExceptionGroup.
    if isinstance(error, BaseExceptionGroup):
        return all(_retryable(item) for item in error.exceptions)
    if isinstance(error, (PermissionError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 429} or error.response.status_code >= 500
    if isinstance(error, McpError):
        return error.error.code not in {-32600, -32601, -32602}
    # Stream closure errors from the SDK's async transport may have no HTTP status.
    return True


class FirecrawlClient:
    def __init__(self, api_key: str, settings: FirecrawlSettings | None = None):
        if not api_key or not api_key.strip():
            raise ValueError("FIRECRAWL_API_KEY is required")
        self._api_key = api_key.strip()
        self.settings = settings if settings is not None else FirecrawlSettings.from_env()

    async def call_tool(self, name: str, arguments: dict, *, authorize: Callable[[], object] | None = None) -> dict:
        if name not in READ_TOOLS:
            raise ValueError("Unsupported Firecrawl tool")
        for attempt in range(self.settings.max_retries + 1):
            if authorize is not None:
                authorize()
            try:
                # A fresh session on every attempt also recovers broken initialization.
                async with asyncio.timeout(self.settings.timeout_seconds):
                    result = await self._call_once(name, arguments)
                if result.isError:
                    raise FirecrawlError("Firecrawl MCP tool failed")
                if authorize is not None:
                    authorize()
                return {
                    "source": "public_web", "tool": name,
                    "content": [block.model_dump(mode="json", exclude_none=True) for block in result.content],
                    "structured_content": result.structuredContent,
                }
            except asyncio.CancelledError:
                raise
            except PermissionError:
                raise
            except Exception as error:
                if attempt == self.settings.max_retries or not _retryable(error):
                    logger.exception("firecrawl.request_failed", extra={"run_id": current_run_id(), "tool": name})
                    raise FirecrawlError("Firecrawl MCP request failed; no completed web result is available") from None
            await asyncio.sleep(min(self.settings.retry_delay_seconds * 2 ** attempt, 10))
        raise AssertionError("Unreachable retry state")

    async def _call_once(self, name: str, arguments: dict):
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(self.settings.timeout_seconds), follow_redirects=False,
        ) as http_client:
            async with streamable_http_client(self.settings.url, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.settings.timeout_seconds)) as session:
                    await session.initialize()
                    return await session.call_tool(name, arguments)


def create_firecrawl_client() -> FirecrawlClient | None:
    """Build locally; missing credentials leave the optional tools disabled."""
    enabled = os.getenv("FIRECRAWL_ENABLED", "true").strip().lower()
    if enabled not in {"true", "false"}:
        raise ValueError("FIRECRAWL_ENABLED must be true or false")
    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if enabled == "false" or not key or key == "your_firecrawl_api_key_here":
        return None
    return FirecrawlClient(key)
