"""Mem0 Platform client for entity-scoped agent memories (stdlib HTTP)."""

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://api.mem0.ai"
_SCOPE_NAMES = ("user_id", "agent_id", "app_id", "run_id")


@dataclass(frozen=True)
class Mem0Settings:
    """Non-secret connection defaults; the API key is supplied separately."""

    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    @classmethod
    def from_env(cls) -> "Mem0Settings":
        return cls(
            base_url=os.getenv("MEM0_BASE_URL", DEFAULT_BASE_URL),
            timeout_seconds=float(os.getenv("MEM0_TIMEOUT_SECONDS", "30")),
        )


class Mem0Error(RuntimeError):
    """Raised when the Mem0 API rejects a request or is unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class Mem0Client:
    """Minimal client for the Mem0 Platform v3 memory endpoints.

    Methods return the parsed JSON body as-is. Inferred adds may return a queued
    event; infer=False returns synchronously stored results. Search returns scored
    memories under "results". Adds and searches require explicit entity scope;
    callers fetching by memory ID must validate the returned scope themselves.
    """

    def __init__(self, api_key: str, settings: Mem0Settings | None = None):
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")
        self._api_key = api_key
        self._settings = settings if settings is not None else Mem0Settings()

    def add(
        self,
        messages: list[dict[str, str]] | str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool | None = None,
        immutable: bool | None = None,
    ) -> dict[str, Any]:
        """Extract memories, or store verbatim when infer=False.

        Accepts [{role, content}, ...] messages or a plain user string.
        Inference normally decomposes the conversation into individual facts.
        Use infer=False and immutable=True to preserve a cache payload verbatim.
        """
        payload: dict[str, Any] = {
            "messages": self._validate_messages(messages),
            **self._validate_scope(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id),
        }
        if metadata is not None:
            payload["metadata"] = metadata
        for name, value in (("infer", infer), ("immutable", immutable)):
            if value is not None:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a boolean")
                payload[name] = value
        return self._post("/v3/memories/add/", payload)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int | None = None, rerank: bool | None = None,
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Search scoped memories; results are ranked by relevance score."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(filters, dict) or not filters:
            raise ValueError(
                "filters are required, e.g. {'user_id': 'user123'}; use the same "
                "scope the memories were added with"
            )
        payload = {"query": query, "filters": filters}
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
                raise ValueError("top_k must be an integer between 1 and 1000")
            payload["top_k"] = top_k
        if rerank is not None:
            if not isinstance(rerank, bool):
                raise ValueError("rerank must be a boolean")
            payload["rerank"] = rerank
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            payload["metadata"] = metadata
        return self._post("/v3/memories/search/", payload)

    def get(self, memory_id: str) -> dict[str, Any]:
        """Fetch one verbatim memory by ID; the caller must validate its scope."""
        if not isinstance(memory_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", memory_id):
            raise ValueError("memory_id must be a valid identifier")
        return self._request("GET", f"/v1/memories/{memory_id}/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = Request(
            self._settings.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            method=method,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self._settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, dict):
                    raise Mem0Error("Mem0 returned a non-object JSON response")
                return body
        except (ValueError, UnicodeDecodeError) as error:
            raise Mem0Error("Mem0 returned invalid JSON") from error
        except urllib.error.HTTPError as error:
            raise Mem0Error(
                f"Mem0 API request failed with status {error.code}: {_error_detail(error)}",
                status_code=error.code,
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else str(error)
            raise Mem0Error(
                f"Mem0 API unreachable at {self._settings.base_url}: {reason}"
            ) from error

    @staticmethod
    def _validate_scope(**scopes: str | None) -> dict[str, str]:
        values = {name: value for name, value in scopes.items() if value is not None}
        if not values:
            raise ValueError(
                "at least one scope (user_id, agent_id, app_id, run_id) is required"
            )
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        return values

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]] | str) -> list[dict[str, str]]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} items")
        validated = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str) or not message["role"].strip()
                or not isinstance(message.get("content"), str) or not message["content"].strip()
            ):
                raise ValueError("each message needs non-empty role and content strings")
            validated.append({"role": message["role"], "content": message["content"]})
        return validated


def create_mem0_client(
    settings: Mem0Settings | None = None, *, api_key: str | None = None
) -> Mem0Client:
    """Create a reusable client without making a request.

    Explicit settings override environment configuration. The key comes from
    the explicit argument or MEM0_API_KEY; missing keys fail locally without
    any network call. Environment loading from .env belongs to application
    entry points.
    """
    key = api_key if api_key is not None else os.getenv("MEM0_API_KEY")
    if not key or not key.strip():
        raise ValueError("MEM0_API_KEY is required for live memory operations")
    return Mem0Client(key, settings if settings is not None else Mem0Settings.from_env())


def _error_detail(error: urllib.error.HTTPError) -> str:
    body = ""
    try:
        body = error.read().decode("utf-8")
        parsed = json.loads(body)
        detail = parsed.get("detail", body) if isinstance(parsed, dict) else body
    except (OSError, UnicodeDecodeError, ValueError):
        detail = body
    return (detail or str(error.reason))[:500]
