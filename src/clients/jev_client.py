"""Shared Jev client construction; evaluation questions belong to callers."""

import math
import os
from dataclasses import dataclass

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient


@dataclass(frozen=True)
class JevSettings:
    """Non-secret connection defaults, independent of ingestion settings."""

    model: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai"
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __post_init__(self):
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("model and base_url cannot be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")

    @classmethod
    def from_env(cls) -> "JevSettings":
        return cls(
            model=os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest"),
            base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
            timeout_seconds=float(os.getenv("TYPESAFE_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("TYPESAFE_MAX_RETRIES", "2")),
        )


def create_jev_client(
    settings: JevSettings | None = None, *, api_key: str | None = None,
) -> TypeSafeClient:
    """Create a reusable SDK client without making a request.

    Explicit settings override environment configuration. The SDK validates the
    explicit key, or reads TYPESAFE_API_KEY when api_key is None. Callers own
    the client: use a context manager or close() to release its resources.
    Environment loading from .env belongs to application entry points.
    """
    settings = settings if settings is not None else JevSettings.from_env()
    return TypeSafeClient(
        api_key=api_key,
        model=settings.model,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        retry=RetryPolicy(max_retries=settings.max_retries),
    )


def create_async_jev_client(
    settings: JevSettings | None = None, *, api_key: str | None = None,
) -> AsyncTypeSafeClient:
    """Async counterpart for cancellable tool middleware; callers own cleanup."""
    settings = settings if settings is not None else JevSettings.from_env()
    return AsyncTypeSafeClient(
        api_key=api_key, model=settings.model, base_url=settings.base_url,
        timeout=settings.timeout_seconds, retry=RetryPolicy(max_retries=settings.max_retries),
    )
