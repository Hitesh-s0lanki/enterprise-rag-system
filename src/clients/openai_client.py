"""Shared LangChain OpenAI chat-model factory."""

import os

from langchain_openai import ChatOpenAI
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient


def create_chat_client(
    *, model: str, temperature: float | None, max_output_tokens: int,
    timeout_seconds: float, max_retries: int,
) -> ChatOpenAI:
    """Construct a client without making a request; entry points load .env."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for agent model calls")
    return ChatOpenAI(
        model=model, temperature=temperature, max_tokens=max_output_tokens,
        timeout=timeout_seconds, max_retries=max_retries,
        stream_usage=True,
        # LangChain's default transport cache can retain connections belonging
        # to a closed Streamlit/synchronous-run event loop. Own these clients.
        http_client=DefaultHttpxClient(), http_async_client=DefaultAsyncHttpxClient(),
    )
