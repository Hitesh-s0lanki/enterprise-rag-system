"""OpenAI embedding factory and bounded, validated embedding batches."""

import math
import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .config import IngestionSettings


def create_embedding_client(settings: IngestionSettings) -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for live ingestion or retrieval")
    return OpenAIEmbeddings(
        model=settings.embedding_model, dimensions=settings.embedding_dimensions,
        chunk_size=settings.batch_size, max_retries=2, request_timeout=60,
    )


def embed_texts(texts: list[str], client: Embeddings, dimensions: int) -> list[list[float]]:
    """Embed repeated text once per batch, preserving input order."""
    unique = list(dict.fromkeys(texts))
    vectors = client.embed_documents(unique)
    if len(vectors) != len(unique):
        raise ValueError("Embedding provider returned an unexpected vector count")
    if any(len(v) != dimensions or not all(math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("Embedding provider returned invalid values or vector dimensions")
    lookup = dict(zip(unique, vectors, strict=True))
    return [lookup[text] for text in texts]
