"""Pinecone Documents API client and hybrid-ready index schema (SDK 10.x)."""

import os

from pinecone import DenseVectorField, Pinecone, SchemaBuilder, StringField

from src.data_ingestion.config import IngestionSettings


def build_hybrid_schema(settings: IngestionSettings) -> dict:
    """OpenAI dense vectors plus server-managed BM25 on the same chunk text."""
    return (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=settings.embedding_dimensions, metric="cosine")
        .add_string_field("chunk_text", full_text_search={"language": "en"})
        .build()
    )


def create_pinecone_client() -> Pinecone:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise ValueError("PINECONE_API_KEY is required for live ingestion or retrieval")
    return Pinecone(api_key=key)


def validate_embedding_mode(info, settings: IngestionSettings) -> str:
    """Accept supplied vectors or matching OpenAI integration, never another model."""
    fields = info.schema.fields
    dense, text = fields.get("embedding"), fields.get("chunk_text")
    if not isinstance(text, StringField) or not text.full_text_search:
        raise ValueError("Index must contain BM25-enabled chunk_text")
    integrated = text.embed
    if integrated is not None:
        if (integrated.model == settings.embedding_model
            and integrated.dimension == settings.embedding_dimensions
            and integrated.metric == "cosine"):
            return "integrated"
        raise ValueError("Integrated embedding model/dimensions must match the configured OpenAI model")
    if (isinstance(dense, DenseVectorField) and dense.dimension == settings.embedding_dimensions
        and dense.metric == "cosine"):
        return "external"
    raise ValueError("Index must contain matching embedding vectors or matching integrated OpenAI embeddings")


def get_index_embedding_mode(client: Pinecone, settings: IngestionSettings) -> str:
    return validate_embedding_mode(client.indexes.describe(name=settings.index_name), settings)


def get_pinecone_index(client: Pinecone, settings: IngestionSettings, *, create: bool = False):
    """Validate an existing index; create only when explicitly requested."""
    if not client.indexes.exists(name=settings.index_name):
        if not create:
            raise ValueError("Pinecone index does not exist. Use --create-index to create it.")
        client.indexes.create(
            name=settings.index_name, schema=build_hybrid_schema(settings),
            deployment={"deployment_type": "managed", "cloud": settings.cloud, "region": settings.region},
            timeout=120,
        )
    info = client.indexes.describe(name=settings.index_name)
    validate_embedding_mode(info, settings)
    if not info.status.ready:
        raise ValueError("Pinecone index is not ready; retry after it becomes ready")
    return client.index(host=info.host)
