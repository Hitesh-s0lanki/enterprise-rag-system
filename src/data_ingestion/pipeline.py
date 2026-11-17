"""Prepare locally, then explicitly embed and upload document chunks."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.db.auth import AuthorizationService, Permission

from .config import IngestionSettings
from .documents import discover_text_files, load_text_document, split_document
from .embeddings import embed_texts


def prepare_documents(
    source: str | Path, settings: IngestionSettings, *, updated_by: str,
    allowed_roles: list[str], updated_at: datetime | None = None,
) -> list[Document]:
    """Load and split without credentials, API calls, or writing files."""
    source = Path(source).resolve(strict=True)
    root = source if source.is_dir() else source.parent
    ingested_at = datetime.now(timezone.utc)
    chunks = []
    for path in discover_text_files(source):
        document = load_text_document(
            path, source_root=root, updated_by=updated_by, allowed_roles=allowed_roles,
            updated_at=updated_at, ingested_at=ingested_at,
        )
        chunks.extend(split_document(document, settings))
    return chunks


def build_records(chunks: list[Document], vectors: list[list[float]] | None = None) -> list[dict]:
    """Store citation text and all source metadata beside its dense vector."""
    if vectors is None:
        return [{"_id": chunk.metadata["chunk_id"], "chunk_text": chunk.page_content, **chunk.metadata}
                for chunk in chunks]
    return [
        {"_id": chunk.metadata["chunk_id"], "chunk_text": chunk.page_content,
         "embedding": vector, **chunk.metadata}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def ingest_documents(
    chunks: list[Document], settings: IngestionSettings, *, embedding_client: Embeddings | None, index,
    actor_id: UUID, authorization: AuthorizationService, embedding_mode: str = "external",
) -> dict:
    """Upsert bounded batches; propagate failures rather than report partial success."""
    if not chunks:
        raise ValueError("No document chunks to ingest")
    if embedding_mode not in {"external", "integrated"}:
        raise ValueError("Unknown embedding mode")
    if embedding_mode == "external" and embedding_client is None:
        raise ValueError("An OpenAI embedding client is required for external vectors")
    # Validate every document before the first embedding or upload side effect.
    actor = authorization.require_access(
        actor_id, Permission.WRITE_KNOWLEDGE, namespace=settings.namespace,
    )
    for chunk in chunks:
        roles = chunk.metadata.get("allowed_roles")
        if not isinstance(roles, list) or not roles:
            raise PermissionError("Every document must declare allowed reader roles")
        authorization.require_access(
            actor_id, Permission.WRITE_KNOWLEDGE, namespace=settings.namespace, allowed_roles=roles,
        )
    uploaded = 0
    for offset in range(0, len(chunks), settings.batch_size):
        batch = chunks[offset:offset + settings.batch_size]
        vectors = embed_texts(
            [chunk.page_content for chunk in batch], embedding_client, settings.embedding_dimensions,
        ) if embedding_mode == "external" else None
        records = build_records(batch, vectors)
        for record in records:
            record.update({"ingested_by_user_id": str(actor.id), "ingested_by": actor.username})
        response = index.documents.upsert(namespace=settings.namespace, documents=records)
        if response.upserted_count != len(batch):
            raise RuntimeError("Pinecone did not acknowledge all chunks; ingestion is incomplete")
        uploaded += response.upserted_count
    return {
        "index": settings.index_name, "namespace": settings.namespace,
        "files": len({c.metadata["document_id"] for c in chunks}),
        "chunks_accepted": uploaded,
        "ingested_by": actor.username,
        "embedding_mode": embedding_mode,
        "revisions": sorted({c.metadata["revision_id"] for c in chunks}),
        "document_revisions": {
            c.metadata["document_id"]: c.metadata["revision_id"] for c in chunks
        },
        "note": "Accepted by Pinecone; search visibility is asynchronous.",
    }
