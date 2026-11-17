"""UTF-8 loading, source metadata and recursive character chunking."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import IngestionSettings

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})


def discover_text_files(source: str | Path) -> list[Path]:
    """Resolve one supported file or recursively discover supported files."""
    source = Path(source).resolve(strict=True)
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported text file: {source.name}")
        return [source]
    files = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.resolve().is_relative_to(source)
    )
    if not files:
        raise ValueError(f"No supported text files found in {source}")
    return files


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_text_document(
    path: Path, *, source_root: Path, updated_by: str,
    allowed_roles: list[str], updated_at: datetime | None = None,
    ingested_at: datetime | None = None,
) -> Document:
    """Require updater identity from the caller; never infer it from the OS."""
    if not updated_by.strip():
        raise ValueError("updated_by must identify the user who last updated the source")
    if not allowed_roles or any(not role.strip() for role in allowed_roles):
        raise ValueError("At least one non-empty allowed role is required")
    path = path.resolve(strict=True)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported text file: {path.name}")
    source = path.relative_to(source_root.resolve()).as_posix()
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError(f"Empty document: {source}")
    last_update = updated_at or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    metadata = {
        "source": source,
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "document_id": hashlib.sha256(source.encode()).hexdigest(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "last_updated_by": updated_by.strip(),
        "last_updated_at": utc_timestamp(last_update),
        "last_updated_at_epoch": last_update.timestamp(),
        "last_updated_at_source": "provided" if updated_at else "filesystem_mtime",
        "ingested_at": utc_timestamp(ingested_at or datetime.now(timezone.utc)),
        "allowed_roles": sorted(set(role.strip() for role in allowed_roles)),
    }
    return Document(page_content=text, metadata=metadata)


def split_document(document: Document, settings: IngestionSettings) -> list[Document]:
    """Carry metadata into every chunk and produce repeatable revision/chunk IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""], add_start_index=True,
    )
    # Ingestion time does not create a new revision when the source is unchanged.
    revision_data = {
        "metadata": {k: v for k, v in document.metadata.items() if k != "ingested_at"},
        "text": document.page_content,
        "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
        "splitter_version": 1,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
    }
    revision = hashlib.sha256(json.dumps(revision_data, sort_keys=True).encode()).hexdigest()
    chunks = splitter.split_documents([document])
    for number, chunk in enumerate(chunks):
        chunk.metadata.update({
            "revision_id": revision,
            "chunk_index": number, "chunk_count": len(chunks),
            "chunk_id": f"{document.metadata['document_id']}:{revision}:{number:06d}",
            "chunk_sha256": hashlib.sha256(chunk.page_content.encode()).hexdigest(),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        })
    return chunks
