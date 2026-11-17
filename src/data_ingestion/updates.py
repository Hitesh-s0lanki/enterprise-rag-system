"""Authorized replacement content, immutable chunk uploads and revision publication."""

import hashlib
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document

from src.clients.pinecone_client import create_pinecone_client, get_index_embedding_mode, get_pinecone_index
from src.db.auth import AuthorizationService, Permission, Role
from src.db.documents import DocumentRecord, RevisionConflict
from src.db.documents.repository import DocumentRepository

from .config import IngestionSettings
from .documents import SUPPORTED_EXTENSIONS, split_document, utc_timestamp
from .embeddings import create_embedding_client
from .pipeline import ingest_documents


def _hash_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Document and revision IDs must be lowercase SHA-256 hashes")


class DocumentUpdateService:
    """Application-bound identity; only existing registered documents are editable.

    The catalog and authorization adapters must share the same transactional
    database for write-lock protection of local permission and publication checks.
    """

    def __init__(self, *, actor_id: UUID, authorization: AuthorizationService,
                 catalog: DocumentRepository, settings: IngestionSettings | None = None,
                 max_content_characters: int = 100000):
        if not isinstance(actor_id, UUID):
            raise ValueError("actor_id must be an authenticated UUID")
        if isinstance(max_content_characters, bool) or not isinstance(max_content_characters, int) or not 1 <= max_content_characters <= 1000000:
            raise ValueError("max_content_characters must be between 1 and 1000000")
        self.actor_id, self.authorization, self.catalog = actor_id, authorization, catalog
        auth_path = getattr(authorization.repository, "path", None)
        catalog_path = getattr(catalog, "path", None)
        if auth_path is not None and catalog_path is not None and Path(auth_path).resolve() != Path(catalog_path).resolve():
            raise ValueError("Local authorization and document catalog must use the same SQLite database")
        self.settings = settings if settings is not None else IngestionSettings.from_env()
        self.max_content_characters = max_content_characters

    def _authorize(self, record: DocumentRecord):
        required = Permission.READ_KNOWLEDGE | Permission.WRITE_KNOWLEDGE
        actor = self.authorization.require_access(self.actor_id, required, namespace=self.settings.namespace,
                                                  allowed_roles=record.reader_roles)
        self.authorization.require_access(self.actor_id, required, namespace=self.settings.namespace,
                                          allowed_roles=record.writer_roles)
        if record.namespace != self.settings.namespace:
            raise PermissionError("Document is outside the bound namespace")
        return actor

    def _load(self, document_id):
        self.authorization.require_access(self.actor_id, Permission.READ_KNOWLEDGE | Permission.WRITE_KNOWLEDGE,
                                          namespace=self.settings.namespace)
        _hash_id(document_id)
        record = self.catalog.get(self.settings.namespace, document_id)
        if record is None:
            raise PermissionError("Document is unavailable")
        self._authorize(record)
        return record

    def get_document_for_update(self, document_id: str) -> dict:
        record = self._load(document_id)
        return {"document_id": record.document_id, "revision_id": record.revision_id,
                "content": record.content, "metadata": record.metadata,
                "update_in_progress": record.pending_update_id is not None}

    def register_existing_document(self, *, document_id: str, revision_id: str, content: str,
                                   metadata: dict, reader_roles: list[str], writer_roles: list[str]) -> None:
        """Trusted maintainer bootstrap only; never exposed as an agent tool.

        The caller verifies that revision/content/metadata describe the intended
        existing provider version. This does not discover ACLs from Pinecone.
        """
        def authorize(record):
            self.authorization.require_access(self.actor_id, Permission.MANAGE_KNOWLEDGE | Permission.WRITE_KNOWLEDGE,
                                              namespace=self.settings.namespace)
        authorize(None)
        _hash_id(document_id)
        _hash_id(revision_id)
        self._validate_content(content)
        if not reader_roles or not writer_roles:
            raise ValueError("Explicit reader and writer roles are required")
        readers = tuple(sorted({Role(role).value for role in reader_roles}))
        writers = tuple(sorted({Role(role).value for role in writer_roles}))
        if not isinstance(metadata, dict) or metadata.get("document_id") != document_id:
            raise ValueError("Source metadata must match the registered document ID")
        for key in ("source", "file_name", "file_type", "last_updated_by", "last_updated_at"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"Source metadata requires {key}")
        if metadata["file_type"] not in SUPPORTED_EXTENSIONS:
            raise ValueError("Only text and Markdown documents can be updated")
        metadata = {**metadata, "allowed_roles": list(readers)}
        record = DocumentRecord(self.settings.namespace, document_id, revision_id, content, metadata, readers, writers, None)
        self.catalog.register(record, str(self.actor_id), authorize)

    def _validate_content(self, content):
        if not isinstance(content, str) or not content.strip() or len(content) > self.max_content_characters:
            raise ValueError(f"Replacement content must contain 1-{self.max_content_characters} characters")

    def update_document(self, document_id: str, expected_revision: str, content: str) -> dict:
        current = self._load(document_id)
        _hash_id(expected_revision)
        self._validate_content(content)
        if current.revision_id != expected_revision or current.pending_update_id:
            raise RevisionConflict("Document changed or an update is already in progress")
        if content == current.content:
            return {"status": "unchanged", "document_id": document_id, "revision_id": current.revision_id}
        actor = self._authorize(current)
        now = datetime.now(timezone.utc)
        metadata = {key: current.metadata[key] for key in ("source", "file_name", "file_type")}
        metadata.update({
            "document_id": document_id, "file_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "allowed_roles": list(current.reader_roles), "last_updated_by": actor.username,
            "last_updated_at": utc_timestamp(now), "last_updated_at_epoch": now.timestamp(),
            "last_updated_at_source": "authenticated_edit", "ingested_at": utc_timestamp(now),
        })
        chunks = split_document(Document(page_content=content, metadata=metadata), self.settings)
        revision_id = chunks[0].metadata["revision_id"]
        staged = replace(current, revision_id=revision_id, content=content, metadata=metadata, pending_update_id=None)
        token = self.catalog.stage(staged, expected_revision, str(self.actor_id), self._authorize)
        try:
            self._authorize(self._load(document_id))
            self._upload_and_verify(chunks)
            self.catalog.publish(token, self._authorize)
        except Exception:
            self.catalog.fail(token)
            raise
        return {"status": "published", "document_id": document_id, "previous_revision": expected_revision,
                "revision_id": revision_id, "chunks_verified": len(chunks), "update_id": token,
                "updated_by": actor.username, "updated_by_user_id": str(self.actor_id),
                "note": "Catalog head published after upload verification; Pinecone search indexing may lag."}

    def _upload_and_verify(self, chunks):
        with create_pinecone_client() as client:
            with get_pinecone_index(client, self.settings) as index:
                mode = get_index_embedding_mode(client, self.settings)
                embeddings = create_embedding_client(self.settings) if mode == "external" else None
                ingest_documents(chunks, self.settings, embedding_client=embeddings, index=index,
                                 actor_id=self.actor_id, authorization=self.authorization, embedding_mode=mode)
                for offset in range(0, len(chunks), self.settings.batch_size):
                    batch = chunks[offset:offset + self.settings.batch_size]
                    response = index.documents.fetch(namespace=self.settings.namespace,
                        ids=[chunk.metadata["chunk_id"] for chunk in batch],
                        include_fields=["chunk_text", *chunks[0].metadata, "ingested_by_user_id", "ingested_by"], timeout=30)
                    for chunk in batch:
                        stored = response.documents.get(chunk.metadata["chunk_id"])
                        if stored is None:
                            raise RuntimeError("Uploaded revision is not yet fetch-visible; head was not published")
                        fields = stored.to_dict()
                        if (fields.get("chunk_text") != chunk.page_content
                            or any(fields.get(key) != value for key, value in chunk.metadata.items())
                            or fields.get("ingested_by_user_id") != str(self.actor_id)):
                            raise RuntimeError("Uploaded revision verification failed; head was not published")
