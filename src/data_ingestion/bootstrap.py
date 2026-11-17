"""Restore explicitly approved demo revisions after ephemeral SQLite storage loss."""

import json
import os
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document

from src.clients.pinecone_client import create_pinecone_client, get_pinecone_index
from src.db.auth import Permission
from src.db.documents import RevisionConflict

from .config import IngestionSettings
from .documents import split_document
from .updates import DocumentUpdateService


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "db/documents/bootstrap.json"


class CatalogBootstrapError(RuntimeError):
    """Sanitized startup failure; never includes provider credentials or responses."""


def bootstrap_catalog(catalog, authorization, settings):
    """Verify missing manifest revisions before registration; never upload or reset heads.

    The checked-in manifest is maintainer approval, not provider-discovered ACLs.
    Authorization is checked before network access and again during registration.
    """
    enabled = os.getenv("CATALOG_BOOTSTRAP_ENABLED", "true").lower()
    if enabled not in {"true", "false"}:
        raise ValueError("CATALOG_BOOTSTRAP_ENABLED must be true or false")
    if enabled == "false":
        return
    try:
        manifest = json.loads(Path(os.getenv("CATALOG_BOOTSTRAP_PATH") or DEFAULT_MANIFEST).read_text())
        if manifest["version"] != 1:
            raise ValueError("Unsupported manifest version")
        approved_settings = IngestionSettings(**manifest["settings"])
        # Approval is bound to an index and namespace, never to arbitrary remote data.
        if (settings.index_name, settings.namespace) != (approved_settings.index_name, approved_settings.namespace):
            return
        missing = [entry for entry in manifest["documents"]
                   if catalog.get(settings.namespace, entry["document_id"]) is None]
        if not missing:
            return
        actor_id = UUID(manifest["actor_id"])
        authorization.require_access(actor_id, Permission.MANAGE_KNOWLEDGE | Permission.WRITE_KNOWLEDGE,
                                     namespace=settings.namespace)
        service = DocumentUpdateService(actor_id=actor_id, authorization=authorization,
                                        catalog=catalog, settings=approved_settings)
        prepared = []
        for entry in missing:
            chunks = split_document(Document(page_content=entry["content"], metadata=entry["metadata"]),
                                    approved_settings)
            if chunks[0].metadata["revision_id"] != entry["revision_id"]:
                raise ValueError("Manifest source does not reproduce the approved revision")
            if sorted(entry["reader_roles"]) != entry["metadata"]["allowed_roles"]:
                raise ValueError("Manifest reader ACL differs from the approved source")
            prepared.append((entry, chunks))
        # Verify every missing document before publishing any of them.
        verified = []
        with create_pinecone_client() as client:
            with get_pinecone_index(client, approved_settings) as index:
                for entry, chunks in prepared:
                    upload_metadata = None
                    for offset in range(0, len(chunks), approved_settings.batch_size):
                        batch = chunks[offset:offset + approved_settings.batch_size]
                        fields = sorted({key for chunk in batch for key in chunk.metadata}
                                        | {"chunk_text", "ingested_by", "ingested_by_user_id"})
                        response = index.documents.fetch(namespace=settings.namespace,
                            ids=[chunk.metadata["chunk_id"] for chunk in batch],
                            include_fields=fields, timeout=30)
                        for chunk in batch:
                            stored = response.documents.get(chunk.metadata["chunk_id"])
                            if stored is None:
                                raise ValueError("Approved chunk is missing from Pinecone")
                            data = stored.to_dict()
                            expected = {k: v for k, v in chunk.metadata.items() if k != "ingested_at"}
                            if data.get("chunk_text") != chunk.page_content or any(data.get(k) != v for k, v in expected.items()):
                                raise ValueError("Approved chunk content or metadata differs")
                            uploader = {k: data.get(k) for k in ("ingested_by", "ingested_by_user_id", "ingested_at")}
                            if any(not isinstance(v, str) or not v.strip() for v in uploader.values()):
                                raise ValueError("Approved chunk has no uploader provenance")
                            UUID(uploader["ingested_by_user_id"])
                            if upload_metadata is not None and upload_metadata != uploader:
                                raise ValueError("Chunk uploader provenance is inconsistent")
                            upload_metadata = uploader
                    if upload_metadata is None:
                        raise ValueError("Approved document has no chunks")
                    verified.append((entry, {**entry["metadata"], **upload_metadata}))
        for entry, metadata in verified:
            try:
                service.register_existing_document(document_id=entry["document_id"],
                    revision_id=entry["revision_id"], content=entry["content"], metadata=metadata,
                    reader_roles=entry["reader_roles"], writer_roles=entry["writer_roles"])
            except RevisionConflict:
                # Another startup may have enrolled or updated it while we fetched.
                if catalog.get(settings.namespace, entry["document_id"]) is None:
                    raise
    except Exception:
        raise CatalogBootstrapError(
            "Knowledge startup verification failed. Check Pinecone credentials, index/namespace, "
            "the approved bootstrap manifest, and its maintainer's database permissions. "
            "Every approved chunk must already exist and match; startup does not upload documents."
        ) from None
