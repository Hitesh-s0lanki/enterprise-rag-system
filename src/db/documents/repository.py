"""Transactional revision storage in the local auth DB, behind an adapter interface."""

import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from src.db.auth.database import connect_database, database_path


class RevisionConflict(RuntimeError):
    """The document changed or another replacement is already being uploaded."""


@dataclass(frozen=True)
class DocumentRecord:
    namespace: str
    document_id: str
    revision_id: str
    content: str
    metadata: dict
    reader_roles: tuple[str, ...]
    writer_roles: tuple[str, ...]
    pending_update_id: str | None


class DocumentRepository(Protocol):
    def get(self, namespace: str, document_id: str) -> DocumentRecord | None: ...
    def register(self, record: DocumentRecord, actor_id: str, authorize: Callable) -> None: ...
    def stage(self, record: DocumentRecord, expected_revision: str, actor_id: str, authorize: Callable) -> str: ...
    def publish(self, update_id: str, authorize: Callable) -> None: ...
    def fail(self, update_id: str) -> None: ...
    def approved_revisions(self, namespace: str) -> dict[str, str]: ...


def initialize_document_catalog(path: str | Path | None = None) -> Path:
    """Explicit additive setup; the auth DB must already exist. Never reseeds it."""
    target = database_path(path)
    with closing(connect_database(target)) as connection:
        connection.executescript((Path(__file__).parent / "schema.sql").read_text())
    return target


class SQLiteDocumentRepository:
    def __init__(self, path: str | Path | None = None):
        self.path = database_path(path)

    @staticmethod
    def _get(connection, namespace, document_id):
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_documents'").fetchone():
            return None
        row = connection.execute(
            """SELECT d.*, r.content, r.metadata_json FROM knowledge_documents d
               JOIN knowledge_revisions r ON r.namespace=d.namespace AND r.document_id=d.document_id
                 AND r.revision_id=d.current_revision AND r.state='published'
               WHERE d.namespace=? AND d.document_id=?""", (namespace, document_id),
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(row["namespace"], row["document_id"], row["current_revision"], row["content"],
                              json.loads(row["metadata_json"]), tuple(json.loads(row["reader_roles_json"])),
                              tuple(json.loads(row["writer_roles_json"])), row["pending_update_id"])

    def get(self, namespace, document_id):
        with closing(connect_database(self.path)) as connection:
            return self._get(connection, namespace, document_id)

    @staticmethod
    def _insert_revision(connection, record, update_id, actor_id, state, parent):
        connection.execute(
            """INSERT INTO knowledge_revisions
               (update_id, namespace, document_id, revision_id, parent_revision, content, metadata_json, actor_id, state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (update_id, record.namespace, record.document_id, record.revision_id, parent, record.content,
             json.dumps(record.metadata, sort_keys=True), actor_id, state),
        )

    def register(self, record, actor_id, authorize):
        with closing(connect_database(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            authorize(record)
            if self._get(connection, record.namespace, record.document_id) is not None:
                raise RevisionConflict("Document is already registered; registration never resets it")
            connection.execute(
                """INSERT INTO knowledge_documents
                   (namespace, document_id, current_revision, reader_roles_json, writer_roles_json) VALUES (?, ?, ?, ?, ?)""",
                (record.namespace, record.document_id, record.revision_id,
                 json.dumps(record.reader_roles), json.dumps(record.writer_roles)),
            )
            self._insert_revision(connection, record, uuid4().hex, actor_id, "published", None)

    def stage(self, record, expected_revision, actor_id, authorize):
        with closing(connect_database(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get(connection, record.namespace, record.document_id)
            if current is None:
                raise PermissionError("Document is unavailable")
            authorize(current)
            if current.revision_id != expected_revision or current.pending_update_id:
                raise RevisionConflict("Document changed or an update is in progress; read its current revision again")
            if (record.reader_roles, record.writer_roles) != (current.reader_roles, current.writer_roles):
                raise RevisionConflict("Document ACL changed during preparation")
            token = uuid4().hex
            self._insert_revision(connection, record, token, actor_id, "staged", expected_revision)
            connection.execute("UPDATE knowledge_documents SET pending_update_id=? WHERE namespace=? AND document_id=?",
                               (token, record.namespace, record.document_id))
            return token

    def publish(self, update_id, authorize):
        with closing(connect_database(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            revision = connection.execute("SELECT * FROM knowledge_revisions WHERE update_id=?", (update_id,)).fetchone()
            if revision is None or revision["state"] != "staged":
                raise RevisionConflict("Update is not staged")
            current = self._get(connection, revision["namespace"], revision["document_id"])
            if current is None:
                raise PermissionError("Document is unavailable")
            authorize(current)
            if current.pending_update_id != update_id or current.revision_id != revision["parent_revision"]:
                raise RevisionConflict("Document head changed before publication")
            staged_roles = json.loads(revision["metadata_json"]).get("allowed_roles")
            if staged_roles != list(current.reader_roles):
                raise RevisionConflict("Document reader ACL changed during upload")
            connection.execute("UPDATE knowledge_revisions SET state='published', published_at=CURRENT_TIMESTAMP WHERE update_id=?",
                               (update_id,))
            connection.execute(
                "UPDATE knowledge_documents SET current_revision=?, pending_update_id=NULL WHERE namespace=? AND document_id=?",
                (revision["revision_id"], revision["namespace"], revision["document_id"]),
            )

    def fail(self, update_id):
        with closing(connect_database(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE knowledge_revisions SET state='failed' WHERE update_id=? AND state='staged'", (update_id,))
            connection.execute("UPDATE knowledge_documents SET pending_update_id=NULL WHERE pending_update_id=?", (update_id,))

    def approved_revisions(self, namespace):
        with closing(connect_database(self.path)) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_documents'").fetchone():
                return {}
            rows = connection.execute("SELECT document_id, current_revision FROM knowledge_documents WHERE namespace=?",
                                      (namespace,)).fetchall()
        return {row["document_id"]: row["current_revision"] for row in rows}
