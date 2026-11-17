"""Offline coverage of deployment recovery and fail-closed provider verification."""

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from src.app.services.runtime import AppRuntime
from src.data_ingestion.bootstrap import DEFAULT_MANIFEST, CatalogBootstrapError
from src.data_ingestion.config import IngestionSettings
from src.data_ingestion.documents import split_document
from src.db.auth import SQLiteUserRepository, initialize_database


class CatalogBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "auth.sqlite3"
        self.environment = patch.dict(os.environ, {
            "AUTH_DB_PATH": str(self.path), "CATALOG_BOOTSTRAP_ENABLED": "true",
            "CATALOG_BOOTSTRAP_PATH": str(DEFAULT_MANIFEST),
            "PINECONE_NAMESPACE": "it-support", "PINECONE_INDEX_NAME": "enterprise-it-knowledge",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.directory.cleanup)
        self.manifest = json.loads(DEFAULT_MANIFEST.read_text())
        self.settings = IngestionSettings(**self.manifest["settings"])
        self.stored = {}
        for entry in self.manifest["documents"]:
            for chunk in split_document(Document(page_content=entry["content"], metadata=entry["metadata"]), self.settings):
                self.stored[chunk.metadata["chunk_id"]] = {
                    **chunk.metadata, "chunk_text": chunk.page_content,
                    "ingested_by": "bob", "ingested_by_user_id": "00000000-0000-0000-0000-000000000002",
                }
        self.client_patch = patch("src.data_ingestion.bootstrap.create_pinecone_client")
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.index_patch = patch("src.data_ingestion.bootstrap.get_pinecone_index")
        self.index_factory = self.index_patch.start()
        self.addCleanup(self.index_patch.stop)
        self.index = self.index_factory.return_value.__enter__.return_value
        self.index.documents.fetch.side_effect = lambda **kwargs: SimpleNamespace(documents={
            identifier: SimpleNamespace(to_dict=lambda fields=self.stored[identifier]: fields)
            for identifier in kwargs["ids"] if identifier in self.stored})
        self.embedding_patch = patch("src.data_ingestion.embeddings.create_embedding_client")
        self.embedding = self.embedding_patch.start()
        self.addCleanup(self.embedding_patch.stop)

    def assert_no_uploads(self):
        self.index.documents.upsert.assert_not_called()
        self.embedding.assert_not_called()

    def test_fresh_startup_restores_both_documents_and_provenance_once(self):
        runtime = AppRuntime.from_env()
        expected = {entry["document_id"]: entry["revision_id"] for entry in self.manifest["documents"]}
        self.assertEqual(runtime.revisions(), expected)
        for did in expected:
            record = runtime.catalog.get(self.settings.namespace, did)
            self.assertEqual(record.metadata["ingested_by"], "bob")
        self.assertEqual(self.index.documents.fetch.call_count, 2)
        self.client.reset_mock()
        self.assertEqual(AppRuntime.from_env().revisions(), expected)
        self.client.assert_not_called()
        self.assert_no_uploads()

    def test_existing_updated_revision_and_acl_are_preserved_while_missing_document_is_restored(self):
        runtime = AppRuntime.from_env()
        first, second = self.manifest["documents"]
        original = runtime.catalog.get(self.settings.namespace, first["document_id"])
        changed = replace(original, revision_id="f" * 64, content="Edited source", reader_roles=("administrator",))
        # Model a current published head with changed ACLs through a trusted maintenance transaction.
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE knowledge_documents SET current_revision=?, reader_roles_json=? WHERE document_id=?",
                               (changed.revision_id, '["administrator"]', original.document_id))
            connection.execute("UPDATE knowledge_revisions SET revision_id=?, content=? WHERE document_id=?",
                               (changed.revision_id, changed.content, original.document_id))
            connection.execute("DELETE FROM knowledge_revisions WHERE document_id=?", (second["document_id"],))
            connection.execute("DELETE FROM knowledge_documents WHERE document_id=?", (second["document_id"],))
        self.index.documents.fetch.reset_mock()
        restarted = AppRuntime.from_env()
        record = restarted.catalog.get(self.settings.namespace, first["document_id"])
        self.assertEqual(record.revision_id, changed.revision_id)
        self.assertEqual(record.reader_roles, ("administrator",))
        self.assertEqual(record.content, changed.content)
        self.assertEqual(self.index.documents.fetch.call_count, 1)
        self.assertEqual(len(restarted.revisions()), 2)
        self.assert_no_uploads()

    def test_mismatched_text_roles_or_revision_never_publish(self):
        original = dict(self.stored)
        identifier = next(iter(self.stored))
        for key, value in (("chunk_text", "tampered"), ("allowed_roles", ["employee"]),
                           ("revision_id", "0" * 64), ("ingested_by_user_id", "")):
            with self.subTest(key=key):
                self.stored[identifier] = {**original[identifier], key: value}
                with self.assertRaises(CatalogBootstrapError):
                    AppRuntime.from_env()
                with closing(sqlite3.connect(self.path)) as connection:
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0], 0)
        self.assert_no_uploads()

    def test_missing_chunk_in_second_document_publishes_neither_document(self):
        self.stored.pop(next(reversed(self.stored)))
        with self.assertRaises(CatalogBootstrapError):
            AppRuntime.from_env()
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0], 0)
        self.assert_no_uploads()

    def test_denied_maintainer_never_calls_providers(self):
        initialize_database(self.path)
        users = SQLiteUserRepository(self.path)
        admin = users.get_by_username("mice")
        assert admin is not None
        for changes in ({"is_active": False}, {"namespaces": ()}):
            with self.subTest(changes=changes):
                users.save_user(replace(admin, **changes))
                with self.assertRaises(CatalogBootstrapError):
                    AppRuntime.from_env()
                self.client.assert_not_called()
        users.save_user(admin)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE roles SET permissions=1 WHERE name='administrator'")
        with self.assertRaises(CatalogBootstrapError):
            AppRuntime.from_env()
        self.client.assert_not_called()
        self.assert_no_uploads()

    def test_other_namespace_or_index_is_never_approved_by_default_manifest(self):
        for settings in ({"PINECONE_NAMESPACE": "other"}, {"PINECONE_INDEX_NAME": "other"}):
            with patch.dict(os.environ, settings):
                self.assertEqual(AppRuntime.from_env().revisions(), {})
        self.client.assert_not_called()

    def test_provider_failure_is_sanitized_in_ui_and_retry_recovers(self):
        self.index.documents.fetch.side_effect = RuntimeError("private-provider-credential")
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/app/main.py")).run()
        self.assertFalse(app.exception)
        self.assertIn("Knowledge startup verification failed", app.error[0].value)
        self.assertNotIn("private-provider-credential", app.error[0].value)
        self.assertEqual(len(app.chat_input), 0)
        self.index.documents.fetch.side_effect = lambda **kwargs: SimpleNamespace(documents={
            identifier: SimpleNamespace(to_dict=lambda fields=self.stored[identifier]: fields)
            for identifier in kwargs["ids"]})
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(app.chat_input[0].disabled)
        self.assert_no_uploads()

    def test_concurrent_registration_preserves_the_winning_revision(self):
        from src.db.documents.repository import SQLiteDocumentRepository
        original = SQLiteDocumentRepository.register

        def race(repository, record, actor_id, authorize):
            original(repository, record, actor_id, authorize)
            original(repository, record, actor_id, authorize)

        with patch.object(SQLiteDocumentRepository, "register", race):
            runtime = AppRuntime.from_env()
        self.assertEqual(len(runtime.revisions()), 2)


if __name__ == "__main__":
    unittest.main()
