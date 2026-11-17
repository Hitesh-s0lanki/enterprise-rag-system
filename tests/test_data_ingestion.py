"""Offline checks for metadata, update identity, splitting and upload failures."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from pinecone import StringField
from pinecone.models.indexes.schema import FieldEmbedConfig, FullTextSearchConfig

from src.data_ingestion import IngestionSettings, prepare_documents, ingest_documents
from src.data_ingestion.documents import discover_text_files
from src.clients.pinecone_client import build_hybrid_schema, get_pinecone_index, validate_embedding_mode
from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "handbook.md"
        self.path.write_text("VPN policy\n\n" + "Connect using MFA. " * 40, encoding="utf-8")
        self.settings = IngestionSettings(chunk_size=100, chunk_overlap=15, embedding_dimensions=3, batch_size=2)
        self.updated = datetime(2026, 11, 16, tzinfo=timezone.utc)
        self.repository = SQLiteUserRepository(initialize_database(self.root / "auth.sqlite3"))
        self.authorization = AuthorizationService(self.repository)
        self.bob = self.repository.get_by_username("bob")

    def prepare(self, **kwargs):
        return prepare_documents(
            self.root, self.settings, updated_by="editor@example.com",
            allowed_roles=["employee", "contributor", "administrator"], updated_at=self.updated, **kwargs,
        )

    def ingest(self, chunks, embedder, index):
        return ingest_documents(chunks, self.settings, embedding_client=embedder, index=index,
                                actor_id=self.bob.id, authorization=self.authorization)

    def test_chunk_bounds_and_source_metadata(self):
        chunks = self.prepare()
        self.assertGreater(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk.page_content), 100)
            meta = chunk.metadata
            self.assertEqual(meta["file_name"], "handbook.md")
            self.assertEqual(meta["last_updated_by"], "editor@example.com")
            self.assertEqual(meta["last_updated_at"], "2026-11-16T00:00:00Z")
            self.assertEqual(meta["last_updated_at_source"], "provided")
            self.assertEqual(meta["chunk_index"], i)
            self.assertEqual(meta["chunk_count"], len(chunks))
            start = meta["start_index"]
            self.assertEqual(self.path.read_text()[start:start + len(chunk.page_content)], chunk.page_content)

    def test_repeatable_ids_and_changed_revision(self):
        before = self.prepare()
        again = self.prepare()
        self.assertEqual([c.metadata["chunk_id"] for c in before], [c.metadata["chunk_id"] for c in again])
        self.path.write_text("Updated VPN policy.", encoding="utf-8")
        after = self.prepare()
        self.assertEqual(before[0].metadata["document_id"], after[0].metadata["document_id"])
        self.assertNotEqual(before[0].metadata["revision_id"], after[0].metadata["revision_id"])

    def test_same_filename_in_different_folders_is_distinct(self):
        nested = self.root / "support"
        nested.mkdir()
        (nested / self.path.name).write_text("Support-only instructions.")
        self.assertEqual(len({c.metadata["document_id"] for c in self.prepare()}), 2)

    def test_access_change_creates_new_revision(self):
        before = self.prepare()[0]
        after = prepare_documents(self.root, self.settings, updated_by="editor@example.com",
                                  allowed_roles=["administrator"], updated_at=self.updated)[0]
        self.assertNotEqual(before.metadata["revision_id"], after.metadata["revision_id"])

    def test_invalid_source_and_provenance_rejected(self):
        with self.assertRaises(ValueError):
            prepare_documents(self.root, self.settings, updated_by="", allowed_roles=["employee"])
        with self.assertRaises(ValueError):
            prepare_documents(self.root, self.settings, updated_by="editor", allowed_roles=[])
        with self.assertRaises(ValueError):
            prepare_documents(self.root, self.settings, updated_by="editor", allowed_roles=["employee"],
                              updated_at=datetime(2026, 11, 16))
        unsupported = self.root / "scan.pdf"
        unsupported.write_bytes(b"not text")
        with self.assertRaises(ValueError):
            discover_text_files(unsupported)
        self.path.write_text("   ")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_upload_preserves_text_metadata_and_batches(self):
        chunks = self.prepare()
        embedder = Mock()
        embedder.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        index = Mock()
        index.documents.upsert.side_effect = lambda **kw: SimpleNamespace(upserted_count=len(kw["documents"]))
        report = self.ingest(chunks, embedder, index)
        self.assertEqual(report["chunks_accepted"], len(chunks))
        records = []
        for call in index.documents.upsert.call_args_list:
            self.assertEqual(call.kwargs["namespace"], "it-support")
            self.assertLessEqual(len(call.kwargs["documents"]), 2)
            records.extend(call.kwargs["documents"])
        for record, chunk in zip(records, chunks, strict=True):
            self.assertEqual(record["_id"], chunk.metadata["chunk_id"])
            self.assertEqual(record["chunk_text"], chunk.page_content)
            self.assertEqual(record["allowed_roles"], chunk.metadata["allowed_roles"])
            self.assertEqual(record["last_updated_by"], "editor@example.com")
            self.assertEqual(record["ingested_by"], "bob")
            self.assertEqual(record["ingested_by_user_id"], str(self.bob.id))

    def test_bad_vectors_do_not_upload(self):
        index = Mock()
        embedder = Mock()
        embedder.embed_documents.side_effect = lambda texts: [[float("nan")]] * len(texts)
        with self.assertRaises(ValueError):
            self.ingest(self.prepare(), embedder, index)
        index.documents.upsert.assert_not_called()

    def test_failed_acknowledgement_does_not_report_success(self):
        index = Mock()
        index.documents.upsert.return_value = SimpleNamespace(upserted_count=0)
        embedder = Mock()
        embedder.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        with self.assertRaises(RuntimeError):
            self.ingest(self.prepare(), embedder, index)

    def test_employee_cannot_ingest_or_spoof_editor_to_gain_access(self):
        index, embedder = Mock(), Mock()
        chunks = self.prepare()
        chunks[0].metadata["last_updated_by"] = "mice"
        with self.assertRaises(PermissionError):
            ingest_documents(chunks, self.settings, embedding_client=embedder, index=index,
                             actor_id=self.repository.get_by_username("alice").id,
                             authorization=self.authorization)
        embedder.embed_documents.assert_not_called()
        index.documents.upsert.assert_not_called()

    def test_all_document_scopes_checked_before_any_upload(self):
        index, embedder = Mock(), Mock()
        chunks = self.prepare()
        chunks[-1].metadata["allowed_roles"] = ["administrator"]
        with self.assertRaises(PermissionError):
            self.ingest(chunks, embedder, index)
        embedder.embed_documents.assert_not_called()
        index.documents.upsert.assert_not_called()

    def test_hybrid_schema_and_incompatible_index(self):
        fields = build_hybrid_schema(self.settings)["fields"]
        self.assertEqual(fields["embedding"]["dimension"], 3)
        self.assertTrue(fields["chunk_text"]["full_text_search"])
        client = Mock()
        client.indexes.exists.return_value = True
        client.indexes.describe.return_value = SimpleNamespace(schema=SimpleNamespace(fields={}))
        with self.assertRaises(ValueError):
            get_pinecone_index(client, self.settings)
        client.index.assert_not_called()

    def test_integrated_openai_schema_requires_matching_model_and_dimensions(self):
        text = StringField(full_text_search=FullTextSearchConfig(language="en"),
                           embed=FieldEmbedConfig(model="text-embedding-3-small", dimension=3, metric="cosine"))
        info = SimpleNamespace(schema=SimpleNamespace(fields={"chunk_text": text}))
        self.assertEqual(validate_embedding_mode(info, self.settings), "integrated")
        text.embed = FieldEmbedConfig(model="llama-text-embed-v2", dimension=3, metric="cosine")
        with self.assertRaises(ValueError):
            validate_embedding_mode(info, self.settings)
        text.embed = FieldEmbedConfig(model="text-embedding-3-small", dimension=1024, metric="cosine")
        with self.assertRaises(ValueError):
            validate_embedding_mode(info, self.settings)

    def test_integrated_ingestion_sends_text_without_duplicate_embedding_calls(self):
        chunks, embedder, index = self.prepare(), Mock(), Mock()
        index.documents.upsert.side_effect = lambda **kw: SimpleNamespace(upserted_count=len(kw["documents"]))
        report = ingest_documents(chunks, self.settings, embedding_client=embedder, index=index,
                                  actor_id=self.bob.id, authorization=self.authorization, embedding_mode="integrated")
        self.assertEqual(report["embedding_mode"], "integrated")
        embedder.embed_documents.assert_not_called()
        for call in index.documents.upsert.call_args_list:
            for record in call.kwargs["documents"]:
                self.assertNotIn("embedding", record)
                self.assertTrue(record["chunk_text"])
                self.assertEqual(record["ingested_by"], "bob")


if __name__ == "__main__":
    unittest.main()
