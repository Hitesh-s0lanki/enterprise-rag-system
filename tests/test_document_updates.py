"""RBAC-protected document updates with transactional heads and offline providers."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pinecone import Document as PineconeDocument

from src.data_ingestion.config import IngestionSettings
from src.data_ingestion.updates import DocumentUpdateService
from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database
from src.db.documents import RevisionConflict, SQLiteDocumentRepository, initialize_document_catalog
from src.tools import create_document_update_tools

DOC, OLD = "a" * 64, "b" * 64


class DocumentUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = initialize_database(Path(self.temp.name) / "auth.sqlite3")
        initialize_document_catalog(self.path)
        self.repo = SQLiteUserRepository(self.path)
        self.authorization = AuthorizationService(self.repo)
        self.catalog = SQLiteDocumentRepository(self.path)
        self.admin = self.repo.get_by_username("mice")
        self.bob = self.repo.get_by_username("bob")
        self.alice = self.repo.get_by_username("alice")
        self.settings = IngestionSettings(embedding_dimensions=3, chunk_size=100, chunk_overlap=10)
        self.metadata = {"document_id": DOC, "source": "handbook.md", "file_name": "handbook.md",
                         "file_type": ".md", "last_updated_by": "original-editor",
                         "last_updated_at": "2026-11-16T00:00:00Z"}
        self.service(self.admin).register_existing_document(document_id=DOC, revision_id=OLD,
            content="Original VPN instructions", metadata=self.metadata,
            reader_roles=["employee", "contributor"], writer_roles=["contributor"])
        self.update = self.service(self.bob)
        self.patches = {}
        for name in ("create_pinecone_client", "get_pinecone_index", "get_index_embedding_mode", "create_embedding_client"):
            patcher = patch(f"src.data_ingestion.updates.{name}")
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)
        self.index = self.patches["get_pinecone_index"].return_value.__enter__.return_value
        self.patches["get_index_embedding_mode"].return_value = "integrated"
        self.stored = {}
        def upsert(**kwargs):
            for row in kwargs["documents"]:
                self.stored[row["_id"]] = dict(row)
            return SimpleNamespace(upserted_count=len(kwargs["documents"]))
        self.index.documents.upsert.side_effect = upsert
        self.index.documents.fetch.side_effect = lambda **kwargs: SimpleNamespace(documents={
            key: PineconeDocument(self.stored[key]) for key in kwargs["ids"] if key in self.stored})

    def service(self, user):
        return DocumentUpdateService(actor_id=user.id, authorization=self.authorization,
                                     catalog=self.catalog, settings=self.settings)

    def rows(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return connection.execute("SELECT revision_id, state FROM knowledge_revisions ORDER BY rowid").fetchall()

    def test_allowed_update_verifies_upload_before_publishing_and_preserves_metadata(self):
        def fetch(**kwargs):
            self.assertEqual(self.catalog.approved_revisions("it-support"), {DOC: OLD})
            return SimpleNamespace(documents={key: PineconeDocument(self.stored[key]) for key in kwargs["ids"]})
        self.index.documents.fetch.side_effect = fetch
        output = self.update.update_document(DOC, OLD, "New VPN instructions. " * 20)
        self.assertEqual(output["status"], "published")
        self.assertNotEqual(output["revision_id"], OLD)
        current = self.catalog.get("it-support", DOC)
        self.assertEqual(current.revision_id, output["revision_id"])
        self.assertIsNone(current.pending_update_id)
        self.assertEqual(current.metadata["source"], "handbook.md")
        self.assertEqual(current.metadata["last_updated_by"], "bob")
        self.assertEqual(current.writer_roles, ("contributor",))
        for row in self.stored.values():
            self.assertEqual(row["document_id"], DOC)
            self.assertEqual(row["revision_id"], current.revision_id)
            self.assertEqual(row["ingested_by_user_id"], str(self.bob.id))
            self.assertEqual(row["last_updated_at_source"], "authenticated_edit")
            self.assertEqual(row["allowed_roles"], ["contributor", "employee"])
        self.patches["create_embedding_client"].assert_not_called()
        self.assertEqual([state for _, state in self.rows()], ["published", "published"])

    def test_denied_employee_inactive_or_out_of_scope_has_no_writes_or_provider_calls(self):
        for user in (self.alice, replace(self.bob, is_active=False), replace(self.bob, namespaces=("other",))):
            with self.subTest(user=user):
                self.repo.save_user(user)
                with self.assertRaises(PermissionError):
                    self.service(user).update_document(DOC, OLD, "Replacement")
                self.repo.save_user(self.bob)
        self.assertEqual(self.rows(), [(OLD, "published")])
        self.patches["create_pinecone_client"].assert_not_called()
        self.patches["create_embedding_client"].assert_not_called()

    def test_document_writer_acl_denies_even_when_role_has_write_permission(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE knowledge_documents SET writer_roles_json=?", (json.dumps(["administrator"]),))
        with self.assertRaises(PermissionError):
            self.update.update_document(DOC, OLD, "Replacement")
        self.assertEqual(self.rows(), [(OLD, "published")])
        self.patches["create_pinecone_client"].assert_not_called()

    def test_database_permission_mask_is_authoritative(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE roles SET permissions=1 WHERE name='contributor'")
        with self.assertRaises(PermissionError):
            self.update.update_document(DOC, OLD, "Replacement")
        self.patches["create_pinecone_client"].assert_not_called()

    def test_external_embeddings_are_used_for_allowed_updates_only(self):
        self.patches["get_index_embedding_mode"].return_value = "external"
        self.patches["create_embedding_client"].return_value.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        self.assertEqual(self.update.update_document(DOC, OLD, "Replacement")["status"], "published")
        self.patches["create_embedding_client"].assert_called_once_with(self.settings)
        self.assertTrue(all(row["embedding"] == [0.1, 0.2, 0.3] for row in self.stored.values()))

    def test_revoked_access_after_staging_prevents_all_provider_calls(self):
        original_stage = self.catalog.stage
        def revoke(*args, **kwargs):
            token = original_stage(*args, **kwargs)
            self.repo.save_user(replace(self.bob, is_active=False))
            return token
        with patch.object(self.catalog, "stage", side_effect=revoke):
            with self.assertRaises(PermissionError):
                self.update.update_document(DOC, OLD, "Replacement")
        self.assertEqual(self.catalog.approved_revisions("it-support"), {DOC: OLD})
        self.patches["create_pinecone_client"].assert_not_called()
        self.patches["create_embedding_client"].assert_not_called()

    def test_document_reader_acl_also_required_for_writers(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE knowledge_documents SET reader_roles_json=?", (json.dumps(["employee"]),))
        with self.assertRaises(PermissionError):
            self.update.update_document(DOC, OLD, "Replacement")
        self.patches["create_pinecone_client"].assert_not_called()

    def test_unchanged_content_checks_permissions_but_performs_no_writes(self):
        result = self.update.update_document(DOC, OLD, "Original VPN instructions")
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(self.rows(), [(OLD, "published")])
        self.patches["create_pinecone_client"].assert_not_called()

    def test_stale_revision_and_invalid_content_are_rejected_before_side_effects(self):
        with self.assertRaises(RevisionConflict):
            self.update.update_document(DOC, "c" * 64, "Replacement")
        for content in (" ", "x" * 100001):
            with self.assertRaises(ValueError):
                self.update.update_document(DOC, OLD, content)
        with self.assertRaises(PermissionError):
            self.update.update_document("d" * 64, OLD, "Replacement")
        self.assertEqual(self.rows(), [(OLD, "published")])
        self.patches["create_pinecone_client"].assert_not_called()

    def test_failed_partial_or_unverified_upload_keeps_old_head_and_releases_reservation(self):
        for mode in ("provider_error", "partial_ack", "not_visible", "corrupt_metadata"):
            with self.subTest(mode=mode):
                self.index.documents.upsert.side_effect = None
                self.index.documents.upsert.return_value = SimpleNamespace(upserted_count=0)
                if mode == "provider_error":
                    self.index.documents.upsert.side_effect = RuntimeError("provider unavailable")
                elif mode != "partial_ack":
                    def upsert(**kwargs):
                        for row in kwargs["documents"]:
                            self.stored[row["_id"]] = dict(row, chunk_text="corrupt")
                        return SimpleNamespace(upserted_count=len(kwargs["documents"]))
                    self.index.documents.upsert.side_effect = upsert
                if mode == "not_visible":
                    self.index.documents.fetch.side_effect = lambda **kwargs: SimpleNamespace(documents={})
                else:
                    self.index.documents.fetch.side_effect = lambda **kwargs: SimpleNamespace(documents={
                        key: PineconeDocument(self.stored[key]) for key in kwargs["ids"]})
                with self.assertRaises(RuntimeError):
                    self.update.update_document(DOC, OLD, "Replacement " + mode)
                self.assertEqual(self.catalog.approved_revisions("it-support"), {DOC: OLD})
                self.assertIsNone(self.catalog.get("it-support", DOC).pending_update_id)
        self.assertEqual([state for _, state in self.rows()], ["published"] + ["failed"] * 4)

    def test_revoked_access_after_upload_blocks_publication(self):
        original_fetch = self.index.documents.fetch.side_effect
        def revoke(**kwargs):
            result = original_fetch(**kwargs)
            self.repo.save_user(replace(self.bob, is_active=False))
            return result
        self.index.documents.fetch.side_effect = revoke
        with self.assertRaises(PermissionError):
            self.update.update_document(DOC, OLD, "Replacement")
        self.assertEqual(self.catalog.approved_revisions("it-support"), {DOC: OLD})
        self.assertEqual(self.rows()[-1][1], "failed")

    def test_changed_reader_acl_during_upload_cannot_publish_stale_provider_acl(self):
        original_fetch = self.index.documents.fetch.side_effect
        def change_acl(**kwargs):
            result = original_fetch(**kwargs)
            with closing(sqlite3.connect(self.path)) as connection, connection:
                connection.execute("UPDATE knowledge_documents SET reader_roles_json=?",
                                   (json.dumps(["administrator", "contributor", "employee"]),))
            return result
        self.index.documents.fetch.side_effect = change_acl
        with self.assertRaises(RevisionConflict):
            self.update.update_document(DOC, OLD, "Replacement")
        self.assertEqual(self.catalog.approved_revisions("it-support"), {DOC: OLD})
        self.assertEqual(self.rows()[-1][1], "failed")

    def test_concurrent_update_cannot_overwrite_reserved_head(self):
        def uploading(chunks):
            with self.assertRaises(RevisionConflict):
                self.service(self.admin).update_document(DOC, OLD, "Concurrent replacement")
        with patch.object(self.update, "_upload_and_verify", side_effect=uploading):
            self.assertEqual(self.update.update_document(DOC, OLD, "First replacement")["status"], "published")

    def test_registration_requires_management_and_never_resets_existing_catalog(self):
        args = dict(document_id=DOC, revision_id=OLD, content="Original VPN instructions", metadata=self.metadata,
                    reader_roles=["employee", "contributor"], writer_roles=["contributor"])
        with self.assertRaises(PermissionError):
            self.update.register_existing_document(**args)
        with self.assertRaises(RevisionConflict):
            self.service(self.admin).register_existing_document(**args)
        initialize_document_catalog(self.path)
        self.assertEqual(self.rows(), [(OLD, "published")])

    def test_tool_schemas_and_argument_injection(self):
        tools = {tool.name: tool for tool in create_document_update_tools(self.update)}
        current = tools["get_document_for_update"].invoke({"document_id": DOC})
        self.assertEqual(current["revision_id"], OLD)
        for extra in ({"actor_id": str(self.admin.id)}, {"allowed_roles": ["contributor"]},
                      {"namespace": "other"}, {"writer_roles": ["contributor"]}):
            result = tools["update_document"].invoke({"document_id": DOC, "expected_revision": OLD,
                                                     "content": "Replacement", **extra})
            self.assertEqual(json.loads(result)["error"]["code"], "invalid_input")
        self.patches["create_pinecone_client"].assert_not_called()
        response = tools["update_document"].invoke({"document_id": DOC, "expected_revision": OLD, "content": "Replacement"})
        self.assertEqual(response["status"], "published")

    def test_tool_denial_and_conflicts_are_error_messages(self):
        tool = create_document_update_tools(self.service(self.alice))[1]
        response = tool.invoke({"type": "tool_call", "name": tool.name, "id": "call-1", "args": {
            "document_id": DOC, "expected_revision": OLD, "content": "Replacement"}})
        self.assertEqual(response.status, "error")
        self.assertEqual(json.loads(response.content)["error"]["code"], "access_denied")

    def test_root_agent_defaults_include_real_edit_tools_and_refresh_published_heads(self):
        from langchain_core.messages import AIMessage
        from src.agents import AgentConfig, RootAgent
        from src.query.pipeline import KnowledgeQueryService
        from test_root_agent import ScriptedModel

        knowledge = KnowledgeQueryService(actor_id=self.bob.id, authorization=self.authorization,
                                           knowledge=self.settings, approved_revisions={DOC: OLD})
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "get_document_for_update", "id": "read-1",
                                               "args": {"document_id": DOC}}]),
            AIMessage(content="", tool_calls=[{"name": "update_document", "id": "edit-1", "args": {
                "document_id": DOC, "expected_revision": OLD, "content": "Root agent replacement"}}]),
            AIMessage(content="The document update was published."),
        ])
        agent = RootAgent(knowledge, AgentConfig(model="scripted"), llm=model, include_firecrawl=False)
        self.assertIn("update_document", {tool.name for tool in agent.get_tools()})
        events = list(agent.stream("Update the handbook to use these new VPN instructions."))
        self.assertEqual(events[-1].type, "run_completed")
        self.assertEqual(sum(event.type == "tool_completed" for event in events), 2)
        current = self.catalog.get("it-support", DOC)
        self.assertEqual(current.content, "Root agent replacement")
        _, revisions = knowledge._preflight("VPN?", graded=True)
        self.assertEqual(revisions, {DOC: current.revision_id})
        self.assertNotEqual(revisions, {DOC: OLD})

    def test_root_agent_readonly_user_edit_is_denied_in_stream_without_provider_calls(self):
        from langchain_core.messages import AIMessage
        from src.agents import AgentConfig, RootAgent
        from src.query.pipeline import KnowledgeQueryService
        from test_root_agent import ScriptedModel

        knowledge = KnowledgeQueryService(actor_id=self.alice.id, authorization=self.authorization,
                                           knowledge=self.settings, approved_revisions={DOC: OLD})
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "update_document", "id": "edit-denied", "args": {
                "document_id": DOC, "expected_revision": OLD, "content": "Unauthorized replacement"}}]),
            AIMessage(content="Document write access is required."),
        ])
        agent = RootAgent(knowledge, AgentConfig(model="scripted"), llm=model, include_firecrawl=False)
        events = list(agent.stream("Update this document."))
        errors = [event for event in events if event.type == "tool_error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("access_denied", json.dumps(errors[0].data))
        self.assertEqual(events[-1].type, "run_completed")
        self.assertEqual(self.catalog.get("it-support", DOC).revision_id, OLD)
        self.patches["create_pinecone_client"].assert_not_called()
        self.patches["create_embedding_client"].assert_not_called()

    def test_root_update_registry_can_disable_edits_and_rejects_cross_user_service(self):
        from src.query.pipeline import KnowledgeQueryService
        from src.tools import create_root_tools
        knowledge = KnowledgeQueryService(actor_id=self.bob.id, authorization=self.authorization,
                                           knowledge=self.settings, approved_revisions={DOC: OLD})
        tools = create_root_tools(knowledge, include_firecrawl=False, include_document_updates=False)
        self.assertNotIn("update_document", {tool.name for tool in tools})
        with self.assertRaises(ValueError):
            create_root_tools(knowledge, include_firecrawl=False, document_update_service=self.service(self.admin))


if __name__ == "__main__":
    unittest.main()
