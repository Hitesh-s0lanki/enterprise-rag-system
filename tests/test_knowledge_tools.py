"""Agent tool dispatch and lazy authorized provider orchestration, fully offline."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.clients.jev_client import JevSettings
from src.data_ingestion.config import IngestionSettings
from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database
from src.grading import GradingSettings
from src.query.config import QueryCacheSettings
from src.query.pipeline import KnowledgeQueryService
from src.retrieval import RetrievalResult, RetrievalSettings
from src.tools import create_knowledge_tools

DOC, REV = "a" * 64, "b" * 64
QUERY = "How do I connect to the VPN?"


class FakeMemory:
    def __init__(self):
        self.entries = {}

    def search(self, query, **kwargs):
        return {"results": [dict(value, id=key, score=0.6) for key, value in self.entries.items()
                            if value["metadata"]["cache_kind"] == "query"]}

    def add(self, text, **kwargs):
        key = f"memory-{len(self.entries)}"
        self.entries[key] = {"memory": text, "metadata": kwargs["metadata"]}
        return {"status": "SUCCEEDED", "results": [{"id": key}]}

    def get(self, key):
        return self.entries[key]


class KnowledgeToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = SQLiteUserRepository(initialize_database(Path(self.temp.name) / "auth.sqlite3"))
        self.authorization = AuthorizationService(self.repo)
        self.user = self.repo.get_by_username("alice")
        self.revisions = {DOC: REV}
        self.service = KnowledgeQueryService(
            actor_id=self.user.id, authorization=self.authorization, approved_revisions=lambda: self.revisions,
            knowledge=IngestionSettings(), retrieval=RetrievalSettings(), grading=GradingSettings(),
            jev=JevSettings(model="test-jev"), cache=QueryCacheSettings(),
        )
        self.tools = {tool.name: tool for tool in create_knowledge_tools(self.service, include_advanced=True)}
        self.patches = {}
        for name in ("create_pinecone_client", "get_pinecone_index", "get_index_embedding_mode",
                     "create_embedding_client", "HybridRetrievalService", "create_jev_client",
                     "DocumentGradingService", "create_mem0_client", "query_equivalence"):
            patcher = patch(f"src.query.pipeline.{name}")
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)
        self.patches["get_index_embedding_mode"].return_value = "integrated"
        row = Document(id="chunk-1", page_content="Connect with MFA.", metadata={
            "document_id": DOC, "revision_id": REV, "chunk_id": "chunk-1",
            "allowed_roles": ["employee"], "source": "handbook.md",
        })
        self.result = RetrievalResult(documents=[row], metrics={"search_calls": 2})
        self.patches["HybridRetrievalService"].return_value.retrieve.return_value = self.result
        self.grade = {"query": QUERY, "documents": self.result.to_dict()["documents"],
                      "summary": {"status": "sufficient_evidence"}, "metrics": {"jev_calls": 1}}
        self.patches["DocumentGradingService"].return_value.grade.return_value.to_dict.return_value = self.grade
        self.memory = FakeMemory()
        self.patches["create_mem0_client"].return_value = self.memory
        self.patches["query_equivalence"].return_value = 0.99

    def test_default_registry_and_provider_schema_expose_query_and_memory_control(self):
        self.assertEqual([tool.name for tool in create_knowledge_tools(self.service)], ["search_knowledge"])
        self.assertEqual(set(self.tools), {"search_knowledge", "retrieve_documents", "retrieve_and_grade_documents"})
        for tool in self.tools.values():
            schema = convert_to_openai_tool(tool)["function"]["parameters"]
            expected = {"query", "use_memory"} if tool.name == "search_knowledge" else {"query"}
            self.assertEqual(set(schema["properties"]), expected)
            self.assertEqual(schema["required"], ["query"])
            if tool.name == "search_knowledge":
                self.assertEqual(schema["properties"]["use_memory"]["type"], "boolean")
                self.assertIs(schema["properties"]["use_memory"]["default"], True)
            self.assertFalse(schema["additionalProperties"])
        for name in ("create_mem0_client", "create_pinecone_client", "create_jev_client"):
            self.patches[name].assert_not_called()

    def test_full_flow_miss_then_exact_hit_skips_pipeline_clients(self):
        tool = self.tools["search_knowledge"]
        first = tool.invoke({"query": QUERY})
        self.assertEqual(first["source"], "rag_pipeline")
        self.assertEqual(first["cache"]["write_status"], "stored")
        for name in ("create_pinecone_client", "create_embedding_client", "create_jev_client",
                     "HybridRetrievalService", "DocumentGradingService"):
            self.patches[name].reset_mock()
        repeated = tool.invoke({"query": QUERY})
        self.assertEqual(repeated["source"], "mem0")
        self.assertEqual(repeated["result"], first["result"])
        self.assertEqual(repeated["execution"]["pipeline_runs"], 0)
        for name in ("create_pinecone_client", "create_embedding_client", "create_jev_client",
                     "HybridRetrievalService", "DocumentGradingService", "query_equivalence"):
            self.patches[name].assert_not_called()

    def test_paraphrase_hit_uses_query_only_check_with_bound_jev_settings(self):
        self.tools["search_knowledge"].invoke({"query": QUERY})
        self.patches["create_pinecone_client"].reset_mock()
        result = self.tools["search_knowledge"].invoke({"query": "How can I access the corporate VPN?"})
        self.assertEqual(result["source"], "mem0")
        self.patches["query_equivalence"].assert_called_once_with(
            "How can I access the corporate VPN?", QUERY, settings=self.service.jev)
        self.patches["create_pinecone_client"].assert_not_called()

    def test_explicit_memory_enabled_reuses_report(self):
        tool = self.tools["search_knowledge"]
        tool.invoke({"query": QUERY})
        self.patches["create_pinecone_client"].reset_mock()
        result = tool.invoke({"query": QUERY, "use_memory": True})
        self.assertEqual(result["source"], "mem0")
        self.patches["create_pinecone_client"].assert_not_called()

    def test_memory_bypass_runs_fresh_pipeline_without_memory_then_defaults_back_on(self):
        tool = self.tools["search_knowledge"]
        first = tool.invoke({"query": QUERY})
        entries = dict(self.memory.entries)
        for mock in self.patches.values():
            mock.reset_mock()
        result = tool.invoke({"query": QUERY, "use_memory": False})
        self.assertEqual(result["source"], "rag_pipeline")
        self.assertEqual(result["result"], first["result"])
        self.assertEqual(result["cache"], {"hit": False, "status": "bypassed", "write_status": "skipped"})
        self.assertEqual(result["execution"]["pipeline_runs"], 1)
        self.assertEqual(result["execution"]["equivalence_checks"], 0)
        self.patches["create_pinecone_client"].assert_called_once()
        self.patches["create_jev_client"].assert_called_once_with(self.service.jev)
        self.patches["create_mem0_client"].assert_not_called()
        self.patches["query_equivalence"].assert_not_called()
        self.assertEqual(self.memory.entries, entries)
        self.assertEqual(tool.invoke({"query": QUERY})["source"], "mem0")

    def test_memory_bypass_still_checks_permissions_and_current_revisions(self):
        self.revisions[DOC] = "c" * 64
        self.tools["search_knowledge"].invoke({"query": QUERY, "use_memory": False})
        call = self.patches["HybridRetrievalService"].return_value.retrieve.call_args
        self.assertEqual(call.kwargs["approved_revisions"], self.revisions)
        for mock in self.patches.values():
            mock.reset_mock()
        self.repo.save_user(replace(self.user, is_active=False))
        response = json.loads(self.tools["search_knowledge"].invoke({"query": QUERY, "use_memory": False}))
        self.assertEqual(response["error"]["code"], "access_denied")
        for name in ("create_mem0_client", "create_pinecone_client", "create_jev_client"):
            self.patches[name].assert_not_called()

    def test_memory_control_requires_a_boolean_before_provider_calls(self):
        for value in (None, "false", "true", 0, 1, [], {}):
            with self.subTest(value=value):
                response = json.loads(self.tools["search_knowledge"].invoke({"query": QUERY, "use_memory": value}))
                self.assertEqual(response["error"]["code"], "invalid_input")
                with self.assertRaises(ValueError):
                    self.service.search_knowledge(QUERY, use_memory=value)
        for name in ("create_mem0_client", "create_pinecone_client", "create_jev_client"):
            self.patches[name].assert_not_called()

    def test_fresh_tools_bypass_memory_and_grade_only_when_requested(self):
        raw = self.tools["retrieve_documents"].invoke({"query": QUERY})
        self.assertEqual(raw, self.result.to_dict())
        self.assertNotIn("grading", raw)
        self.patches["create_jev_client"].assert_not_called()
        graded = self.tools["retrieve_and_grade_documents"].invoke({"query": QUERY})
        self.assertEqual(graded["grading"], self.grade)
        self.patches["create_mem0_client"].assert_not_called()
        self.patches["create_jev_client"].assert_called_once_with(self.service.jev)

    def test_external_mode_creates_query_embeddings_integrated_mode_does_not(self):
        self.tools["retrieve_documents"].invoke({"query": QUERY})
        self.patches["create_embedding_client"].assert_not_called()
        self.patches["get_index_embedding_mode"].return_value = "external"
        self.tools["retrieve_documents"].invoke({"query": QUERY})
        self.patches["create_embedding_client"].assert_called_once_with(self.service.knowledge)

    def test_denied_user_never_constructs_providers_or_memory(self):
        self.repo.save_user(replace(self.user, is_active=False))
        for tool in self.tools.values():
            response = json.loads(tool.invoke({"query": QUERY}))
            self.assertEqual(response["error"]["code"], "access_denied")
        for name in ("create_pinecone_client", "create_mem0_client", "create_jev_client"):
            self.patches[name].assert_not_called()

    def test_query_validation_and_identity_injection_rejected_before_side_effects(self):
        for args in ({"query": " "}, {"query": 5}, {"query": "x" * 4001}, {},
                     {"query": QUERY, "actor_id": str(self.user.id)},
                     {"query": QUERY, "namespace": "other"},
                     {"query": QUERY, "approved_revisions": self.revisions},
                     {"query": QUERY, "documents": []}):
            with self.subTest(args=list(args)):
                response = json.loads(self.tools["search_knowledge"].invoke(args))
                self.assertEqual(response["error"]["code"], "invalid_input")
        self.patches["create_mem0_client"].assert_not_called()
        self.patches["create_pinecone_client"].assert_not_called()

    def test_current_revision_source_is_read_for_each_invocation(self):
        self.tools["retrieve_documents"].invoke({"query": QUERY})
        self.revisions = {DOC: "c" * 64}
        self.tools["retrieve_documents"].invoke({"query": QUERY})
        call = self.patches["HybridRetrievalService"].return_value.retrieve.call_args
        self.assertEqual(call.kwargs["approved_revisions"], self.revisions)
        self.revisions.clear()
        for mock in self.patches.values():
            mock.reset_mock()
        response = json.loads(self.tools["search_knowledge"].invoke({"query": QUERY}))
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.patches["create_mem0_client"].assert_not_called()

    def test_tool_call_returns_langchain_message_with_json_report(self):
        response = self.tools["retrieve_documents"].invoke({
            "type": "tool_call", "name": "retrieve_documents", "id": "call-1", "args": {"query": QUERY},
        })
        self.assertIsInstance(response, ToolMessage)
        self.assertEqual(response.tool_call_id, "call-1")
        self.assertEqual(json.loads(response.content)["documents"], self.result.to_dict()["documents"])

    def test_provider_error_is_sanitized_and_no_partial_report_returned(self):
        self.patches["HybridRetrievalService"].return_value.retrieve.side_effect = RuntimeError("secret-provider-detail")
        response = self.tools["retrieve_and_grade_documents"].invoke({
            "type": "tool_call", "name": "retrieve_and_grade_documents", "id": "call-2", "args": {"query": QUERY},
        })
        self.assertEqual(response.status, "error")
        self.assertNotIn("secret-provider-detail", response.content)
        self.assertEqual(json.loads(response.content)["error"]["code"], "lookup_failed")
        self.patches["create_jev_client"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
