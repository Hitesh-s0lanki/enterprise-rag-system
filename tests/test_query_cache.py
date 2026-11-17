"""Memory-first orchestration: hits skip providers and never bypass authorization."""

import io
import json
import tempfile
import unittest
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.clients.mem0_client import Mem0Error
from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database
from src.query import MemoryFirstQueryService, QueryCacheSettings

DOC, REV = "a" * 64, "b" * 64
QUERY = "How do I connect to the VPN?"


def report():
    row = {"chunk_id": "chunk-1", "text": "Use the VPN with MFA.", "metadata": {
        "document_id": DOC, "revision_id": REV, "chunk_id": "chunk-1",
        "allowed_roles": ["employee"], "file_name": "handbook.md",
    }}
    return {"documents": [row], "metrics": {"search_calls": 2}, "grading": {
        "query": QUERY, "documents": [deepcopy(row)],
        "summary": {"status": "sufficient_evidence"}, "metrics": {"jev_calls": 1},
    }}


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = SQLiteUserRepository(initialize_database(Path(self.temp.name) / "auth.sqlite3"))
        self.authorization = AuthorizationService(self.repo)
        self.user = self.repo.get_by_username("alice")
        self.memory = Mock()
        self.memory.search.return_value = {"results": []}
        self.memory.add.return_value = {"status": "SUCCEEDED", "results": [{"id": "memory-1"}]}
        self.pipeline = Mock(side_effect=report)
        self.verifier = Mock(return_value=0.99)
        self.service = MemoryFirstQueryService(memory=self.memory, authorization=self.authorization,
            namespace="it-support", pipeline_config={"version": "test-v1"}, equivalence=self.verifier)

    def run_query(self, query=QUERY, **kwargs):
        return self.service.run(query, actor_id=self.user.id,
                                approved_revisions=kwargs.get("revisions", {DOC: REV}), pipeline=self.pipeline)

    def warm(self):
        result = self.run_query()
        options = self.memory.add.call_args.kwargs
        report_call = self.memory.add.call_args_list[-2]
        self.memory.get.return_value = {"memory": report_call.args[0],
                                        "metadata": deepcopy(report_call.kwargs["metadata"])}
        candidate = {"id": "memory-1", "score": 0.6, "memory": QUERY,
                     "metadata": deepcopy(options["metadata"])}
        self.memory.search.return_value = {"results": [candidate]}
        self.pipeline.reset_mock()
        return result, candidate

    def test_miss_stores_verbatim_report_with_isolated_scope(self):
        result, candidate = self.warm()
        self.assertEqual(result["source"], "rag_pipeline")
        self.assertEqual(result["cache"]["write_status"], "stored")
        options = self.memory.add.call_args.kwargs
        self.assertFalse(options["infer"])
        self.assertTrue(options["immutable"])
        self.assertEqual(options["user_id"], str(self.user.id))
        self.assertEqual(json.loads(self.memory.get.return_value["memory"]), result["result"])
        self.assertEqual(candidate["metadata"]["report_memory_id"], "memory-1")

    def test_exact_hit_skips_pipeline_and_verifier(self):
        original, _ = self.warm()
        result = self.run_query()
        self.assertEqual(result["source"], "mem0")
        self.assertEqual(result["result"], original["result"])
        self.assertEqual(result["execution"]["pipeline_runs"], 0)
        self.assertEqual(result["execution"]["equivalence_checks"], 0)
        self.pipeline.assert_not_called()
        self.verifier.assert_not_called()

    def test_server_filters_prevent_reports_and_expired_queries_crowding_out_a_hit(self):
        _, candidate = self.warm()
        expired = {**candidate, "metadata": {**candidate["metadata"], "stored_at": 0}}
        payload = {**candidate, "memory": self.memory.get.return_value["memory"],
                   "metadata": self.memory.get.return_value["metadata"]}

        def search(query, *, filters, top_k, rerank, **kwargs):
            # Model the live API: top-level metadata is ignored; ranking report
            # payloads first crowds the useful query out unless filters exclude them.
            conditions = filters["AND"]
            has_query_filter = {"metadata": {"schema": "rag-report-cache-v2", "cache_kind": "query"}} in conditions
            cutoff = next((item["created_at"]["gte"] for item in conditions if "created_at" in item), None)
            if not has_query_filter:
                return {"results": [payload] * top_k}
            if cutoff is None:
                return {"results": [expired] * top_k}
            self.assertIsNotNone(datetime.fromisoformat(cutoff).tzinfo)
            self.assertFalse(rerank)
            self.assertNotIn("metadata", kwargs)
            return {"results": [candidate]}

        self.memory.search.side_effect = search
        result = self.run_query()
        self.assertEqual(result["source"], "mem0")
        self.pipeline.assert_not_called()
        self.verifier.assert_not_called()

    def test_lookup_logs_current_cache_status_and_stage_timings_without_content(self):
        self.warm()
        with patch("src.query.service.logger") as logger:
            result = self.run_query()
        fields = logger.info.call_args.kwargs["extra"]
        self.assertEqual(fields["cache_source"], "mem0")
        self.assertEqual(fields["cache_status"], "hit")
        self.assertEqual(fields["pipeline_runs"], 0)
        for stage in ("memory_search_latency_ms", "memory_fetch_latency_ms", "total_latency_ms"):
            self.assertGreaterEqual(result["execution"][stage], 0)
            self.assertEqual(fields[stage], result["execution"][stage])
        self.assertNotIn(QUERY, str(logger.info.call_args))
        self.assertNotIn("Use the VPN with MFA.", str(logger.info.call_args))

    def test_exact_repeat_precedes_higher_scoring_other_queries(self):
        _, candidate = self.warm()
        other = {**candidate, "memory": "VPN failure troubleshooting", "score": 0.99}
        candidate["score"] = 0.1
        self.memory.search.return_value = {"results": [other, candidate]}
        self.assertEqual(self.run_query()["source"], "mem0")
        self.verifier.assert_not_called()
        self.pipeline.assert_not_called()

    def test_paraphrase_reuses_report_after_query_only_verification(self):
        self.warm()
        result = self.run_query("How can I access the corporate VPN?")
        self.assertEqual(result["source"], "mem0")
        self.assertEqual(result["cache"]["matched_query"], QUERY)
        self.assertEqual(result["result"]["grading"]["query"], QUERY)
        self.assertEqual(result["execution"]["equivalence_checks"], 1)
        self.pipeline.assert_not_called()

    def test_same_topic_different_intent_runs_pipeline(self):
        self.warm()
        self.verifier.return_value = 0.01
        result = self.run_query("How do I troubleshoot a VPN failure?")
        self.assertEqual(result["source"], "rag_pipeline")
        self.pipeline.assert_called_once()

    def test_unavailable_equivalence_check_is_a_miss(self):
        self.warm()
        self.verifier.side_effect = RuntimeError("verifier down")
        self.assertEqual(self.run_query("VPN access steps?")["source"], "rag_pipeline")

    def test_unavailable_payload_fetch_is_a_miss(self):
        self.warm()
        self.memory.get.side_effect = Mem0Error("memory unavailable")
        self.assertEqual(self.run_query()["source"], "rag_pipeline")
        self.pipeline.assert_called_once()

    def test_changed_payload_or_wrong_payload_scope_is_rejected(self):
        self.warm()
        stored = deepcopy(self.memory.get.return_value)
        for change in ({"memory": "modified"}, {"metadata": {**stored["metadata"], "actor_id": "other"}}):
            self.memory.get.return_value = {**stored, **change}
            self.assertEqual(self.run_query()["source"], "rag_pipeline")

    def test_stale_low_score_malformed_or_corrupt_candidates_are_misses(self):
        _, candidate = self.warm()
        for change in (
            {"score": 0.1, "memory": "VPN access steps?"}, {"score": float("nan")}, {"metadata": {}},
            {"metadata": {**candidate["metadata"], "stored_at": 0}},
            {"metadata": {**candidate["metadata"], "report_sha256": "corrupted"}},
            {"metadata": {**candidate["metadata"], "actor_id": "someone-else"}},
            {"metadata": {**candidate["metadata"], "namespace": "other"}},
            {"metadata": {**candidate["metadata"], "schema": "old"}},
        ):
            with self.subTest(change=change):
                self.memory.search.return_value = {"results": [{**candidate, **change}]}
                self.assertEqual(self.run_query()["source"], "rag_pipeline")

    def test_revision_or_configuration_changes_invalidate_cache(self):
        self.warm()
        result = self.run_query(revisions={DOC: "c" * 64})
        self.assertEqual(result["source"], "rag_pipeline")
        self.service.pipeline_config["version"] = "test-v2"
        self.assertEqual(self.run_query()["source"], "rag_pipeline")

    def test_other_actor_cannot_reuse_result(self):
        self.warm()
        self.user = self.repo.get_by_username("mice")
        self.assertEqual(self.run_query()["source"], "rag_pipeline")

    def test_inactive_user_is_denied_before_memory_or_pipeline(self):
        self.repo.save_user(replace(self.user, is_active=False))
        with self.assertRaises(PermissionError):
            self.run_query()
        self.memory.search.assert_not_called()
        self.pipeline.assert_not_called()

    def test_current_document_roles_rechecked_on_cache_hit(self):
        _, candidate = self.warm()
        # Simulate a role transition during Mem0's network lookup.
        def search(*args, **kwargs):
            from src.db.auth import Role
            self.repo.save_user(replace(self.user, role=Role.CONTRIBUTOR))
            return {"results": [candidate]}
        self.memory.search.side_effect = search
        with self.assertRaises(PermissionError):
            self.run_query()
        self.pipeline.assert_called_once()  # Cached employee content was rejected.

    def test_memory_failure_falls_back_and_write_failure_keeps_output(self):
        self.memory.search.side_effect = Mem0Error("unavailable")
        self.memory.add.side_effect = Mem0Error("unavailable")
        result = self.run_query()
        self.assertEqual(result["source"], "rag_pipeline")
        self.assertEqual(result["cache"]["status"], "unavailable")
        self.assertEqual(result["cache"]["write_status"], "failed")
        self.assertEqual(result["result"], report())

    def test_pending_write_is_not_reported_as_stored(self):
        self.memory.add.return_value = {"status": "PENDING", "event_id": "event"}
        self.assertEqual(self.run_query()["cache"]["write_status"], "pending")

    def test_insufficient_or_oversized_reports_are_not_cached(self):
        insufficient = report()
        insufficient["grading"]["summary"]["status"] = "insufficient_evidence"
        self.pipeline.side_effect = lambda: insufficient
        self.assertEqual(self.run_query()["cache"]["write_status"], "skipped")
        self.memory.add.assert_not_called()
        self.pipeline.side_effect = report
        self.service.settings = QueryCacheSettings(max_payload_bytes=10)
        self.assertEqual(self.run_query()["cache"]["write_status"], "skipped")
        self.memory.add.assert_not_called()

    def test_bad_query_or_revisions_never_call_providers(self):
        for query in ("", "x" * 4001):
            with self.assertRaises(ValueError):
                self.run_query(query)
        with self.assertRaises(ValueError):
            self.run_query(revisions={})
        self.memory.search.assert_not_called()
        self.pipeline.assert_not_called()

    def test_pipeline_error_propagates_without_cache_write(self):
        self.pipeline.side_effect = RuntimeError("retrieval failed")
        with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
            self.run_query()
        self.memory.add.assert_not_called()

    def test_settings_environment_and_invalid_values(self):
        with patch.dict(os.environ, {"MEM0_CACHE_TTL_SECONDS": "300", "MEM0_CACHE_EQUIVALENCE_THRESHOLD": "0.9"}):
            settings = QueryCacheSettings.from_env()
        self.assertEqual(settings.ttl_seconds, 300)
        self.assertEqual(settings.equivalence_threshold, 0.9)
        for options in ({"similarity_threshold": float("nan")}, {"equivalence_threshold": 1.1},
                        {"ttl_seconds": 0}, {"top_k": True}, {"max_payload_bytes": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                QueryCacheSettings(**options)

    def test_cli_hit_never_constructs_retrieval_or_grading_clients(self):
        from src.retrieval.__main__ import main
        with patch("sys.argv", ["retrieval", QUERY, "--user", "alice", "--grade",
                                "--memory-cache", "--revision", f"{DOC}:{REV}"]), \
                patch("src.retrieval.__main__.load_dotenv"), \
                patch("src.retrieval.__main__.SQLiteUserRepository", return_value=self.repo), \
                patch("src.retrieval.__main__.create_mem0_client"), \
                patch("src.retrieval.__main__.MemoryFirstQueryService") as service, \
                patch("src.retrieval.__main__.create_pinecone_client") as pinecone, \
                patch("src.retrieval.__main__.create_embedding_client") as embeddings, \
                patch("src.retrieval.__main__.create_jev_client") as jev, \
                patch("sys.stdout", io.StringIO()):
            service.return_value.run.return_value = {"source": "mem0"}
            main()
        pinecone.assert_not_called()
        embeddings.assert_not_called()
        jev.assert_not_called()


if __name__ == "__main__":
    unittest.main()
