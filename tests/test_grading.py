"""Evidence grading with real SDK responses and offline provider doubles."""

import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from typesafe_sdk import SystemOneResponse

from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database
from src.grading import DocumentGradingService, GradingSettings
from src.retrieval.service import RetrievalResult

DOC_ID, REV_ID = "a" * 64, "b" * 64


def document(number, **metadata):
    chunk_id = f"{DOC_ID}:{REV_ID}:{number:06d}"
    return Document(id=chunk_id, page_content=f"VPN instructions {number}", metadata={
        "chunk_id": chunk_id, "document_id": DOC_ID, "revision_id": REV_ID,
        "allowed_roles": ["employee"], "file_name": "handbook.md",
        "last_updated_by": "editor", "ingested_by": "uploader", **metadata,
    })


def response(relevance=4, coverage=4, sufficient=0.9):
    def score(value):
        return {"type": "score", "score": value, "confidence": 0.8,
                "probabilities": {0: 0.1, 4: 0.9}, "legend": {0: "none", 4: "complete"}}
    return SystemOneResponse(model="test-jev", usage={"input_tokens": 100, "output_tokens": 20},
                             answers={"relevance": score(relevance), "coverage": score(coverage),
                                      "sufficient_evidence": {"type": "noul", "noul": sufficient}})


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        repository = SQLiteUserRepository(initialize_database(Path(self.temp.name) / "auth.sqlite3"))
        self.authorization = AuthorizationService(repository)
        self.actor_id = repository.get_by_username("alice").id
        self.client = Mock()
        self.client.system_one.return_value = response()
        self.service = DocumentGradingService(client=self.client, authorization=self.authorization,
                                             namespace="it-support")

    def grade(self, documents, query="How do I connect to the VPN?"):
        return self.service.grade(query, documents, actor_id=self.actor_id,
                                  approved_revisions={DOC_ID: REV_ID})

    def test_ranking_report_and_provenance(self):
        self.client.system_one.side_effect = [response(1, 1, 0.1), response(4, 3, 0.95)]
        documents = [document(1), document(2)]
        report = self.grade(documents).to_dict()
        first, second = report["documents"]
        self.assertEqual((first["original_rank"], first["rank"]), (2, 1))
        self.assertEqual(first["metadata"], documents[1].metadata)
        self.assertEqual(first["relevance"], 1)
        self.assertEqual(first["coverage"], 0.75)
        self.assertTrue(first["accepted"])
        self.assertFalse(second["accepted"])
        self.assertEqual(report["summary"]["status"], "sufficient_evidence")
        self.assertEqual(report["summary"]["accepted_chunks"], 1)
        self.assertEqual(report["metrics"]["input_tokens"], 200)
        self.assertEqual(report["metrics"]["output_tokens"], 40)
        self.assertEqual(first["evaluation"]["answers"]["relevance"]["confidence"], 0.8)
        json.dumps(report, allow_nan=False)
        self.assertNotIn("rank", documents[0].metadata)
        self.assertEqual(self.client.system_one.call_args.kwargs["state"]["query"], report["query"])

    def test_stable_ties_and_partial_evidence(self):
        self.client.system_one.return_value = response(3, 2, 0.5)
        report = self.grade([document(1), document(2)]).to_dict()
        self.assertEqual([row["original_rank"] for row in report["documents"]], [1, 2])
        self.assertEqual(report["documents"][0]["decision"], "partial_evidence")
        self.assertEqual(report["summary"]["status"], "insufficient_evidence")

    def test_coverage_breaks_relevance_ties(self):
        self.client.system_one.side_effect = [response(3, 1), response(3, 4)]
        self.assertEqual(self.grade([document(1), document(2)]).documents[0]["original_rank"], 2)

    def test_all_permissions_checked_before_any_provider_call(self):
        for roles in (["administrator"], [], None):
            with self.subTest(roles=roles), self.assertRaises(PermissionError):
                self.grade([document(1), document(2, allowed_roles=roles)])
        self.client.system_one.assert_not_called()

    def test_wrong_namespace_denied_without_provider_call(self):
        self.service.namespace = "unassigned"
        with self.assertRaises(PermissionError):
            self.grade([document(1)])
        self.client.system_one.assert_not_called()

    def test_invalid_inputs_rejected_before_any_provider_call(self):
        cases = [
            ([document(1)], " "), ([document(1)], "x" * 4001),
            ([document(1), document(2, revision_id="c" * 64)], "query"),
            ([document(1), document(1)], "query"),
            ([document(1, chunk_id="different")], "query"),
        ]
        oversized = document(2)
        oversized.page_content = "x" * 20001
        cases.append(([document(1), oversized], "query"))
        cases.append(([document(i) for i in range(21)], "query"))
        for documents, query in cases:
            with self.subTest(query=query[:10], count=len(documents)), self.assertRaises(ValueError):
                self.grade(documents, query)
        self.client.system_one.assert_not_called()

    def test_empty_result_has_complete_summary_without_provider_call(self):
        report = self.grade([]).to_dict()
        self.assertEqual(report["summary"]["status"], "no_documents")
        self.assertIsNone(report["summary"]["best_chunk_id"])
        self.assertEqual(report["metrics"]["jev_calls"], 0)
        self.client.system_one.assert_not_called()

    def test_provider_failure_never_returns_partial_report(self):
        self.client.system_one.side_effect = [response(), RuntimeError("provider unavailable")]
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.grade([document(1), document(2)])

    def test_invalid_or_missing_answers_fail(self):
        for result in (response(relevance=5), response().model_copy(update={"answers": {}})):
            self.client.system_one.return_value = result
            with self.subTest(result=result), self.assertRaises((ValueError, KeyError)):
                self.grade([document(1)])

    def test_access_revoked_after_provider_call_blocks_report(self):
        def revoke(**kwargs):
            self.authorization.require_access = Mock(side_effect=PermissionError("revoked"))
            return response()
        self.client.system_one.side_effect = revoke
        with self.assertRaisesRegex(PermissionError, "revoked"):
            self.grade([document(1)])

    def test_settings_from_environment_and_validation(self):
        with patch.dict(os.environ, {"GRADING_RELEVANCE_THRESHOLD": "0.8", "GRADING_MAX_DOCUMENTS": "5"}):
            settings = GradingSettings.from_env()
        self.assertEqual(settings.relevance_threshold, 0.8)
        self.assertEqual(settings.max_documents, 5)
        for values in ({"relevance_threshold": float("nan")}, {"evidence_threshold": 1.1},
                       {"max_documents": 0}, {"max_document_characters": -1}, {"max_query_characters": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GradingSettings(**values)

    def test_retrieval_cli_includes_real_grading_report(self):
        from src.retrieval.__main__ import main

        output = io.StringIO()
        with patch("sys.argv", ["retrieval", "VPN?", "--user", "alice", "--grade",
                                "--revision", f"{DOC_ID}:{REV_ID}"]), \
                patch("src.retrieval.__main__.load_dotenv"), \
                patch("src.retrieval.__main__.SQLiteUserRepository",
                      return_value=self.authorization.repository), \
                patch("src.retrieval.__main__.create_pinecone_client"), \
                patch("src.retrieval.__main__.get_pinecone_index"), \
                patch("src.retrieval.__main__.get_index_embedding_mode", return_value="external"), \
                patch("src.retrieval.__main__.create_embedding_client"), \
                patch("src.retrieval.__main__.create_jev_client") as factory, \
                patch("src.retrieval.__main__.HybridRetrievalService") as retrieval, \
                patch("sys.stdout", output):
            factory.return_value.__enter__.return_value = self.client
            retrieval.return_value.retrieve.return_value = RetrievalResult([document(1)], {"search_calls": 2})
            main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["metrics"]["search_calls"], 2)
        self.assertEqual(report["grading"]["summary"]["accepted_chunks"], 1)
        self.assertEqual(report["grading"]["query"], "VPN?")
        self.client.system_one.assert_called_once()


if __name__ == "__main__":
    unittest.main()
