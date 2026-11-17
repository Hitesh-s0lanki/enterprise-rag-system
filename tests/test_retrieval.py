"""Hybrid retrieval behavior using real SDK response types and offline clients."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

from pinecone import Document as PineconeDocument
from pinecone.models.documents.responses import SearchDocumentsResponse, DocumentSearchUsage

from src.data_ingestion.config import IngestionSettings
from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database
from src.retrieval import HybridRetrievalService, HybridRetriever, RetrievalSettings

DOC_A, DOC_B, REV_A, REV_B = "a" * 64, "b" * 64, "c" * 64, "d" * 64


def hit(document_id=DOC_A, revision=REV_A, number=0, text="Connect to VPN with MFA.", score=0.8, **fields):
    chunk_id = f"{document_id}:{revision}:{number:06d}"
    return PineconeDocument({
        "_id": chunk_id, "_score": score, "chunk_id": chunk_id, "chunk_text": text,
        "document_id": document_id, "revision_id": revision, "chunk_index": number,
        "allowed_roles": ["employee", "contributor", "administrator"],
        "embedding_model": "text-embedding-3-small", "embedding_dimensions": 3,
        "file_name": "handbook.md", "source": "handbook.md",
        "last_updated_by": "bob", "last_updated_at": "2026-11-16T00:00:00Z",
        **fields,
    })


def response(matches):
    return SearchDocumentsResponse(namespace="it-support", matches=matches, usage=DocumentSearchUsage(read_units=1))


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = SQLiteUserRepository(initialize_database(Path(self.temp.name) / "auth.sqlite3"))
        self.authorization = AuthorizationService(self.repository)
        self.alice = self.repository.get_by_username("alice")
        self.knowledge = IngestionSettings(embedding_dimensions=3)
        self.settings = RetrievalSettings(top_k=3, candidate_k=5, dense_weight=0.5)
        self.embedder = Mock()
        self.embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        self.index = Mock()
        self.rankings = {"dense_vector": [], "text": []}
        self.index.documents.search.side_effect = lambda **kw: response(self.rankings[kw["score_by"][0]["type"]])
        self.service = HybridRetrievalService(
            index=self.index, embedding_client=self.embedder, authorization=self.authorization,
            knowledge=self.knowledge, settings=self.settings,
        )
        encoder = Mock()
        encoder.encode.return_value = [1, 2, 3]
        self.encoder_patch = patch("src.retrieval.service.tiktoken.encoding_for_model", return_value=encoder)
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)

    def retrieve(self, **kwargs):
        return self.service.retrieve("How do I connect to VPN?", actor_id=self.alice.id,
                                     approved_revisions={DOC_A: REV_A, DOC_B: REV_B}, **kwargs)

    def test_fuses_both_signals_deduplicates_and_preserves_source_metadata(self):
        a, b, c = hit(number=0), hit(number=1, text="Reset VPN client."), hit(DOC_B, REV_B, text="VPN gateway.")
        self.rankings = {"dense_vector": [a, b], "text": [c, a]}
        result = self.retrieve()
        self.assertEqual([d.id for d in result.documents], [a.id, c.id, b.id])
        first = result.documents[0]
        self.assertEqual(first.metadata["file_name"], "handbook.md")
        self.assertEqual(first.metadata["last_updated_by"], "bob")
        self.assertEqual(first.metadata["last_updated_at"], "2026-11-16T00:00:00Z")
        self.assertEqual(set(first.metadata["retrieval"]["signals"]), {"dense", "keyword"})
        self.assertAlmostEqual(first.metadata["retrieval"]["rrf_score"], 0.5 / 61 + 0.5 / 62)
        self.assertEqual(result.metrics["pinecone_read_units"], 2)
        self.assertEqual(result.metrics["embedding_input_tokens_estimated"], 3)
        self.assertIsNone(result.metrics["embedding_input_tokens_actual"])
        self.assertGreaterEqual(result.metrics["total_latency_ms"], result.metrics["retrieval_latency_ms"])
        self.embedder.embed_query.assert_called_once()

    def test_both_searches_apply_identical_role_model_and_paired_revision_filters(self):
        self.retrieve()
        calls = self.index.documents.search.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["filter"], calls[1].kwargs["filter"])
        clauses = calls[0].kwargs["filter"]["$and"]
        self.assertIn({"allowed_roles": {"$in": ["employee"]}}, clauses)
        self.assertEqual(clauses[0]["$or"], [
            {"$and": [{"document_id": {"$eq": DOC_A}}, {"revision_id": {"$eq": REV_A}}]},
            {"$and": [{"document_id": {"$eq": DOC_B}}, {"revision_id": {"$eq": REV_B}}]},
        ])
        self.assertIn({"embedding_model": {"$eq": self.knowledge.embedding_model}}, clauses)
        for call in calls:
            self.assertEqual(call.kwargs["namespace"], "it-support")
            self.assertEqual(call.kwargs["top_k"], 5)
            self.assertNotIn("embedding", call.kwargs["include_fields"])

    def test_unpermitted_stale_cross_paired_and_malformed_records_never_return(self):
        good = hit()
        self.rankings["dense_vector"] = [
            hit(number=1, allowed_roles=["administrator"]),
            hit(number=2, revision=REV_B),  # Revision belongs to a different document.
            hit(number=3, allowed_roles=[]),
            hit(number=4, embedding_dimensions=1536),
            hit(number=5, embedding_model="different-model"),
            hit(number=6, chunk_text=""),
            hit(number=7, document_id="e" * 64, revision_id=None),
            good,
        ]
        self.assertEqual([d.id for d in self.retrieve().documents], [good.id])

    def test_empty_results_are_not_an_error(self):
        result = self.retrieve()
        self.assertEqual(result.documents, [])
        self.assertEqual(result.metrics["returned_chunks"], 0)

    def test_duplicate_hit_does_not_inflate_fusion_score(self):
        self.rankings = {"dense_vector": [hit(), hit()], "text": [hit()]}
        docs = self.retrieve().documents
        self.assertEqual(len(docs), 1)
        self.assertAlmostEqual(docs[0].metadata["retrieval"]["rrf_score"], 1 / 61)

    def test_raw_score_scale_cannot_dominate_rank_fusion(self):
        self.rankings = {"dense_vector": [hit(score=0.4)],
                         "text": [hit(DOC_B, REV_B, score=999999)]}
        docs = self.retrieve().documents
        self.assertEqual(docs[0].metadata["retrieval"]["rrf_score"], docs[1].metadata["retrieval"]["rrf_score"])

    def test_cosine_cutoff_is_inclusive_and_applies_before_fusion(self):
        self.rankings["dense_vector"] = [hit(number=0, score=0.299), hit(number=1, score=0.3)]
        result = self.retrieve()
        self.assertEqual([d.id for d in result.documents], [hit(number=1).id])
        self.assertEqual(result.metrics["dense_candidates_filtered"], 1)
        self.assertEqual(result.metrics["min_cosine_similarity"], 0.3)
        self.assertAlmostEqual(result.documents[0].metadata["retrieval"]["rrf_score"], 0.5 / 61)

    def test_weak_dense_match_can_still_qualify_through_keyword_search(self):
        self.rankings = {"dense_vector": [hit(score=0.1)], "text": [hit(score=12)]}
        result = self.retrieve()
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].metadata["retrieval"]["signals"], {
            "keyword": {"rank": 1, "raw_score": 12},
        })

    def test_missing_nonfinite_or_out_of_range_dense_scores_are_excluded(self):
        for score in (None, float("nan"), float("inf"), -1.1, 1.1, True, "0.8"):
            with self.subTest(score=score):
                self.rankings["dense_vector"] = [hit(score=score)]
                self.assertEqual(self.retrieve().documents, [])

    def test_cosine_cutoff_settings_validation_and_environment(self):
        for cutoff in (float("nan"), float("inf"), -1.01, 1.01):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                RetrievalSettings(min_cosine_similarity=cutoff)
        for cutoff in (-1, 0, 1):
            self.assertEqual(RetrievalSettings(min_cosine_similarity=cutoff).min_cosine_similarity, cutoff)
        with patch.dict("os.environ", {"RETRIEVAL_MIN_COSINE_SIMILARITY": "0.65"}, clear=True):
            self.assertEqual(RetrievalSettings.from_env().min_cosine_similarity, 0.65)

    def test_unknown_inactive_and_out_of_scope_users_do_not_call_providers(self):
        for actor_id in (UUID(int=99), self.alice.id):
            if actor_id == self.alice.id:
                self.repository.save_user(replace(self.alice, is_active=False))
            with self.assertRaises(PermissionError):
                self.service.retrieve("VPN", actor_id=actor_id, approved_revisions={DOC_A: REV_A})
        self.repository.save_user(replace(self.alice, namespaces=("finance",)))
        with self.assertRaises(PermissionError):
            self.retrieve()
        self.embedder.embed_query.assert_not_called()
        self.index.documents.search.assert_not_called()

    def test_invalid_queries_revisions_and_vectors_fail_without_search(self):
        for query, revisions in [(" ", {DOC_A: REV_A}), ("a" * 4001, {DOC_A: REV_A}), ("VPN", {}),
                                 ("VPN", {"bad": "bad"})]:
            with self.assertRaises(ValueError):
                self.service.retrieve(query, actor_id=self.alice.id, approved_revisions=revisions)
        self.embedder.embed_query.assert_not_called()
        self.embedder.embed_query.return_value = [float("nan"), 0, 0]
        with self.assertRaises(ValueError):
            self.retrieve()
        self.index.documents.search.assert_not_called()

    def test_provider_failure_is_not_silently_returned_as_hybrid_success(self):
        def search(**kwargs):
            if kwargs["score_by"][0]["type"] == "text":
                raise TimeoutError("Keyword search failed")
            return response([hit()])
        self.index.documents.search.side_effect = search
        with self.assertRaises(TimeoutError):
            self.retrieve()

    def test_admin_access_stays_namespace_scoped(self):
        self.rankings["text"] = [hit(allowed_roles=["contributor"])]
        mice = self.repository.get_by_username("mice")
        result = self.service.retrieve("VPN", actor_id=mice.id, approved_revisions={DOC_A: REV_A})
        self.assertEqual(len(result.documents), 1)
        self.assertNotIn("allowed_roles", str(self.index.documents.search.call_args_list[0].kwargs["filter"]))

    def test_conflicting_text_for_same_chunk_fails(self):
        self.rankings = {"dense_vector": [hit()], "text": [hit(text="Conflicting content.")]}
        with self.assertRaises(ValueError):
            self.retrieve()

    def test_langchain_invoke_returns_documents(self):
        self.rankings["text"] = [hit()]
        retriever = HybridRetriever(service=self.service, actor_id=self.alice.id, approved_revisions={DOC_A: REV_A})
        self.assertEqual(retriever.invoke("VPN")[0].page_content, "Connect to VPN with MFA.")

    def test_missing_source_update_time_is_explicitly_unknown(self):
        self.rankings["text"] = [hit(last_updated_at=None)]
        self.assertEqual(self.retrieve().documents[0].metadata["last_updated_at"], "unknown")

    def test_integrated_search_uses_matching_text_embed_clause_without_external_call(self):
        service = HybridRetrievalService(
            index=self.index, embedding_client=self.embedder, authorization=self.authorization,
            knowledge=self.knowledge, settings=self.settings, embedding_mode="integrated",
        )
        def search(**kwargs):
            result = response([hit(score=0.1), hit(number=1, score=0.8)])
            if kwargs["score_by"][0]["type"] == "embed":
                self.assertEqual(kwargs["score_by"][0]["fields"], ["chunk_text"])
                self.assertEqual(kwargs["score_by"][0]["query"], "VPN")
                result.usage.embed_total_tokens = 7
            return result
        self.index.documents.search.side_effect = search
        result = service.retrieve("VPN", actor_id=self.alice.id, approved_revisions={DOC_A: REV_A})
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.metrics["dense_candidates_filtered"], 1)
        weak = next(d for d in result.documents if d.id == hit().id)
        self.assertEqual(set(weak.metadata["retrieval"]["signals"]), {"keyword"})
        self.assertEqual(result.metrics["embedding_input_tokens_actual"], 7)
        self.assertIsNone(result.metrics["embedding_latency_ms"])
        self.embedder.embed_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
