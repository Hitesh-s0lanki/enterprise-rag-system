"""Permission-filtered Pinecone dense/BM25 retrieval with parallel searches."""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.data_ingestion.config import IngestionSettings
from src.db.auth import AuthorizationService, Permission, Role, User

from .config import RetrievalSettings
from .fusion import fuse_rankings

RETURN_FIELDS = [
    "chunk_text", "source", "file_name", "file_type", "document_id", "revision_id",
    "file_sha256", "last_updated_by", "last_updated_at", "last_updated_at_epoch",
    "last_updated_at_source", "ingested_at", "ingested_by_user_id", "ingested_by",
    "allowed_roles", "chunk_id", "chunk_index", "chunk_count", "chunk_sha256", "start_index",
    "embedding_model", "embedding_dimensions",
]


def validate_revisions(approved_revisions: dict[str, str]) -> dict[str, str]:
    """Snapshot exact approved document/revision pairs supplied by trusted code."""
    if not approved_revisions or len(approved_revisions) > 100:
        raise ValueError("Supply 1-100 approved document/revision pairs; unrestricted revision search is disabled")
    revisions = dict(approved_revisions)
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for pair in revisions.items() for value in pair):
        raise ValueError("Document and revision IDs must be 64-character lowercase SHA-256 IDs")
    return revisions


def build_search_filter(user: User, approved_revisions: dict[str, str], knowledge: IngestionSettings) -> dict:
    """Pair document and revision predicates; separate $in lists would cross-match."""
    clauses = [
        {"$or": [
            {"$and": [{"document_id": {"$eq": doc_id}}, {"revision_id": {"$eq": revision}}]}
            for doc_id, revision in sorted(approved_revisions.items())
        ]},
        {"embedding_model": {"$eq": knowledge.embedding_model}},
        {"embedding_dimensions": {"$eq": knowledge.embedding_dimensions}},
    ]
    if user.role != Role.ADMINISTRATOR:
        clauses.append({"allowed_roles": {"$in": [user.role.value]}})
    return {"$and": clauses}


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    metrics: dict

    def to_dict(self) -> dict:
        return {
            "documents": [{"chunk_id": d.id, "text": d.page_content, "metadata": d.metadata}
                          for d in self.documents],
            "metrics": self.metrics,
        }


class HybridRetrievalService:
    def __init__(
        self, *, index, embedding_client: Embeddings | None, authorization: AuthorizationService,
        knowledge: IngestionSettings, settings: RetrievalSettings | None = None,
        embedding_mode: str = "external",
    ):
        if embedding_mode not in {"external", "integrated"}:
            raise ValueError("Unknown embedding mode")
        if embedding_mode == "external" and embedding_client is None:
            raise ValueError("External vector search requires an embedding client")
        self.embedding_mode = embedding_mode
        self.index = index
        self.embedding_client = embedding_client
        self.authorization = authorization
        self.knowledge = knowledge
        self.settings = settings or RetrievalSettings()

    def _search(self, clause: dict, search_filter: dict):
        start = perf_counter()
        response = self.index.documents.search(
            namespace=self.knowledge.namespace, top_k=self.settings.candidate_k,
            score_by=[clause], filter=search_filter, include_fields=RETURN_FIELDS,
            timeout=self.settings.timeout_seconds,
        )
        return response, (perf_counter() - start) * 1000

    def _checked_documents(self, response, actor_id: UUID, revisions: dict[str, str]) -> list[Document]:
        documents = []
        for match in response.matches:
            fields = match.to_dict()
            chunk_id, text = fields.get("_id"), fields.get("chunk_text")
            if (not isinstance(chunk_id, str) or chunk_id != fields.get("chunk_id")
                or not isinstance(text, str) or not text.strip()
                or not isinstance(fields.get("document_id"), str)
                or fields["document_id"] not in revisions
                or revisions.get(fields["document_id"]) != fields.get("revision_id")
                or fields.get("embedding_model") != self.knowledge.embedding_model
                or fields.get("embedding_dimensions") != self.knowledge.embedding_dimensions):
                continue
            roles = fields.get("allowed_roles")
            if not isinstance(roles, list) or not roles:
                continue
            try:
                self.authorization.require_access(
                    actor_id, Permission.READ_KNOWLEDGE, namespace=self.knowledge.namespace, allowed_roles=roles,
                )
            except PermissionError:
                continue
            metadata = {key: fields[key] for key in RETURN_FIELDS if key in fields and key != "chunk_text"}
            if not metadata.get("last_updated_at"):
                metadata["last_updated_at"] = "unknown"
            metadata["retrieval_raw_score"] = fields.get("_score")
            documents.append(Document(id=chunk_id, page_content=text, metadata=metadata))
        return documents

    def retrieve(self, query: str, *, actor_id: UUID, approved_revisions: dict[str, str]) -> RetrievalResult:
        start = perf_counter()
        user = self.authorization.require_access(actor_id, Permission.READ_KNOWLEDGE, namespace=self.knowledge.namespace)
        if not isinstance(query, str) or not query.strip() or len(query) > self.settings.max_query_characters:
            raise ValueError(f"Query must contain text and be at most {self.settings.max_query_characters} characters")
        query = query.strip()
        revisions = validate_revisions(approved_revisions)
        search_filter = build_search_filter(user, revisions, self.knowledge)
        embedding_start = perf_counter()
        if self.embedding_mode == "external":
            vector = self.embedding_client.embed_query(query)
            if len(vector) != self.knowledge.embedding_dimensions or not all(math.isfinite(x) for x in vector):
                raise ValueError("Query embedding has invalid values or dimensions")
            dense_clause = {"type": "dense_vector", "fields": ["embedding"], "values": vector}
            embedding_latency = (perf_counter() - embedding_start) * 1000
        else:
            dense_clause = {"type": "embed", "fields": ["chunk_text"], "query": query}
            embedding_latency = None  # Included in the integrated dense search, not measured separately.
        retrieval_start = perf_counter()
        # Both signals are required; provider failure propagates instead of silently degrading to one.
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self._search, dense_clause, search_filter)
            keyword_future = executor.submit(self._search, {
                "type": "text", "fields": ["chunk_text"], "query": query,
            }, search_filter)
            dense_response, dense_ms = dense_future.result()
            keyword_response, keyword_ms = keyword_future.result()
        dense = self._checked_documents(dense_response, actor_id, revisions)
        keyword = self._checked_documents(keyword_response, actor_id, revisions)
        documents = fuse_rankings(dense, keyword, self.settings)
        # Recheck current user state even when there were no candidates to validate.
        self.authorization.require_access(actor_id, Permission.READ_KNOWLEDGE, namespace=self.knowledge.namespace)
        for document in documents:
            self.authorization.require_access(
                actor_id, Permission.READ_KNOWLEDGE, namespace=self.knowledge.namespace,
                allowed_roles=document.metadata["allowed_roles"],
            )
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        try:
            encoding = tiktoken.encoding_for_model(self.knowledge.embedding_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        estimated_tokens = len(encoding.encode(query, disallowed_special=()))
        read_units = [getattr(response.usage, "read_units", None) if response.usage else None
                      for response in (dense_response, keyword_response)]
        return RetrievalResult(documents=documents, metrics={
            "retrieval_latency_ms": retrieval_ms, "total_latency_ms": (perf_counter() - start) * 1000,
            "embedding_latency_ms": embedding_latency,
            "dense_search_latency_ms": dense_ms, "keyword_search_latency_ms": keyword_ms,
            "embedding_calls": 1, "search_calls": 2,
            "embedding_input_tokens_estimated": estimated_tokens,
            "embedding_mode": self.embedding_mode,
            "embedding_input_tokens_actual": (
                getattr(dense_response.usage, "embed_total_tokens", None)
                if self.embedding_mode == "integrated" and dense_response.usage else None
            ),
            "pinecone_read_units": sum(read_units) if all(v is not None for v in read_units) else None,
            "dense_candidates": len(dense), "keyword_candidates": len(keyword),
            "returned_chunks": len(documents), "estimated_cost": None,
        })
