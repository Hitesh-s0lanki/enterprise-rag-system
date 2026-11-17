"""Application-bound knowledge operations shared by agent tool adapters."""

from collections.abc import Callable, Mapping
from dataclasses import asdict
from time import perf_counter
from uuid import UUID

from src.clients.jev_client import JevSettings, create_jev_client
from src.clients.mem0_client import create_mem0_client
from src.clients.pinecone_client import create_pinecone_client, get_index_embedding_mode, get_pinecone_index
from src.data_ingestion.config import IngestionSettings
from src.data_ingestion.embeddings import create_embedding_client
from src.db.auth import AuthorizationService, Permission
from src.grading import DocumentGradingService, GradingSettings
from src.grading.service import RUBRIC_VERSION
from src.retrieval import HybridRetrievalService, RetrievalSettings
from src.retrieval.service import validate_revisions

from .config import QueryCacheSettings
from .equivalence import query_equivalence
from .service import MemoryFirstQueryService


class KnowledgeQueryService:
    """Bind a trusted caller and revision source; never accept them from an LLM.

    A mapping snapshots approvals for this service's lifetime. A callable reads
    current approvals per invocation and is preferred for long-lived agents.
    Construction reads settings only; it does not load .env or create clients.
    """

    def __init__(
        self, *, actor_id: UUID, authorization: AuthorizationService,
        approved_revisions: Mapping[str, str] | Callable[[], Mapping[str, str]],
        knowledge: IngestionSettings | None = None, retrieval: RetrievalSettings | None = None,
        grading: GradingSettings | None = None, jev: JevSettings | None = None,
        cache: QueryCacheSettings | None = None,
    ):
        if not isinstance(actor_id, UUID):
            raise ValueError("actor_id must be the authenticated user's UUID")
        self.actor_id = actor_id
        self.authorization = authorization
        self._publication_catalog = None
        if callable(approved_revisions):
            self._revision_source = approved_revisions
        else:
            snapshot = validate_revisions(dict(approved_revisions))
            self._revision_source = lambda: dict(snapshot)
        self.knowledge = knowledge if knowledge is not None else IngestionSettings.from_env()
        self.retrieval = retrieval if retrieval is not None else RetrievalSettings.from_env()
        self.grading = grading if grading is not None else GradingSettings.from_env()
        self.jev = jev if jev is not None else JevSettings.from_env()
        self.cache = cache if cache is not None else QueryCacheSettings.from_env()

    def follow_published_revisions(self, catalog):
        """Refresh approved document heads without expanding the caller's document set."""
        self._publication_catalog = catalog

    def _preflight(self, query: str, *, graded: bool) -> tuple[str, dict[str, str]]:
        self.authorization.require_access(self.actor_id, Permission.READ_KNOWLEDGE,
                                          namespace=self.knowledge.namespace)
        limit = self.retrieval.max_query_characters
        if graded:
            limit = min(limit, self.grading.max_query_characters)
        if not isinstance(query, str) or not query.strip() or len(query) > limit:
            raise ValueError(f"Query must contain text and be at most {limit} characters")
        revisions = validate_revisions(dict(self._revision_source()))
        if self._publication_catalog is not None:
            published = self._publication_catalog.approved_revisions(self.knowledge.namespace)
            revisions = {doc_id: published.get(doc_id, revision) for doc_id, revision in revisions.items()}
        return query.strip(), validate_revisions(revisions)

    def _pipeline(self, query: str, revisions: dict[str, str], *, graded: bool) -> dict:
        with create_pinecone_client() as client:
            with get_pinecone_index(client, self.knowledge) as index:
                mode = get_index_embedding_mode(client, self.knowledge)
                embeddings = create_embedding_client(self.knowledge) if mode == "external" else None
                retriever = HybridRetrievalService(
                    index=index, embedding_client=embeddings, authorization=self.authorization,
                    knowledge=self.knowledge, settings=self.retrieval, embedding_mode=mode,
                )
                result = retriever.retrieve(query, actor_id=self.actor_id, approved_revisions=revisions)
        output = result.to_dict()
        if graded:
            with create_jev_client(self.jev) as client:
                grader = DocumentGradingService(client=client, authorization=self.authorization,
                                               namespace=self.knowledge.namespace, settings=self.grading)
                output["grading"] = grader.grade(query, result.documents, actor_id=self.actor_id,
                                                 approved_revisions=revisions).to_dict()
        return output

    def retrieve_documents(self, query: str) -> dict:
        """Fresh hybrid retrieval only; returned chunks have not been evidence graded."""
        query, revisions = self._preflight(query, graded=False)
        return self._pipeline(query, revisions, graded=False)

    def retrieve_and_grade_documents(self, query: str) -> dict:
        """Fresh retrieval plus Jev evidence grading, bypassing Mem0."""
        query, revisions = self._preflight(query, graded=True)
        return self._pipeline(query, revisions, graded=True)

    def search_knowledge(self, query: str, *, use_memory: bool = True) -> dict:
        """Return a full report; optionally bypass all Mem0 access for this call."""
        start = perf_counter()
        if not isinstance(use_memory, bool):
            raise ValueError("use_memory must be a boolean")
        query, revisions = self._preflight(query, graded=True)
        if not use_memory:
            report = self._pipeline(query, revisions, graded=True)
            return {
                "query": query, "source": "rag_pipeline", "result": report,
                "cache": {"hit": False, "status": "bypassed", "write_status": "skipped"},
                "execution": {"pipeline_runs": 1, "equivalence_checks": 0,
                              "total_latency_ms": (perf_counter() - start) * 1000},
            }
        service = MemoryFirstQueryService(
            memory=create_mem0_client(), authorization=self.authorization, namespace=self.knowledge.namespace,
            settings=self.cache, pipeline_config={
                "knowledge": asdict(self.knowledge), "retrieval": asdict(self.retrieval),
                "grading": asdict(self.grading), "jev": asdict(self.jev), "rubric_version": RUBRIC_VERSION,
            },
            equivalence=lambda current, saved: query_equivalence(current, saved, settings=self.jev),
        )
        return service.run(query, actor_id=self.actor_id, approved_revisions=revisions,
                           pipeline=lambda: self._pipeline(query, revisions, graded=True))
