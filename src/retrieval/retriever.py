"""LangChain adapter for callers that need a standard retriever interface."""

from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .service import HybridRetrievalService


class HybridRetriever(BaseRetriever):
    service: HybridRetrievalService
    actor_id: UUID
    approved_revisions: dict[str, str]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.service.retrieve(query, actor_id=self.actor_id,
                                     approved_revisions=self.approved_revisions).documents
