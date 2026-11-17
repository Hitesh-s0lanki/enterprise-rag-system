"""Explicit tools for the root agent; never discover executable files dynamically."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool

from src.query.pipeline import KnowledgeQueryService
from src.clients.firecrawl_client import FirecrawlClient, create_firecrawl_client
from src.db.auth import Permission, SQLiteUserRepository
from src.db.documents import SQLiteDocumentRepository
from src.data_ingestion.updates import DocumentUpdateService

from .knowledge import create_knowledge_tools
from .firecrawl import create_firecrawl_tools
from .document_updates import create_document_update_tools


def create_root_tools(
    service: KnowledgeQueryService, *, include_advanced: bool = False,
    extra_tools: Sequence[BaseTool] = (),
    include_firecrawl: bool = True, firecrawl_client: FirecrawlClient | None = None,
    include_document_updates: bool = True, document_update_service: DocumentUpdateService | None = None,
) -> list[BaseTool]:
    tools = [*create_knowledge_tools(service, include_advanced=include_advanced), *extra_tools]
    if include_document_updates:
        updates = document_update_service
        if updates is None and isinstance(service.authorization.repository, SQLiteUserRepository):
            updates = DocumentUpdateService(
                actor_id=service.actor_id, authorization=service.authorization, settings=service.knowledge,
                catalog=SQLiteDocumentRepository(service.authorization.repository.path),
            )
        if updates is not None:
            if (updates.actor_id != service.actor_id or updates.authorization is not service.authorization
                or updates.settings != service.knowledge):
                raise ValueError("Root update tools must use the same caller, authorization and knowledge settings")
            tools.extend(create_document_update_tools(updates))
            service.follow_published_revisions(updates.catalog)
    if include_firecrawl:
        client = firecrawl_client if firecrawl_client is not None else create_firecrawl_client()
        if client is not None:
            def authorize():
                return service.authorization.require_access(
                    service.actor_id, Permission.READ_KNOWLEDGE, namespace=service.knowledge.namespace,
                )
            tools.extend(create_firecrawl_tools(client, authorize=authorize))
    names = set()
    for tool in tools:
        if not isinstance(tool, BaseTool) or not tool.name or tool.name in names:
            raise ValueError("Root tools must be LangChain tools with unique nonempty names")
        names.add(tool.name)
    return tools
