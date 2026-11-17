"""Main agent: knowledge tools and caller authorization over the shared runtime."""

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.clients.firecrawl_client import FirecrawlClient
from src.db.auth import Permission
from src.data_ingestion.updates import DocumentUpdateService
from src.query.pipeline import KnowledgeQueryService
from src.tools import create_root_tools
from src.utils.events import AgentRunError

from .base_agent import AgentConfig, AgentRunSettings, BaseAgent

# Existing callers retain their error import while all agents share one type.
RootAgentError = AgentRunError


class RootAgentSettings(AgentRunSettings):
    @classmethod
    def from_env(cls):
        return super().from_env(prefix="ROOT_AGENT")


class RootAgent(BaseAgent):
    prompt_name = "root_agent"

    def __init__(
        self, service: KnowledgeQueryService, config: AgentConfig | None = None, *,
        settings: RootAgentSettings | None = None, llm: BaseChatModel | None = None,
        include_advanced_tools: bool = False, extra_tools: Sequence[BaseTool] = (),
        include_firecrawl: bool = True, firecrawl_client: FirecrawlClient | None = None,
        include_document_updates: bool = True, document_update_service: DocumentUpdateService | None = None,
    ):
        super().__init__(config, llm=llm, settings=settings if settings is not None else RootAgentSettings.from_env())
        self.service = service
        self.tools = create_root_tools(service, include_advanced=include_advanced_tools, extra_tools=extra_tools,
                                       include_firecrawl=include_firecrawl, firecrawl_client=firecrawl_client,
                                       include_document_updates=include_document_updates,
                                       document_update_service=document_update_service)

    def get_tools(self) -> Sequence[BaseTool]:
        return self.tools

    def _authorize(self):
        self.service.authorization.require_access(self.service.actor_id, Permission.READ_KNOWLEDGE,
                                                  namespace=self.service.knowledge.namespace)
