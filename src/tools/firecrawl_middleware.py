"""Bounded retrieve/grade/rewrite loop behind the Firecrawl search tool."""

import asyncio
from collections.abc import Callable

from langchain_core.runnables import RunnableConfig

from src.clients.firecrawl_client import FirecrawlClient
from src.grading.web import WebEvidenceGrader


class FirecrawlSearchMiddleware:
    # Hard application limit, never selected by the root or the enhancement model.
    MAX_ENHANCEMENTS = 2

    def __init__(self, client: FirecrawlClient, *, authorize: Callable[[], object],
                 grader=None, enhancer_factory=None):
        self.client = client
        self.authorize = authorize
        self.grader = grader if grader is not None else WebEvidenceGrader()
        self.enhancer_factory = enhancer_factory

    async def search(self, query: str, *, runnable_config: RunnableConfig | None = None) -> dict:
        async with asyncio.timeout(120):
            return await self._search(query, runnable_config=runnable_config)

    async def _search(self, query: str, *, runnable_config: RunnableConfig | None) -> dict:
        original = current = query
        attempts = []
        seen_queries = {query.strip().casefold()}
        enhancer = None
        try:
            for index in range(self.MAX_ENHANCEMENTS + 1):
                self.authorize()
                result = await self.client.call_tool("firecrawl_search", {"query": current, "limit": 5},
                                                     authorize=self.authorize)
                self.authorize()
                grade = await self.grader.grade(original, result)
                self.authorize()
                attempts.append({"query": current, "grade": grade})
                if grade["accepted"] or index == self.MAX_ENHANCEMENTS:
                    break
                if enhancer is None:
                    if self.enhancer_factory is None:
                        # Lazy import avoids coupling the root's agent imports to tool internals.
                        from src.agents.query_enhancement_agent import QueryEnhancementAgent
                        enhancer = QueryEnhancementAgent()
                    else:
                        enhancer = self.enhancer_factory()
                self.authorize()
                enhanced = await enhancer.enhance(original, attempts, runnable_config=runnable_config)
                self.authorize()
                if (not isinstance(enhanced, str) or not enhanced.strip() or len(enhanced) > 4000):
                    raise ValueError("Invalid enhanced query")
                current = enhanced.strip()
                if current.casefold() in seen_queries:
                    break
                seen_queries.add(current.casefold())
            self.authorize()
            return {**result, "grading": {"original_query": original, "status": grade["status"],
                    "attempts": attempts, "enhancements": len(attempts) - 1,
                    "max_enhancements": self.MAX_ENHANCEMENTS}}
        finally:
            if enhancer is not None:
                await enhancer.aclose()
