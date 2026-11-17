"""Tool-private search rewriting agent, with no root-agent connection or tools."""

import json

from langchain_core.runnables import RunnableConfig

from .base_agent import BaseAgent


class QueryEnhancementAgent(BaseAgent):
    prompt_name = "query_enhancement_agent"

    async def enhance(self, original_query: str, attempts: list[dict], *,
                      runnable_config: RunnableConfig | None = None) -> str:
        # Only public queries and numeric feedback enter this agent, never page text.
        response = await self.ainvoke_model(
            original_query, context=json.dumps({"previous_attempts": attempts}),
            runnable_config={**(runnable_config or {}),
                             "tags": [*((runnable_config or {}).get("tags") or []), "query_enhancement"]},
        )
        query = response.content
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 4000:
            raise ValueError("Query enhancement did not return a bounded search query")
        return query.strip()
