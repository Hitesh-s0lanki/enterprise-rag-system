# Query enhancement agent

`QueryEnhancementAgent` in `src/agents/query_enhancement_agent.py` inherits `BaseAgent` and loads its role instructions from `src/agents/prompts/query_enhancement_agent.md`. It uses the existing shared `AGENT_*` model configuration and inherited model invocation APIs. It has no tools, company-knowledge service, caller-selected identity, or direct root-agent connection.

The Firecrawl search middleware creates the agent lazily after a weak Jev grade. `enhance(original_query, attempts)` returns one bounded public-web query. Each attempt contains a public search query and grading metadata. Raw retrieved content is excluded. The middleware owns access checks, the maximum of two enhancements, provider calls, repeated-query detection, result reporting and agent cleanup. The agent cannot select its retry budget.

See [Firecrawl tools](firecrawl-tools.md#search-evidence-middleware) for the complete flow, output contract, failure handling and limitations. This agent enhances public searches; internal Pinecone retrieval and company document grading retain their existing workflow.
