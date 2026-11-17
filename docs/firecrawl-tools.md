# Firecrawl MCP tools

The root agent registers three public-web tools explicitly through `src/tools/registry.py`. LangChain adapters live in `src/tools/firecrawl.py`; the shared MCP client lives in `src/clients/firecrawl_client.py`. Construction makes no network calls. The CLI and Streamlit already load `.env` before building their root agents.

Configure `FIRECRAWL_API_KEY` privately in `.env`. The client connects to `https://mcp.firecrawl.dev/v2/mcp` using Streamable HTTP and the header `Authorization: Bearer <FIRECRAWL_API_KEY>`, following the [official Firecrawl MCP setup](https://github.com/firecrawl/firecrawl-mcp-server#hosted-mcp-keyless-free-tier). Credentials are never tool arguments or part of the URL. This application is an MCP client; it does not host a new server.

| Tool | Model arguments | Behavior |
| --- | --- | --- |
| `firecrawl_search` | `query` | Public web search, limited to five results per attempt; Jev grading and up to two query enhancements |
| `firecrawl_scrape` | `url` | Read a public page as Markdown, with main content only |
| `firecrawl_map` | `url` | Discover up to twenty site URLs |

Only these tools are registered. Remote tool discovery does not automatically add tools. Crawling jobs, browser interactions, monitors, extraction jobs and other remote tools are not exposed. Arguments accept at most 4,000 characters; unknown arguments are rejected. The agent must keep private company text, credentials, personal data and private URLs out of web requests. URL validation checks syntax and credentials; it does not provide a network egress policy.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIRECRAWL_API_KEY` | unset | Missing/blank key or the example placeholder disables registration |
| `FIRECRAWL_ENABLED` | `true` | Set `false` to disable registration even with a key |
| `FIRECRAWL_MCP_URL` | `https://mcp.firecrawl.dev/v2/mcp` | Trusted HTTPS MCP endpoint; no embedded credentials or query string |
| `FIRECRAWL_TIMEOUT_SECONDS` | `30` | Deadline per attempt, including connection, initialization and tool call |
| `FIRECRAWL_MAX_RETRIES` | `2` | Retries after the first attempt, from zero to five |
| `FIRECRAWL_RETRY_DELAY_SECONDS` | `1` | Initial exponential backoff, capped at ten seconds per delay |

Each attempt opens a fresh HTTP client and MCP session, initializes it, calls the tool and closes the resources. Connection failures, timeouts, HTTP 408/429/5xx, MCP tool-error results and other transport failures retry up to the configured bound. Other HTTP 4xx and invalid MCP request/method/argument errors stop immediately. A tool-error result does not always contain a machine-readable error category, so it uses the same bounded retry policy; a failing read may still consume provider credits. There is no second REST fallback.

Cancellation propagates through requests and backoff without retrying. The root agent's overall deadline can stop attempts earlier. With defaults, one failing tool can take approximately 93 seconds (three 30-second attempts and 1+2 seconds backoff), within the root's 120-second total budget only if enough time remains. Provider error bodies are replaced by a sanitized `firecrawl_failed` tool error. The model can explain the failure and finish; `run_completed` does not imply a successful web request.

## Authorization and evidence

The tools bind the existing trusted actor and namespace from `KnowledgeQueryService`. The client rechecks `READ_KNOWLEDGE` and namespace access before every attempt and before returning results. The root agent also checks authorization throughout streaming. Tools cannot select identity, permissions or approved revisions.

Successful output contains `source: public_web`, the tool name, MCP content blocks and structured content when supplied. Search also includes `grading`, described below. Scrape and map remain ungraded because their URL-only requests carry no question to grade or enhance. Public web results do not receive company document ACL checks. The root prompt requires URL citations and separation of external guidance from company policy; web tools must not bypass denied or insufficient internal evidence. These instructions are not a deterministic data-loss-prevention or final-answer grading system.

## Search evidence middleware

The application is the MCP client, so middleware wraps the local search adapter rather than modifying the hosted Firecrawl server:

```text
Root → firecrawl_search tool → Firecrawl MCP → Jev web-evidence grade
                                ↑                    ↓ weak evidence
                                └── enhanced query ← QueryEnhancementAgent
Root ← tool result + grade/history (accepted or after two enhancements)
```

`src/tools/firecrawl_middleware.py` owns the loop. The root never receives an enhancement tool and has no direct reference to `QueryEnhancementAgent`. The agent inherits `BaseAgent`, uses the shared chat configuration, loads `src/agents/prompts/query_enhancement_agent.md`, and has no tools. It receives only the original public query, previous search queries and numeric grading feedback; web page text is excluded from rewrite input. It makes one model call per enhancement. The middleware closes its owned enhancement agent on success, error or cancellation. Rewrite tokens are excluded from streamed answer text; model lifecycle and usage events still count them.

`src/grading/web.py` uses the shared async Jev client to grade the returned search evidence as a bundle against the **original query on every attempt**, so rewriting cannot lower the evidence target. It uses `GRADING_RELEVANCE_THRESHOLD` and `GRADING_EVIDENCE_THRESHOLD` with the same normalized rubric as document grading. Evidence sent to Jev is capped by `GRADING_MAX_DOCUMENT_CHARACTERS`; truncation is reported, and the grade applies to that excerpt. This separate web grader does not assign company roles or manufacture approved revisions for external documents. Jev grades evidence, not the final answer.

An accepted result returns immediately. Otherwise, at most two enhancements cause new searches: three total logical MCP searches, each with its existing transport retries. Repeated rewritten queries stop early. The final payload retains the last MCP result, including URLs when the provider supplies them, plus `grading.original_query`, `grading.status`, `grading.attempts` (queries and grades), `grading.enhancements`, and `grading.max_enhancements`. Exhausted weak evidence is explicitly `insufficient_evidence`; no answer is manufactured. Grade/rewrite failures return a sanitized `firecrawl_failed` error, never a silently ungraded result. The root prompt instructs the root to explain insufficient evidence rather than repeating equivalent searches to bypass the bound. The bound is per tool invocation, not a cross-invocation quota.

Search now requires valid TypeSafe credentials, and weak results also require OpenAI credentials. There is no ungraded fallback when either provider fails. No new environment variables or dependencies are required. The middleware has a 120-second overall deadline, and the root's remaining run deadline can interrupt it earlier. Authorization is checked before and after each provider stage, and async Jev/model/MCP cancellation propagates without starting a new search. This changes search latency and provider cost; it does not change the root's configured call limits or run budget.

Trusted callers can use `RootAgent(service, include_firecrawl=False)` to disable web tools or inject a `FirecrawlClient` through `firecrawl_client=`. Library callers load environment configuration themselves. The standalone adapters are async LangChain tools; the root's existing `stream`/`run` wrappers support synchronous consumers.

## Verification

Run `uv run --locked python -m unittest discover -s tests -v`. Offline tests cover Bearer headers, MCP initialization and cleanup, reconnects after failed initialization, backoff, retry exhaustion, HTTP error classification, MCP error results, deadlines, cancellation, authorization revocation, argument validation and real scripted LangChain root-tool streaming. No live Firecrawl request is required or implied by these checks.
