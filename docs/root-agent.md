# Root Agent and Event Streaming

See [application logging](logging.md) for correlated run/tool diagnostics and the OpenAI client event-loop lifecycle fix. Owned chat clients are closed before each managed run's loop ends; injected models remain caller-owned.

`src/agents/root_agent.py` defines the main `RootAgent`, inheriting `BaseAgent`. It uses LangChain `create_agent` for the model/tool loop. Shared instructions come from `src/agents/prompts/base_agent.md` and root-specific instructions from `src/agents/prompts/root_agent.md`.

Execution and streaming are inherited from `src/agents/base_agent.py`: root code supplies its tool registry and authorization hook. `src/utils/events.py` defines `AgentEvent` and `AgentRunError`; `src/utils/streaming.py` normalizes callbacks and maintains per-run event/usage state. The existing sync/async/SSE APIs and event format are unchanged. `RootAgentError` remains an alias for the shared error type.

The explicit tool registry is `src/tools/registry.py`, alongside the existing knowledge tools in `src/tools/knowledge.py`. The default registry exposes `search_knowledge`, which runs the existing authorized memory-first retrieval/grading flow. Its optional `use_memory` boolean defaults to `true`; the root agent is instructed to retry with `false` when a cached report does not adequately answer the current question. This skips Mem0 reads and writes for that call while preserving fresh retrieval, grading, and access checks. Optional advanced tools expose fresh retrieval or retrieval-plus-grading. Adding a Python file does not execute or automatically register a tool; trusted application code must register it explicitly. Duplicate names are rejected.

When `FIRECRAWL_API_KEY` is configured, the registry also includes public-web search, scrape and map tools. See [Firecrawl MCP tools](firecrawl-tools.md) for connection settings, retry behavior and evidence boundaries. Both the CLI and Streamlit use this registry.

For local SQLite authorization, the default root registry also includes
`get_document_for_update` and `update_document`. They bind the same caller,
namespace and database as the knowledge service. The root prompt requires an
explicit user edit request, a full-source read before replacement, and handling
of access denial and revision conflicts without bypasses. Service RBAC remains
authoritative; read-only users cannot write even though the tools are visible.
Use `RootAgent(service, include_document_updates=False)` to disable these tools.
Custom adapters can supply `document_update_service=...`; mismatched identities,
authorization objects or knowledge settings are rejected.

Document enrollment and ACL setup remain explicit maintainer work. No tables are
created and no documents are enrolled when constructing the root agent. See
[document editing setup](document-update-tools.md). An uninitialized catalog
does not break existing read lookups. Once a document has a published catalog
head, root knowledge lookups follow that head within their existing approved
document set, so an edit changes the revision-based memory cache context.

## Trusted application setup

Construct one `KnowledgeQueryService` for the authenticated actor, namespace and approved revision source, then pass it to the root agent. User IDs, roles and revision approvals are never model-controlled tool arguments. A callable revision source is preferable for long-lived agents because the service reads its current approvals for each lookup.

```python
from src.agents import RootAgent
from src.query.pipeline import KnowledgeQueryService

# authorization, actor_id and load_current_approved_revisions come from
# trusted application code, not a user message or the language model.
service = KnowledgeQueryService(
    actor_id=actor_id,
    authorization=authorization,
    approved_revisions=load_current_approved_revisions,
)
agent = RootAgent(service)

for event in agent.stream("How do I connect to the VPN?"):
    handle_event(event.to_dict())
```

Construction does not call providers. The root agent checks read permission and namespace scope before creating its runtime and checks access again while streaming model/tool output. The existing tools enforce document-level access and current revision approvals. Do not share an actor-bound instance with another user. Each run starts with fresh messages; there is no shared global conversation history or checkpoint persistence.

## Event contract

Each `AgentEvent` carries `type`, root `run_id`, a monotonically increasing zero-based `sequence`, a UTC timestamp, JSON-serializable `data`, and optional `source_run_id`/`parent_ids` for correlating nested operations. Consumer-visible events intentionally exclude raw provider exception bodies and hidden reasoning.

| Event | Meaning and payload |
| --- | --- |
| `run_started` | A new request has started. |
| `model_started` | A model turn started; `source_run_id` identifies the turn. |
| `text_delta` | Public model text fragment in `data.text`. |
| `tool_call_delta` | Partial tool name/ID/arguments, including the tool-call `index`. These are display-only fragments, not executable arguments. |
| `model_completed` | That model turn completed. |
| `usage` | That turn's reported usage and aggregate reported usage. |
| `tool_started` | Tool name and validated-runtime input; use `source_run_id` to track this execution. |
| `tool_completed` | Tool output and its provider `tool_call_id` when supplied. |
| `tool_error` | A handled tool failure, or an interrupted/failed execution. Knowledge tools sanitize provider errors. |
| `tool_cancelled` | An active tool execution was cancelled through the explicit cancellation signal. |
| `custom` | A named custom progress event from a tool/callback. |
| `run_completed` | Authoritative final answer plus run usage/call counts and elapsed time. |
| `run_error` | Sanitized failure code/message and available metrics; there is no completed answer. |
| `run_cancelled` | Explicit cancellation was requested; there is no completed answer. |

A fully consumed run emits one start and exactly one terminal event (`run_completed`, `run_error`, or `run_cancelled`). Provider/cleanup failure is not reported as success. A handled tool error can be followed by a model explanation and a completed run; completion means the agent finished, not that an IT operation succeeded or evidence was sufficient.

Group `text_delta` by `source_run_id` because intermediate model turns may precede tool calls. Use the final `run_completed.data.answer` as the authoritative answer rather than concatenating every model turn into it. Do not append that final answer to already-rendered text a second time.

Accumulate tool argument fragments by `(source_run_id, index)`; fragments can be incomplete JSON and can omit the name/ID after the first fragment. Execute nothing from these events. LangChain assembles and validates tool calls before execution. Match tool lifecycle events using `source_run_id`, since parallel tool calls can finish out of order. `tool_completed` also supplies the model's tool-call ID when available.

Usage is aggregated once per completed model turn, not once per token. `usage.complete=false` means some turns did not report usage; totals then represent only known usage. Provider prices and monetary costs are not invented. Tool reports carry their own retrieval, grading, cache and usage metrics separately from root-model usage.

## Synchronous, asynchronous and SSE APIs

- `stream(query)`: synchronous `Iterator[AgentEvent]` for CLI and Streamlit.
- `astream(query, cancel_event=...)`: asynchronous iterator for async applications.
- `stream_sse` / `astream_sse`: identical events framed as Server-Sent Events with IDs `<run_id>:<sequence>`.
- `run` / `arun`: consume the same event path and return the final answer/run metrics dictionary; failures raise `RootAgentError` with a safe `code`.

Use async methods inside an existing asyncio loop. Calling synchronous methods from that loop raises a clear error instead of nesting event loops. These APIs accept `runnable_config` for LangChain callbacks, tags and tracing; configured safety call/recursion limits remain enforced.

```python
import asyncio

cancel = asyncio.Event()
async for event in agent.astream(question, cancel_event=cancel):
    await send_to_client(event.to_dict())
    # Another application task can call cancel.set().
```

For server-side HTTP streaming, return `astream_sse()` as `text/event-stream`. This is framing support, not a new HTTP server or persisted replay queue. Custom progress events are forwarded when tools dispatch them; no artificial progress percentages are generated.

On disconnect, close/aclose the iterator or cancel the consumer task. Cleanup propagates cancellation into the runtime. A closed iterator cannot deliver a final cancellation event; use `cancel_event` and continue consuming when the UI needs an explicit `run_cancelled` event. Synchronous provider calls already running in worker threads cannot be forcibly terminated; they may finish under their own timeouts, and cancellation does not roll back completed external side effects.

## Limits and configuration

The inherited `AGENT_*` settings control model, request timeout, output tokens and initial query/context size. Root-specific controls are:

| Variable | Default |
| --- | --- |
| `ROOT_AGENT_MAX_MODEL_CALLS` | `4` |
| `ROOT_AGENT_MAX_TOOL_CALLS` | `6` |
| `ROOT_AGENT_TIMEOUT_SECONDS` | `120` |

Model/tool limits use LangChain middleware and fail with a terminal error instead of silently running indefinitely. The overall stream timeout bounds orchestration; provider request timeouts still apply. These controls do not implement a full monetary budget or a bound on accumulated tool-result context. The frontend must still manage how much tool output it displays or retains.

## CLI

```sh
uv run --locked python -m src.agents "How do I connect to the VPN?" \
  --user alice --revision DOCUMENT_ID:REVISION_ID --format jsonl
```

Replace the hashes with the intended approved document/revision pair; repeat `--revision` for additional documents. `--format sse` emits SSE frames instead. The CLI is a demo identity selector, not authentication. It loads `.env`; library classes do not. Model generation needs `OPENAI_API_KEY`, and tool operations need their configured provider credentials.

## Verification

Offline tests exercise the real LangChain runtime with scripted streaming chat models and temporary SQLite users. They cover fragmented arguments, text, tool outputs, parallel calls, custom events, usage aggregation, access revocation, sanitized errors, call limits, timeouts, cancellation, cleanup failures and sync/async/SSE paths. No live model/provider calls are needed.

The root prompt requires accepted evidence and citations. Prompt instructions do not constitute a deterministic output-grounding validator; a separate final-answer grading stage is not implemented by this class.

References: [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents) and [streaming](https://docs.langchain.com/oss/python/langchain/streaming). This adapter normalizes the supported `astream_events(version="v2")` callback feed into its own stable application event schema.
