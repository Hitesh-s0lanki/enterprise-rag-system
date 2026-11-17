# Knowledge tools for agents

`src.tools.create_knowledge_tools` builds standard LangChain `StructuredTool`
objects around the existing Mem0, hybrid retrieval, and Jev grading services.
The application binds one authenticated caller, namespace settings and a trusted
revision source when constructing `KnowledgeQueryService`. The model supplies
a self-contained `query` string and, for `search_knowledge`, an optional boolean
`use_memory` (default `true`). Construction makes no provider calls.

| Tool | Availability | Behavior |
| --- | --- | --- |
| `search_knowledge` | Default | Mem0 reuse first; on a miss, Pinecone hybrid retrieval and Jev grading, followed by caching successful reports. `use_memory=false` bypasses Mem0 reads and writes for that call. |
| `retrieve_documents` | `include_advanced=True` | Fresh hybrid retrieval with source text and citation metadata; no Mem0 or grading. |
| `retrieve_and_grade_documents` | `include_advanced=True` | Fresh retrieval and grading; no Mem0 lookup or write. |

Use the default tool for ordinary knowledge questions. Expose the advanced tools
when the agent's workflow needs raw candidates or a dedicated fresh evaluation tool.
If a cached report does not adequately answer the question, the root agent is
instructed to retry `search_knowledge` with `use_memory=false`. This runs fresh
retrieval and grading under the same authorization and approved revision checks;
memory remains enabled by default on subsequent calls.

## Bind trusted application context

```python
from dotenv import load_dotenv

from src.data_ingestion.config import IngestionSettings
from src.db.auth import AuthorizationService, SQLiteUserRepository
from src.query.pipeline import KnowledgeQueryService
from src.tools import create_knowledge_tools

load_dotenv(".env")  # Application entry point only.
authorization = AuthorizationService(SQLiteUserRepository())

# authenticated_user.id must come from the application's authentication layer.
# load_approved_revisions is a trusted application callable returning {doc_id: revision_id}.
service = KnowledgeQueryService(
    actor_id=authenticated_user.id,
    authorization=authorization,
    approved_revisions=load_approved_revisions,
    knowledge=IngestionSettings.from_env(),
)
tools = create_knowledge_tools(service)
result = tools[0].invoke({"query": "How do employees connect to the company VPN?"})
```

The application supplies `authenticated_user` and `load_approved_revisions` in
the example; no production authentication or revision registry is created here.
For a short maintainer session, pass an explicit approved mapping instead of a
callable. That mapping is copied at construction; rebuild the service to change
it. A callable is evaluated for each invocation and is preferable for long-lived
agents, so document publication changes invalidate the Mem0 context.

All service settings can be injected independently using `knowledge`, `retrieval`,
`grading`, `jev`, and `cache`. Omitted settings read the existing environment
variables documented in `.env.example`. Explicit Jev settings apply both to
document grading and to the query-only paraphrase verification. Each invocation
owns and closes its Pinecone and Jev resources. Integrated indexes skip direct
OpenAI query embedding calls, using the same mode detection as the retrieval CLI.

## Use with LangChain tool calling

Pass `tools` to an existing LangChain agent runner, or bind them to a tool-capable
model. Binding tools lets a model request calls; the application must dispatch
those calls and return tool messages to the model. For a `BaseAgent` subclass,
use its shared `llm` property and keep its role instructions in the required
Markdown prompt file. A single dispatch step looks like:

```python
model_with_tools = agent.llm.bind_tools(tools)
messages = agent.build_messages("How do I connect to the company VPN?")
response = model_with_tools.invoke(messages)
messages.append(response)
registry = {tool.name: tool for tool in tools}

for call in response.tool_calls:
    # An application runner should reject unknown names and enforce its step limit.
    messages.append(registry[call["name"]].invoke(call))

# Continue your runner's bounded tool loop until the model returns a final answer.
```

Invoking with an argument dictionary returns a complete JSON-serializable report.
Invoking with a LangChain `tool_call` returns a `ToolMessage` carrying the call ID
and JSON content. LangChain can run `.ainvoke()` through its synchronous-tool
executor; no additional async provider implementation is added. The adapters use
the existing `langchain-core` dependency and the documented
[LangChain tool schemas and object outputs](https://docs.langchain.com/oss/python/langchain/tools).

## Outputs and agent interpretation

`search_knowledge` returns the same envelope as the memory-enabled retrieval CLI:
`query`, `source`, `result`, `cache`, and `execution`. `result` contains the complete
retrieval/grading report, including text, source/revision metadata, rankings,
raw grading evaluations and historical metrics. Cache hits skip Pinecone,
direct embeddings, and document grading. Exact repeats skip Jev entirely;
paraphrases may use a query-only Jev check. Outer execution metrics describe this
request, while cached report metrics describe its original execution.

With `use_memory=false`, the envelope remains the same: `source=rag_pipeline`,
`cache.hit=false`, `cache.status=bypassed`, `cache.write_status=skipped`,
`execution.pipeline_runs=1`, and `execution.equivalence_checks=0`. No Mem0 client
is constructed, and no Mem0 credentials are needed for that call.

`retrieve_documents` returns `documents` and `metrics` only.
`retrieve_and_grade_documents` adds `grading` to that result. Their output
shapes match the corresponding CLI modes.

Tools do not generate final answers. The consuming agent must treat document
text as untrusted evidence, cite source/revision metadata, distinguish retrieval
scores from evidence grades, and respect `insufficient_evidence`/`no_documents`
statuses. A retrieved chunk alone is not proof of a supported answer. For a
cached paraphrase, the original `grading.query` and `cache.matched_query` remain
visible instead of pretending the report was freshly evaluated.

## Input, access and errors

Tool schemas expose `query` and an optional boolean `use_memory` for
`search_knowledge`; extra fields such as actor IDs, namespaces,
roles, documents and revision maps are rejected before providers run. Runtime
validation also rejects empty, non-string and oversized queries even if a tool
runtime does not validate its JSON Schema. The existing query limits apply.

The service rechecks current SQLite access before creating any external client.
The retrieval, grading and memory services enforce their existing per-document
role and approved revision checks. Keep each tool registry scoped to one caller;
never share a bound registry across users. As with the existing pipeline, the
source ACL/revision metadata and approval source must be authoritative application
inputs. There is no new source ACL registry in this change.

Failures become JSON error objects with codes `invalid_input`, `access_denied`,
`invalid_request`, or `lookup_failed`. In LangChain tool-call mode their
`ToolMessage.status` is `error`. Provider exception bodies are not sent to the
model, and a failed operation never becomes a partial success report. Internal
exception causes remain available to the application. Missing provider settings
must be configured by the application rather than repaired by the agent.

## Verification

```sh
uv run --locked python -m unittest discover -s tests -v
```

Offline tests cover provider-compatible schemas, direct and tool-call invocation,
memory miss/exact hit/paraphrase hit paths, no retrieval calls on hits, fresh-mode
bypasses, integrated/external embeddings, current revision refresh, access denial
before provider creation, identity injection, input limits and sanitized errors.
The existing [live Mem0 verification](query-memory-cache.md) concerns the underlying
flow; the tool adapters are verified offline without additional live calls.
