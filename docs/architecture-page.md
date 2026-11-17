# Client, agent and search tools

Open **Architecture** in the app navigation, or `/architecture` directly. The [standalone interactive diagram](diagrams/system.html) also opens in a browser.

The page defaults to **Client and tools**, showing the request/reply cycle and tool boundaries. Switch to **Inside search knowledge** for the [expanded routing diagram](diagrams/knowledge.html), or **Inside document updates** for the RBAC and publication flow.

## Knowledge search routing

The expanded view follows the current `search_knowledge(query, use_memory=True)` implementation:

1. Validate the query, access and approved revisions.
2. With `use_memory=False`, bypass all Mem0 reads and writes. Otherwise search Mem0 within the user's revision/configuration context.
3. Validate cached candidates: scope, freshness, digest, current access and revisions. Exact repeats need no query-equivalence call; paraphrases require Jev equivalence verification.
4. A valid hit returns a cached retrieval/grading report without Pinecone search, query embedding or document grading.
5. A miss, rejected candidate or Mem0 operation failure runs fresh retrieval. Pinecone executes dense and BM25 searches in parallel. External mode uses OpenAI embeddings; integrated mode uses Pinecone embedding.
6. Check returned chunks, merge with weighted reciprocal rank fusion, deduplicate and select the top results.
7. Jev grades every chunk for relevance, coverage and sufficient evidence. Acceptance requires relevance and sufficiency thresholds; coverage participates in ranking. At least one accepted chunk yields `sufficient_evidence`; other outcomes are `insufficient_evidence` or `no_documents`.
8. With memory enabled, write only sufficient, validated reports within the payload limit. Cache write failure does not discard the fresh report. Return source/revision evidence, grading, cache provenance and metrics to the agent.

The tool never generates the final answer or automatically invokes Firecrawl. The agent decides whether to answer, refine the query, bypass memory or call the web tool within its run limits. Retrieval and grading failures return tool errors, not partial evidence. Missing provider configuration can fail before an operation-level fallback is available.

The route is traced through `src/query/pipeline.py`, `src/query/service.py`, `src/query/equivalence.py`, `src/retrieval/service.py`, `src/retrieval/fusion.py`, and `src/grading/service.py`. It excludes application and storage plumbing.

## Client and tools

The page shows this high-level flow:

1. The client sends a request to the agent.
2. The agent calls Firecrawl web search, search knowledge, or both.
3. Each tool returns results to the agent. The agent can refine a query and repeat within its run limits.
4. The agent uses the evidence to send a final reply to the client.

The Firecrawl cycle searches public sources, then Jev grades the evidence inside the tool middleware against the original query. Accepted results return directly to the root agent. Low grades route to the query enhancement agent, which sends an enhanced query back to Firecrawl for another search and grade. At most two enhancements run (three searches total); exhausted weak evidence returns to the root with an insufficient status. There is no direct root-to-enhancement-agent connection. URL-only scrape and map calls remain ungraded.

The knowledge cycle checks reusable reports, retrieves documents on a miss, grades evidence, and returns knowledge results. The overview also shows the document-update tool boundary and its RBAC gate. This is a presentation change; runtime tool registration is unchanged.

Both views show the knowledge yes/no branch explicitly: a previously seen question with a valid Mem0 report returns its saved Jev grades; a miss or rejected candidate runs Pinecone retrieval and fresh Jev document grading, saves eligible sufficient reports in Mem0, then returns. A cache hit does not regrade documents. Insufficient or empty evidence returns without saving, and `use_memory=false` skips all Mem0 reads and writes.

Labeled boundaries separate the client, root orchestration, Firecrawl search workflow, and knowledge-search workflow. Inside Firecrawl, the middleware boundary encloses grading and query enhancement; the root stays outside it. The detailed knowledge view separates request validation, the Mem0 cache route, fresh retrieval/grading, and save/return. These are logical workflow boundaries, not network or deployment boundaries: external providers still run outside the application.

## Document updates and RBAC

Select **Inside document updates** for the detailed flow in `docs/diagrams/updates.html`, generated from `updates.architecture.json`. The overview includes a separate document-update tools boundary.

An explicit edit request leads to a current `READ_KNOWLEDGE | WRITE_KNOWLEDGE`, namespace and catalog reader/writer ACL check. The root reads the full source using `get_document_for_update`, then submits the complete replacement and exact expected revision to `update_document`. A stale revision or in-progress update returns a conflict; identical content returns unchanged without provider calls. An initial denial prevents staging and uploads. Identity and ACLs are application-bound, never model-controlled.

Changed content is staged as a new immutable revision, uploaded and fetched for verification. Access is rechecked before providers and inside staging/publication transactions. Only verified chunks and a successful atomic catalog head switch yield `status=published`. Failed or interrupted operations require checking the current head before retrying; orphan chunks may remain. Subsequent reads follow published heads within approved scope and use a new revision-based Mem0 context. Search indexing may lag publication.

The diagram describes the existing implementation; it does not alter authorization or publish a document. See [document editing tools](document-update-tools.md) for catalog enrollment, administrator policy, failure handling and operational limitations.

`src/app/app_pages/architecture.py` renders the page without creating an application runtime or calling providers. `docs/diagrams/system.architecture.json` and `knowledge.architecture.json` are the editable Archify sources; their corresponding HTML files are generated viewers. The selected diagram and its JSON source can be downloaded from the page.

Generation uses [Archify](https://github.com/tt-a1i/archify) 3.2.0. From the repository root, regenerate with:

```sh
node /path/to/archify/archify/bin/archify.mjs finalize architecture \
  docs/diagrams/system.architecture.json docs/diagrams/system.html \
  --quality showcase --out-dir /tmp/it-copilot-search-review --json

node /path/to/archify/archify/bin/archify.mjs finalize architecture \
  docs/diagrams/knowledge.architecture.json docs/diagrams/knowledge.html \
  --quality showcase --out-dir /tmp/it-copilot-knowledge-review --json

node /path/to/archify/archify/bin/archify.mjs finalize architecture \
  docs/diagrams/updates.architecture.json docs/diagrams/updates.html \
  --quality showcase --out-dir /tmp/it-copilot-updates-review --json
```

Use a fresh review directory for subsequent generations. Automated validation and browser-check receipts accompany the generated diagram. Archify license notices remain in `docs/diagrams`.

Offline verification:

```sh
uv run --locked python -m unittest discover -s tests -p test_architecture_page.py -v
```
