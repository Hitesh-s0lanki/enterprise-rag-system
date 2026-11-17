# Streamlit IT support showcase

Start the project with one command from the repository root:

```sh
uv run --locked streamlit run src/app/main.py
```

The entry point calls `main()`, which presents top navigation for IT copilot and Architecture. The copilot page loads `.env` without overriding existing environment variables. Configuration uses the existing variables documented in `.env.example`; there are no additional app settings. Live queries use the configured OpenAI, Pinecone, Jev, and Mem0 services and may incur charges. Opening the UI does not call these providers.

The **Architecture** page at `/architecture` shows a high-level Archify diagram: client request → agent → Firecrawl web search or search knowledge → tool results back to the agent → final reply to the client. The default **Inside search knowledge** view expands cache validation and bypass, Mem0 hits and misses, parallel Pinecone retrieval, ranking fusion, Jev grading and conditional report caching. **Client and tools** retains the high-level overview. HTML and editable JSON downloads are available. The page does not construct `AppRuntime`, read `.env`, initialize SQLite, or create provider clients. See [architecture-page.md](architecture-page.md) for regeneration instructions.

The SQLite user selector is explicitly a local demo identity selector. Anyone who can reach this app can select its demo users. Use it only in a trusted showcase environment; production requires authenticated identity binding before deployment.

Startup automatically creates the configured SQLite database and document catalog, including missing parent directories, and seeds demo users once. Later starts and Streamlit reruns preserve edited, deleted, or deactivated users, namespace scopes, permission changes, and document revision history. No separate initialization commands are required. The database location must be writable; corrupted or inaccessible databases produce a startup error rather than being reset.

With `CATALOG_BOOTSTRAP_ENABLED=true` (default), startup restores missing demo document registrations from the trusted `src/db/documents/bootstrap.json` manifest. It includes the two previously approved demo sources, their exact revisions and original source metadata, explicit reader/writer ACLs, ingestion settings, and the maintainer UUID. For its bound index and namespace, startup checks the maintainer's current database management/write permissions, reproduces the approved chunks, and fetches every expected chunk from Pinecone. Text, revision/chunk metadata, reader roles, and uploader provenance must match before registration. Missing or different data produces a sanitized startup error and keeps chat unavailable. This makes provider read requests only when registrations are missing; it never embeds, uploads, scans Pinecone for approvals, or resets an existing catalog head. Concurrent startup registrations preserve the winning head. Set `CATALOG_BOOTSTRAP_ENABLED=false` to disable recovery, or `CATALOG_BOOTSTRAP_PATH` to a different trusted manifest. A different index or namespace is not approved by the bundled manifest.

## Published knowledge

The application reads approved document/revision pairs from `SQLiteDocumentRepository` in the auth database, using a callable to read current heads for each tool invocation. It never accepts approved revisions or document ACLs from a user or the model. An empty catalog disables chat. The retrieval service currently supports at most 100 approved document heads per namespace; the UI disables queries above that limit.

The bundled demo revisions are registered by the verified startup recovery described above. Other documents require a trusted maintainer to ingest and register each verified existing revision through `DocumentUpdateService.register_existing_document`, with its full original content, metadata, and explicit reader/writer roles. See `src/data_ingestion/updates.py` for that API. Registration requires database-derived management and write permissions. The registration API itself does not upload documents or verify provider contents for the maintainer. A raw ingestion upsert alone does not publish a catalog head. The app does not automatically register arbitrary files from `src/data` or infer approvals from timestamps.

## Components

- An empty conversation shows four clickable starter questions covering VPN access, password requirements, MFA and outage ticket priority. Selecting one submits it through the normal chat/streaming flow. Suggestions disappear when the first question is submitted and return for a new conversation. They are disabled whenever chat is unavailable because of the published-document limit.
- `src/app/main.py`: startup, setup states, session orchestration, and `main()`.
- `src/app/app_pages/architecture.py`: offline architecture reference and interactive diagram downloads.
- `src/app/components/suggestions.py`: native suggestion buttons and their question text.
- `src/app/components/sidebar.py`: demo identity, new conversation, and capability information.
- `src/app/components/chat.py`: native user/assistant messages, live text deltas, and tool lifecycle status.
- `src/app/components/evidence.py`: supporting chunks, revisions, editors/uploaders, source update times, evidence assessment, retrieval latency, and provider-reported metrics.
- `src/app/services/runtime.py`: caller-bound `RootAgent` and `KnowledgeQueryService`, SQLite catalog access, and authorization checks.
- `src/app/services/conversation.py`: event reduction and bounded follow-up context.

History is session-local, limited to 20 exchanges, and cleared when the selected user or namespace changes. Up to 6,000 characters of complete recent exchanges are provided as untrusted context within the configured agent input limit. History is not new evidence: the agent must retrieve current supporting knowledge for factual IT answers. Failed and incomplete turns are excluded from follow-up context. Before history is displayed or reused, the app rechecks current read access, source role access, and revision heads. Changes clear the conversation.

On each rerun, saved turns are rebuilt with the current `Turn` schema before history authorization. This supports older objects retained in Streamlit session state after a code reload, including turns created before `run_id` existed. Missing run references are recovered from stored events when possible; abandoned running responses become failed turns with partial answers removed. Source evidence is preserved for the normal access checks.

Text is rendered as events arrive. On completion, the authoritative final answer replaces intermediate text; failed streams discard partial answers. A compact status line shows public progress labels while tools run. Collapsed agent activity preserves the detailed lifecycle events both immediately after a response and on reruns. The UI does not display hidden reasoning or raw provider errors.

The native theme in `.streamlit/config.toml` uses a blue accent, slate text, smaller headings, and a light workspace sidebar. Starter questions appear in a two-column card layout. Response timing, tool calls, and agent tokens share a compact caption; usage limitations and sources remain expandable. Chat input is disabled during submission to avoid interrupting an active response.

## Current limits

The default root registry exposes `search_knowledge` and, with the SQLite adapter, document read/update tools. Firecrawl search/scrape/map tools are included when configured. Service permissions still control every invocation. Dedicated document upload/edit forms and administration screens are not implemented; advanced fresh-retrieval tools require explicit opt-in.

Jev results assess retrieved document evidence, not the final generated answer. Agent token counts exclude embedding, Jev, and other tool-provider usage; reported tool usage remains available in lookup details. Cached retrieval/grading metrics are labeled as the original execution. Monetary totals, aggregate spending, and final-answer grading remain unavailable. Unknown source update times are labeled unknown.

Offline verification:

```sh
uv run --locked python -m unittest discover -s tests -v
```

UI integration tests use Streamlit `AppTest`, temporary SQLite databases, and fake agent streams. No provider calls are required.
