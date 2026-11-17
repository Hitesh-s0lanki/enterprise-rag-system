# Application Logging and Request Failures

`src/utils/logger.py` configures application-owned Python logging to stderr and a rotating JSONL file. Startup configures logging after `.env` is loaded, and programmatic agent streaming also initializes it. Repeated Streamlit reruns do not add duplicate handlers or modify third-party/root logger configuration.

Default settings in `.env.example`:

```dotenv
LOG_LEVEL=INFO
LOG_FILE=.local/logs/app.jsonl
LOG_MAX_BYTES=5242880
LOG_BACKUP_COUNT=3
```

Set `LOG_FILE` to an empty string for console-only output. If the file cannot be opened, application logging falls back to stderr. The default directory is ignored by Git.

Follow requests in another terminal:

```sh
tail -f .local/logs/app.jsonl
```

Find an individual request using the reference ID displayed beneath a UI error:

```sh
rg 'REFERENCE_ID' .local/logs/app.jsonl*
```

Records include UTC time, severity, module, event name, run ID, execution stage, tool/model identifiers, lifecycle sequence and measured counts where applicable. Failed agent runs include exception type, a redacted/truncated message, stack locations and chained causes. UI consumers continue receiving a safe error message rather than provider internals. Tool failures are logged before the adapters convert them into sanitized tool results.

Memory-enabled knowledge lookups emit `knowledge.lookup_completed`, correlated
by the agent's `run_id`. It records `cache_source`, `cache_status`,
`cache_write_status` when applicable, `pipeline_runs`, `equivalence_checks`, and
current stage timings in milliseconds (`memory_search_latency_ms`,
`memory_fetch_latency_ms`, `equivalence_latency_ms`, `pipeline_latency_ms`,
`memory_write_latency_ms`, and `total_latency_ms`). Only stages actually executed
are included. Cached reports retain their original provider metrics; these log
fields measure the current lookup. Query text and report contents are omitted.

Prompts, retrieved document bodies, tool arguments/results, request headers and local variable dumps are not intentionally logged. Known secret environment values, OpenAI key patterns, bearer credentials and common secret fields are redacted from messages. Keep logs private: exception messages can still contain operational details, usernames or resource paths, and redaction cannot recognize every possible secret format.

## Closed-event-loop failure

The fast failure was reproduced as `RuntimeError: Event loop is closed`. The synchronous streaming bridge creates a runner per request. LangChain's default cached OpenAI HTTP transport could retain connections associated with a previous runner after that loop closed, causing a later request to fail before receiving tokens or executing tools.

The OpenAI factory now supplies fresh owned HTTP clients. Shared agent cleanup closes them while the request's event loop remains alive, then resets the owned model/runtime for the next run. Injected models remain caller-owned. A single agent instance rejects overlapping runs rather than closing another run's client. Separate instances can process independent requests. `stream_usage=True` requests token counts explicitly with the custom transport.

Two consecutive live greeting requests on the same agent completed successfully after the lifecycle fix. Offline regression tests cover loop-bound model reuse, log redaction, handler idempotency, rotating files and failure references in the UI.

`tests/test_openai_transport.py` additionally runs the actual LangChain/OpenAI SDK HTTP streaming stack against a local test server: three new agent instances each complete two runs across separate synchronous event loops. This requires no external provider calls. A browser refresh does not restart the Python process; stop the old server before relaunching when imported client code remains stale.

If the already-running Streamlit process retains old imported modules or environment values, restart it with the project environment:

```sh
uv run --locked streamlit run src/app/main.py
```

No automatic retry of an entire failed agent run is added: repeating a run could repeat a completed tool action.
