# Reuse query results with Mem0

The optional `MemoryFirstQueryService` looks in Mem0 before running retrieval and
document grading. This implements the requested Mem0 reuse path; the previously
planned Redis cache remains unimplemented. Ordinary retrieval commands retain
their existing behavior. Enable reuse explicitly:

```sh
uv run --locked python -m src.retrieval "How do employees connect to the company VPN?" \
  --user alice --revision DOCUMENT_ID:REVISION_ID --grade --memory-cache
```

Use your actual approved document/revision hashes, repeating `--revision` for
additional documents. Set `MEM0_API_KEY` in `.env`. The demo username selector and
caller-supplied revision list remain local maintainer interfaces, not production
authentication or an authoritative publication registry.

## Execution

1. Resolve the user and check current SQLite permissions and namespace access.
2. Search Mem0 within the user's UUID and a context-specific run ID. Query-only
   schema metadata belongs inside `filters.AND`, alongside a `created_at.gte`
   cutoff for the TTL. This excludes report payloads and expired query records
   before `top_k` ranking. The live v3 endpoint returned report records despite
   the top-level `metadata` search option; those records previously crowded out
   reusable queries. The context
   includes namespace, role, approved revision pairs, knowledge/retrieval/grading
   settings, Jev connection settings, and rubric version. CLI context fields are
   non-secret. Direct service callers must include all output-affecting settings
   in `pipeline_config` and bump their version when their pipeline changes.
3. Search with `rerank=False`; the managed reranker adds latency and is not
   needed to verify cache reuse. Reject expired, malformed, wrong-scope, or low-scoring paraphrase candidates. Exact query
   matches bypass the similarity cutoff and need no additional model call. Paraphrases require a query-only Jev
   check that the same information is requested; sharing a topic is insufficient.
4. Fetch the full report from Mem0, verify its digest, scope and revisions, and
   recheck current permissions against every chunk's role metadata. A valid hit
   returns immediately, without creating Pinecone or embedding clients or
   running document grading.
5. On a miss, run the normal retrieval/grading callback once. Store only reports
   with `sufficient_evidence` status and acceptable size. Mem0 search, fetch, or
   write errors do not discard the normal pipeline output. A failed equivalence
   check causes a miss. Pipeline errors and access denials still propagate.

The service does not generate answers yet: it reuses the existing complete
retrieval and grading report. The current request is at the outer `query` field;
the report's `grading.query` remains the original query it was evaluated against.
The cache also reports `matched_query` on a hit, making reuse explicit.

## Mem0 storage

Mem0's normal add operation extracts facts. This cache uses `infer=False` and
`immutable=True` to preserve supplied content. Each successful cache entry uses
two scoped memories: a verbatim JSON report and a query text entry pointing to
the report ID. Only query entries participate in the cache lookup. The split
keeps metadata below the 2,000-character limit observed in the live service and
keeps document text out of the query's search representation. If the report is
stored but the query entry fails, it cannot be reused until a later successful
write; the caller receives `write_status=failed` or `unconfirmed`.

The client now supports verbatim/immutable writes, reranked search with metadata
filters, and fetching one memory by ID. API behavior follows the official
[add](https://docs.mem0.ai/api-reference/memory/add-memories),
[search](https://docs.mem0.ai/api-reference/memory/search-memories), and
[get](https://docs.mem0.ai/api-reference/memory/get-memory) references.

## Output and settings

With `--memory-cache`, output uses this envelope:

```json
{
  "query": "The current request",
  "source": "mem0",
  "result": {"documents": [], "metrics": {}, "grading": {}},
  "cache": {"hit": true, "status": "hit", "matched_query": "The saved request"},
  "execution": {"pipeline_runs": 0, "equivalence_checks": 0, "total_latency_ms": 0}
}
```

The shape above is illustrative. A real hit contains the entire previous result,
including source text, revision metadata, grading decisions and raw evaluations.
Metrics inside `result` describe the original execution; outer `execution` metrics
describe this request. Hits on paraphrases report the query equivalence check
count and probability. This check sends only the two queries to Jev, not documents.
Mem0 performs server-side semantic search; cache lookup disables its managed
reranker. Jev still verifies semantic equivalence for paraphrases. The root agent
is instructed to pass standalone questions verbatim to preserve exact repeats.

| Environment setting | Default | Purpose |
| --- | --- | --- |
| `MEM0_CACHE_SIMILARITY_THRESHOLD` | `0.3` | Minimum returned Mem0 candidate score. |
| `MEM0_CACHE_EQUIVALENCE_THRESHOLD` | `0.85` | Minimum query-only Jev probability for paraphrase reuse. |
| `MEM0_CACHE_TTL_SECONDS` | `900` | Maximum entry age, enforced locally. |
| `MEM0_CACHE_TOP_K` | `5` | Query candidates considered per lookup. |
| `MEM0_CACHE_MAX_PAYLOAD_BYTES` | `256000` | Maximum report size accepted for storage/reuse. |

These thresholds are initial policies checked on a few live examples. Mem0's
hybrid score is not an equivalence probability: the live exact-query score was
0.3611, while a paraphrase scored higher. Jev intent checks also remain probabilistic.
Calibrate both thresholds on representative positive and negative queries;
sharing vocabulary, changing dates or entities, or adding constraints must not
be treated as an automatic hit. No general quality or latency guarantee is made.

## Access and lifecycle limits

The namespace, identity, pipeline configuration, document metadata and revision
approvals must come from trusted application code. Current SQLite user permissions
are checked on every request. Source ACL changes must publish a new document
revision and update the trusted approved-revision map to invalidate old results;
this cache does not query Pinecone to rediscover ACLs on a hit. Direct access to
the Mem0 project must therefore remain restricted. The report digest detects
corruption and is not an authentication signature.

Entries are isolated per user; there is no cross-user sharing. Insufficient-evidence
reports are not cached. Local TTL checks prevent reuse, but do not delete provider
records. Old, expired and orphaned report memories need a future cleanup policy.
Requests arriving simultaneously can create duplicate entries. This path does
not implement conversational history, pronoun resolution, or follow-up-query
rewriting; callers must supply a self-contained query.

## Repeat-query latency verification

On November 17, 2026, live verification used the standalone question
"How do I connect to the company VPN from home?" as Alice across three fresh
root-agent instances, without conversation history:

| Session | Cache source | Knowledge lookup | Entire agent run | Fresh retrieval/grading runs |
| --- | --- | --- | --- | --- |
| 1 | `rag_pipeline`, successfully stored | 8.30 s | 14.24 s | 1 |
| 2 | `mem0` | 1.16 s | 5.73 s | 0 |
| 3 | `mem0` | 1.14 s | 4.39 s | 0 |

The second session used one query-equivalence check; the third matched exactly
and needed none. Both confirmed cross-session reuse. Before the fix, the latest
two application runs took 16.13 s and 15.41 s; their searches took 10.76 s and
10.53 s. A separate reproduction of the old reranked lookup returned five report
records instead of query records, causing a miss after local validation.

The five-second goal is an end-to-end target rather than a deadline or guarantee:
the root agent still makes model calls to choose a tool and generate an answer,
and network/model latency varies. Cold retrieval, grading and synchronous Mem0
writes remain more expensive than a hit. Current stage timings are returned in
`execution` and logged as `knowledge.lookup_completed` without query or source
text.

Offline verification:

```sh
uv run --locked python -m unittest discover -s tests -v
```
