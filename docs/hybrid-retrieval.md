# Hybrid Document Retrieval

`src/retrieval` combines Pinecone semantic search over OpenAI embeddings with Pinecone BM25 search over `chunk_text`. Both searches use the index schema already created by ingestion. Their ranked results are merged with weighted reciprocal rank fusion (RRF), deduplicated by chunk ID, and returned as LangChain `Document` objects.

The reference repository's [dense/sparse notebook](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/hybrid-search-strategies/src/dense_sparse.ipynb) combines FAISS/HuggingFace dense retrieval with local BM25. This project keeps both signals in Pinecone and continues using OpenAI embeddings. Its [MMR](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/hybrid-search-strategies/src/maximal_marginal_relevance.ipynb) and [reranking](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/hybrid-search-strategies/src/re_ranking.ipynb) examples were reviewed; those additional stages are not included in this retrieval implementation.

## Query flow

1. Resolve the acting user from SQLite and verify read permission and namespace scope.
2. Validate query length and the explicit list of approved document/revision pairs.
3. Use the same OpenAI embedding model and dimensions as ingestion. For an external-vector index, generate the query vector directly; for an integrated index, send an `embed` clause over `chunk_text` so Pinecone generates it. The CLI detects the mode from the validated schema.
4. Search Pinecone dense vectors and BM25 text concurrently, with identical access, revision and embedding-provenance filters.
5. Validate each returned candidate again against current user permissions, approved revisions and embedding provenance.
6. Exclude dense candidates with missing, nonfinite or out-of-range cosine scores, or scores below `RETRIEVAL_MIN_COSINE_SIMILARITY` (default `0.3`, inclusive). This applies to both external-vector and integrated embedding searches.
7. Merge the remaining rankings with weighted RRF, retain the top chunks, and recheck output access before returning them.

The cosine cutoff applies only to the dense branch. BM25 scores use a different scale, so keyword matches remain independently eligible, including chunks rejected by the dense cutoff. A keyword-only chunk may have no measured cosine score in the fetched dense candidates. This is not a minimum cosine guarantee for every final chunk. The cutoff is an initial setting that needs tuning against representative queries; Jev evidence grading remains a separate step.

For each chunk, fusion adds `dense_weight / (rrf_k + dense_rank)` and `(1 - dense_weight) / (rrf_k + keyword_rank)` for the rankings where it appears. Ranks start at 1. Raw BM25 and dense scores are retained for inspection but are not averaged. Duplicate IDs within one ranking contribute once. Ties are resolved by chunk ID for reproducible results.

The starting weights are 0.7 semantic / 0.3 keyword. These are configurable initial values, not a measured optimum for the company handbook. Tune them against representative queries before judging retrieval quality. RRF scores are ranking values, not evidence grades, confidence probabilities or factual correctness checks.

## Modules

| File | Responsibility |
| --- | --- |
| `src/retrieval/config.py` | Validated limits, weights, candidate depth and request timeouts. |
| `src/retrieval/service.py` | Authorization, filters, parallel provider calls, candidate validation and metrics. |
| `src/retrieval/fusion.py` | Weighted RRF and duplicate removal. |
| `src/retrieval/retriever.py` | `HybridRetriever`, a LangChain `BaseRetriever` adapter supporting `.invoke(query)`. |
| `src/retrieval/__main__.py` | Maintainer CLI for retrieving source chunks without generating an answer. |

## Run

Initialize auth and ingest the intended documents first, following [data ingestion](data-ingestion.md). Use the actual accepted `document_revisions` mapping in the ingestion report to identify versions, and approve the intended versions separately. A dry run alone does not upload searchable content.

```sh
uv run --locked python -m src.retrieval "How do I connect to the VPN?" \
  --user alice \
  --revision DOCUMENT_ID:REVISION_ID
```

Replace the placeholders with actual 64-character document/revision hashes. Repeat `--revision` for additional documents. A single request accepts at most 100 document/revision pairs, with one version per document. Omitting the list or supplying invalid/conflicting pairs fails locally before provider calls.

The CLI reads the same `AUTH_DB_PATH`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_EMBEDDING_DIMENSIONS`, and `PINECONE_API_KEY` settings as auth and ingestion. It does not create a missing Pinecone index.

| Additional environment setting | Default | Meaning |
| --- | --- | --- |
| `RETRIEVAL_TOP_K` | `5` | Maximum number of final chunks. |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidates fetched per signal; at least `top_k`, at most 200. |
| `RETRIEVAL_DENSE_WEIGHT` | `0.7` | Semantic weight; strictly between 0 and 1. Keyword weight is the remainder. |
| `RETRIEVAL_MIN_COSINE_SIMILARITY` | `0.3` | Inclusive minimum dense cosine score before fusion; finite value from -1 to 1. Does not filter BM25 matches. |
| `RETRIEVAL_RRF_K` | `60` | Positive RRF smoothing constant. |
| `RETRIEVAL_TIMEOUT_SECONDS` | `30` | Timeout for each Pinecone search request. |
| `RETRIEVAL_MAX_QUERY_CHARACTERS` | `4000` | Input character limit, configurable from 1 to 4000. |

## Access and revision boundaries

Non-administrators search only chunks whose `allowed_roles` contains their exact role; roles do not implicitly inherit document access. Administrators can read documents across their assigned namespace, but remain subject to valid document-role metadata, approved version filters and read permission checks.

Document and revision IDs are filtered as exact pairs, not as separate independent lists. This prevents pairing a document with another document's approved revision. Results with missing/invalid roles, unapproved versions, incompatible embedding provenance or empty text are excluded. Both search branches must complete successfully; provider failures propagate rather than producing an unlabeled single-signal result.

`approved_revisions` is an explicit input from trusted application code. It is not automatically an authoritative catalog: the existing publication/approval workflow remains to be implemented. A production frontend must obtain this mapping from that catalog and must not allow employees to supply arbitrary historical revisions. The CLI is a local maintainer/demo interface. The service never infers that the newest timestamp or any raw upsert is an approved current revision.

## Returned data and metrics

Each returned chunk includes its citation/source metadata, filename, last updater, last-updated time, revision and chunk IDs, uploader metadata when present, fused score and per-signal rank/raw score. Missing source update times are labeled `unknown`. Ingestion timestamps remain separate from source update timestamps.

`HybridRetrievalService.retrieve()` returns a `RetrievalResult` containing documents and metrics. Use `to_dict()` for a JSON representation. `HybridRetriever.invoke()` returns only its documents; use the service when the caller needs timing and usage metrics too.

- `retrieval_latency_ms`: search, validation and fusion time; excludes separately generated query embeddings. In integrated mode the embedding time is part of the dense search.
- `embedding_latency_ms` and individual dense/keyword search latencies. Separate embedding latency is `null` for integrated mode because it cannot be isolated from the search.
- `total_latency_ms`: service time including authorization and embedding.
- One embedding call, two search calls, candidate counts and returned chunk count.
- `min_cosine_similarity`: applied dense cutoff. `dense_candidates_filtered`: otherwise valid dense candidates excluded by score validation or the cutoff. `dense_candidates` counts the survivors.
- `embedding_input_tokens_estimated`: local tokenizer estimate, not a billed provider count. Actual token usage is reported from integrated search when Pinecone supplies it; otherwise it is `null`.
- Reported Pinecone read units summed across both searches; `null` if not reported.
- `estimated_cost` remains `null`; no pricing is invented or assumed.

Returned chunks are retrieval candidates. This layer does not call Jev grading, generate answers, implement Redis caching, or perform LLM reranking. Empty results do not mean an answer exists. Live-provider relevance and latency have not been benchmarked by the offline tests.

## Verification and sources

```sh
uv run --locked python -m unittest discover -s tests -v
```

The retrieval tests use real Pinecone document response types with mocked provider calls and temporary SQLite users. They cover fusion, source metadata, role/namespace denials, exact version pairing, malformed hits, duplicate IDs, invalid embeddings, provider failure and LangChain invocation.

- [Pinecone hybrid search](https://docs.pinecone.io/guides/search/hybrid-search).
- [Pinecone reciprocal rank fusion](https://docs.pinecone.io/guides/search/reciprocal-rank-fusion).
- [Pinecone metadata filtering](https://docs.pinecone.io/guides/search/filter-by-metadata).
