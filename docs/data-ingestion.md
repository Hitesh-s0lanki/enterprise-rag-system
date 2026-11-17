# Text ingestion

This package loads UTF-8 `.txt`, `.md`, and `.markdown` files, splits them with LangChain's `RecursiveCharacterTextSplitter`, generates OpenAI embeddings, and upserts the chunks into Pinecone. It supports the two existing documents under `src/data`.

Pinecone's Documents API supports either a supplied `embedding` dense vector field alongside BM25-enabled `chunk_text`, or a `chunk_text` field with matching integrated OpenAI embeddings and BM25. The CLI detects and validates the mode from the index schema. For integrated mode, it uploads text and metadata without generating duplicate external embeddings. Model, dimensions and cosine metric must match the configured settings; a Llama or mismatched OpenAI integration is rejected. Query-time searches and their fusion are implemented in `src/retrieval`; see [hybrid retrieval](hybrid-retrieval.md).

## Helpers

| Module | Helpers | Responsibility |
| --- | --- | --- |
| `config.py` | `IngestionSettings.from_env` | Validate chunk sizes, overlap, batch size, index, namespace and embedding configuration. |
| `documents.py` | `discover_text_files` | Find supported files recursively; reject unsupported individual files and empty directories. |
| `documents.py` | `load_text_document`, `utc_timestamp` | Load text and attach source identity, editor, timestamps, hashes and allowed reader roles. |
| `documents.py` | `split_document` | Split by paragraphs, lines, words and characters; preserve metadata and add revision/chunk IDs and offsets. |
| `embeddings.py` | `create_embedding_client`, `embed_texts` | Build the OpenAI client; deduplicate identical text within each batch and validate returned vectors. |
| `src/clients/pinecone_client.py` | `build_hybrid_schema` | Declare OpenAI dense vectors and server-managed BM25 keyword indexing. |
| `src/clients/pinecone_client.py` | `create_pinecone_client`, `get_pinecone_index` | Connect, optionally create a missing index, and reject incompatible schemas before embedding. |
| `pipeline.py` | `prepare_documents`, `build_records`, `ingest_documents` | Prepare locally, build records, and upload bounded batches with acknowledgement checks. |

## Run

From the repository root:

```sh
uv sync --locked
uv run python -m src.db.auth init
uv run python -m src.data_ingestion src/data \
  --user bob \
  --updated-by YOUR_USERNAME \
  --allowed-role employee --allowed-role contributor --allowed-role administrator \
  --dry-run
```

`--user` selects the demo actor (`alice`, `bob`, or `mice`). Alice cannot ingest; Bob can ingest documents permitting contributors; Mice can manage knowledge within the assigned namespace. `--updated-by` is optional and defaults to the actor's username. Override it with the actual source editor when someone else last edited the document. If files have different editors or access policies, ingest them separately with their correct metadata. The roles above illustrate making the sample handbook and runbook readable by all three demo roles; assign the appropriate roles for each actual document.

For a live upload, configure `OPENAI_API_KEY` and `PINECONE_API_KEY` in the existing `.env`, then remove `--dry-run`. Add `--create-index` only when creating a new index is intended and use `--user mice` for that administrative action. The command does not change `.env` or upload anything during a dry run.

Optional settings:

| Variable | Default |
| --- | --- |
| `PINECONE_INDEX_NAME` | `enterprise-it-knowledge` |
| `PINECONE_NAMESPACE` | `it-support` |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` |
| `INGESTION_CHUNK_SIZE` / `INGESTION_CHUNK_OVERLAP` | `1000` / `150` characters |
| `INGESTION_BATCH_SIZE` | `64` chunks |

Use an embedding model that supports the configured dimensions. The same model and dimensions must be used by the later query service. Keep separate indexes/namespaces for incompatible embedding models. Existing legacy vector indexes without a BM25 text field are rejected rather than silently treated as hybrid indexes.

Pass `--updated-at 2026-11-16T17:00:00+05:30` to supply the actual source update time. Otherwise the loader uses filesystem modification time and labels that origin explicitly; copying a file can change that value. Ingestion time is recorded separately in UTC. The supplied username is required and is never guessed from filesystem ownership.

## Metadata on every chunk

- Source: `source` (relative path), `file_name`, `file_type`, `document_id`, `file_sha256`.
- Update provenance: `last_updated_by`, `last_updated_at`, `last_updated_at_epoch`, `last_updated_at_source`, `ingested_at`.
- Uploader provenance (added on live upload): `ingested_by_user_id`, `ingested_by`.
- Access: `allowed_roles`.
- Chunk identity: `revision_id`, `chunk_id`, `chunk_index`, `chunk_count`, `chunk_sha256`, `start_index`.
- Embedding provenance: `embedding_model`, `embedding_dimensions`.

The chunk text is stored as `chunk_text`. In external mode its OpenAI vector is supplied as `embedding`; in integrated mode Pinecone generates the vector from `chunk_text` using its configured OpenAI integration. The Documents API stores metadata as document fields alongside these ranking fields, rather than nesting it in a legacy `metadata` dictionary. A separate `OPENAI_API_KEY` is required for external mode; integrated mode uses Pinecone's configured embedding service.

## Update and access boundaries

Identical sources, editor metadata, access roles and processing settings produce the same chunk IDs on repeated ingestion. Use a consistent source root so relative paths remain stable. Upsert retries therefore overwrite the same records rather than duplicating them. Repeated ingestion still incurs embedding calls; there is no persistent embedding cache yet.

Changed content, editor/access metadata or processing settings produce a new revision. Raw `ingest_documents` upserts do not activate revisions, retire old revisions, delete documents, or invalidate Redis. Use the optional [document update workflow](document-update-tools.md) for an existing document enrolled in the authoritative catalog. The retrieval layer must select approved `(document_id, revision_id)` pairs and enforce the caller's allowed roles on both dense and keyword searches. Partial batch failures can leave accepted records behind; retrying is safe for IDs, but raw upserts are not an atomic publication operation.

`allowed_roles` is access metadata. The ingestion service requires an actor ID and checks bitwise write permission, namespace scope and submitted document roles through `src/db/auth` before embedding or uploading. Users, namespace scopes and role permission masks are read from SQLite. Existing-document update tools additionally check stored catalog ACLs; submitted roles alone cannot authorize replacing a document or changing its access policy. The local CLI selects demo identities and is not a user authentication system. See [users and RBAC](auth.md).

## References reviewed

- [Reference repo document processor](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/e2e-project/src/document_ingestion/document_processor.py): loading/splitting separation. Its source-only text cache is not reused because it would require freshness validation for document updates.
- [Reference repo hybrid notebook](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/hybrid-search-strategies/src/dense_sparse.ipynb): dense retrieval combined with BM25. This package prepares equivalent retrieval signals in Pinecone instead of introducing FAISS and a separate local BM25 store.
- [Reference repo Pinecone notebook](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/vector-store-database/src/pinecone-vector-database.ipynb): OpenAI embeddings and Pinecone storage; the local implementation uses the current Documents API.
- [LangChain recursive splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).
- [Pinecone bring your own vectors](https://docs.pinecone.io/guides/search/bring-your-own-vectors) and [hybrid search](https://docs.pinecone.io/guides/search/hybrid-search).
