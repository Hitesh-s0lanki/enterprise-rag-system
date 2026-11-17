# Authorized document editing tools

`create_document_update_tools` provides two LangChain tools:

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `get_document_for_update` | `document_id` | Read the full current source and revision after checking edit access. |
| `update_document` | `document_id`, `expected_revision`, `content` | Replace the full source, upload new immutable chunks, verify them, and publish the new revision. |

The application binds the authenticated actor, namespace and catalog. The agent
cannot submit a username, role, ACL, namespace, uploader identity or source file
path. These tools never edit arbitrary local files, change access policy, create
unregistered documents, or overwrite an existing revision's chunks.

## RBAC and document ACL

Every operation loads the current SQLite user and role permission mask. Editing
requires both `READ_KNOWLEDGE` and `WRITE_KNOWLEDGE`, namespace access, and access
under the document catalog's stored reader and writer roles. The existing
administrator policy permits administrators to access documents throughout their
assigned namespace; action bits and namespace scope still apply. A document's
Pinecone role metadata or a caller-written role assertion is not the write ACL.

Checks run before staging or provider calls, inside staging and publication
transactions, and again before opening upload providers. A user who loses access
after upload cannot publish the replacement. Denied initial edits call neither
embeddings nor Pinecone and create no revision record.

## Explicit catalog setup

The Streamlit app restores the two explicitly approved demo revisions from its
trusted bootstrap manifest after verifying their Pinecone chunks; see
[startup recovery](streamlit-app.md). Existing catalog heads and ACLs are preserved.
Other existing Pinecone documents are not automatically registered as editable. Their
full source, intended published revision and reader/writer ACL must be approved
and enrolled by trusted maintainer code. This is necessary because the prior
ingestion flow had no authoritative source catalog or edit ACL registry.

Initialize the additional catalog tables explicitly in the existing auth DB:

```python
from src.db.documents import initialize_document_catalog, SQLiteDocumentRepository

initialize_document_catalog()  # Uses AUTH_DB_PATH; auth initialization must already exist.
catalog = SQLiteDocumentRepository()
```

Initialization adds tables and preserves all existing users, policies, documents
and revisions. It does not register or upload any document. Do not point the
catalog and auth repositories at different SQLite files: the local update service
requires the same file, so `BEGIN IMMEDIATE` also prevents concurrent permission
writes during its authorization/publication transactions.

Trusted bootstrap code uses a service bound to an authenticated maintainer with
`MANAGE_KNOWLEDGE | WRITE_KNOWLEDGE` in the target namespace:

```python
from src.data_ingestion.config import IngestionSettings
from src.data_ingestion.updates import DocumentUpdateService

maintainer_service = DocumentUpdateService(
    actor_id=authenticated_maintainer.id,
    authorization=authorization,
    catalog=catalog,
    settings=IngestionSettings.from_env(),
)
maintainer_service.register_existing_document(
    document_id=verified_document_id,
    revision_id=verified_current_revision,
    content=verified_full_source_text,
    metadata=verified_source_metadata,
    reader_roles=approved_reader_roles,
    writer_roles=approved_writer_roles,
)
```

The application supplies the verified values above. Metadata must include
`document_id`, `source`, `file_name`, `file_type`, `last_updated_by`, and
`last_updated_at`. Supported types are UTF-8 text and Markdown. Verify these values
against the actual accepted ingestion version before registration; this bootstrap
method records that trusted approval and does not independently validate an
existing Pinecone upload. Re-registering an existing document fails rather than
resetting its ACL or published revision. Registration is not an agent tool.

## Wire the tools into the agent

```python
from src.tools import create_document_update_tools, create_root_tools

updates = DocumentUpdateService(
    actor_id=authenticated_user.id, authorization=authorization, catalog=catalog,
    settings=knowledge_service.knowledge,
)
edit_tools = create_document_update_tools(updates)
root_tools = create_root_tools(knowledge_service, document_update_service=updates)

current = edit_tools[0].invoke({"document_id": target_document_id})
# Only proceed when the read succeeded and the user requested this replacement.
result = edit_tools[1].invoke({
    "document_id": target_document_id,
    "expected_revision": current["revision_id"],
    "content": complete_replacement_text,
})
```

`RootAgent(knowledge_service)` now registers these tools automatically for the local
SQLite auth adapter, using the same caller, settings, authorization and database.
An application can supply `document_update_service=updates` explicitly or use
`include_document_updates=False` for a read-only root agent. Do not also pass the
same edit tools through `extra_tools`, since tool names must be unique. For other
auth/storage adapters, supply the bound update service explicitly.

The standalone `create_knowledge_tools` registry remains read-only. Service RBAC is enforced for every
invocation, including calls made by the root agent. Missing records or an uninitialized
catalog fail closed; root construction does not initialize tables or enroll records.
Replacement content must be nonempty
and at most 100,000 characters by default; applications can configure
`max_content_characters` when constructing the service.

## Publication and cache invalidation

1. Read the authoritative source and check the supplied expected revision. A stale
   revision or an existing in-progress update fails with `revision_conflict`.
2. Prepare a new full source and chunks in memory. Preserve source identity and
   filename, update source hashes/timestamps, and record the authenticated editor.
   Ingestion separately records the uploader UUID and username.
3. Reserve the update transactionally. The old head remains published while the
   new immutable chunk IDs are uploaded. A competing update cannot reserve the
   same document simultaneously.
4. Require all upsert acknowledgements, then fetch every new chunk and compare
   its text, metadata, revision identity and uploader UUID. Missing or mismatched
   data prevents publication.
5. Recheck current access and atomically move the catalog's current revision to
   the new head. Preserve earlier revisions and the staged/published/failed audit
   records. Return `status=published` only after that transaction succeeds.

Bind retrieval's approved revision callable to the catalog so future reads and
Mem0 cache contexts follow the new head:

```python
knowledge_service = KnowledgeQueryService(
    actor_id=authenticated_user.id,
    authorization=authorization,
    knowledge=knowledge_settings,
    approved_revisions=lambda: catalog.approved_revisions(knowledge_settings.namespace),
)
```

The root registry also follows catalog-published heads for documents already in
the knowledge service's approved set, so a prior fixed revision map cannot keep
using the pre-edit cache entry. It does not expand the caller's approved document
set. A callable remains preferable for adding/removing approvals over time.
With the callable,
publication changes the Mem0 context hash, so old cached results cannot be reused
for that new revision set. Existing Mem0 entries and old Pinecone chunks remain
stored but are excluded by current approvals; this change does not delete them.

## Output, failures and limits

Successful replacements return `published`, old/new revision IDs, document ID,
verified chunk count, update ID and acting editor identity. Unchanged content
returns `unchanged` without writes or provider calls, after checking access.
Tool failures return JSON errors with `invalid_input`, `access_denied`,
`revision_conflict`, or `update_failed`; LangChain tool-call messages have error
status. Provider exceptions and partial content are not exposed to the model.

The SQLite catalog stores the current full source; this flow does not write back
to the original local ingestion file. Pinecone fetch visibility is checked once;
indexing delays can therefore fail an update conservatively. Even after a fetch
succeeds, search indexing may lag publication briefly.

SQLite and Pinecone do not share a distributed transaction. An upload failure or
revoked publication leaves the old head active and may leave orphan chunks in
Pinecone, excluded by revision filters. A process crash may leave a staged
reservation; a trusted operator must inspect the audit state and abandon it via
the repository only after confirming the uploader is no longer running. Automatic
recovery, provider cleanup and an ACL-management tool are not implemented.

Repository methods are storage primitives; adapters replacing SQLite must retain
the transaction and authorization behavior. The local document tables reference
the existing auth users and do not hardcode role permission masks.

Verification uses temporary databases and offline provider doubles:

```sh
uv run --locked python -m unittest discover -s tests -v
```
