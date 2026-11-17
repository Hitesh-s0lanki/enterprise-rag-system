"""High-level client, agent and search-tool flow."""

from pathlib import Path

import streamlit as st


DIAGRAMS = Path(__file__).resolve().parents[3] / "docs" / "diagrams"


def render_architecture():
    st.title("System architecture", icon=":material/account_tree:")
    st.caption("Client → Agent → Search and update tools → Agent → Client · Built with Archify")
    st.write(
        "The client sends a request to the agent. The agent uses Firecrawl web search or "
        "search knowledge, reviews the results, and replies to the client. Firecrawl search results "
        "pass through Jev grading inside the tool middleware, with up to two query enhancements for weak evidence. "
        "Explicit document edits use separate tools guarded by current RBAC, stored document ACLs and revision checks."
    )
    view = st.segmented_control(
        "Diagram", ["Client and tools", "Inside search knowledge", "Inside document updates"],
        default="Client and tools", required=True, key="architecture_view",
    )
    stem = {"Inside search knowledge": "knowledge", "Inside document updates": "updates"}.get(view, "system")
    descriptions = {
        "knowledge": "Knowledge search routing through Mem0, Pinecone and Jev",
        "updates": "Document update flow through RBAC, document ACLs, revision checks, upload verification and atomic publication",
        "system": "Client and root agent flow with Firecrawl and knowledge search, plus authorized document updates",
    }
    st.iframe(DIAGRAMS / f"{stem}.html", height=1100 if stem == "system" else 900,
              width="stretch", alt=descriptions[stem])
    with st.container(horizontal=True):
        st.download_button("Download interactive diagram", (DIAGRAMS / f"{stem}.html").read_bytes(),
                           file_name=f"it-copilot-{stem}.html", mime="text/html",
                           icon=":material/download:", key=f"architecture_html_{stem}")
        st.download_button("Download diagram source", (DIAGRAMS / f"{stem}.architecture.json").read_bytes(),
                           file_name=f"{stem}.architecture.json", mime="application/json",
                           icon=":material/data_object:", key=f"architecture_json_{stem}")

    if stem == "knowledge":
        render_knowledge_routing()
        return
    if stem == "updates":
        render_update_routing()
        return

    with st.container(horizontal=True):
        with st.container(border=True):
            st.subheader("Firecrawl web search", icon=":material/travel_explore:")
            st.markdown(
                "1. The agent sends a web search query.\n"
                "2. Firecrawl searches public sources.\n"
                "3. Jev grades the results inside the tool middleware against the original query.\n"
                "4. **Accepted grade:** results and grades return directly to the root agent.\n"
                "5. **Low grade:** the query enhancement agent rewrites the query, then Firecrawl searches "
                "again and Jev regrades the evidence. At most **two enhancements** (three searches total).\n"
                "6. When retries are exhausted, results and grades return to the root agent, "
                "marked insufficient if the evidence is still weak."
            )
            st.caption(
                "The middleware calls the query enhancement agent; the root has no direct connection to it. "
                "URL-only scrape and map calls remain ungraded."
            )
        with st.container(border=True):
            st.subheader("Search knowledge", icon=":material/manage_search:")
            st.markdown(
                "1. The agent sends a company knowledge query.\n"
                "2. **Previously seen question in Mem0?** Check the match, freshness, access and revisions.\n"
                "3. **Yes:** validate the saved Jev-graded report and return evidence and grades to the root agent.\n"
                "4. **No:** search Pinecone, merge the retrieved documents, then grade them with Jev.\n"
                "5. Save eligible sufficient reports in Mem0, then return the fresh report to the root agent. "
                "Insufficient or empty evidence returns without saving; a failed save does not discard the report."
            )
            st.caption(
                "Mem0 hits reuse saved document grades; they do not regrade documents. "
                "use_memory=false bypasses every Mem0 read and write."
            )
    st.write(
        "The agent can use either tool or both, repeating the search cycle as needed within its run limits. "
        "It combines the available evidence into a final reply to the client."
    )
    with st.container(border=True):
        st.subheader("Document updates and RBAC", icon=":material/edit_document:")
        st.markdown(
            "Explicit edit request → check **READ + WRITE**, namespace and stored document ACLs → "
            "read full source → validate expected revision → upload and verify new chunks → "
            "recheck access and atomically publish → return confirmed status to the root."
        )
        st.caption(
            "Denied access stops the operation. Open Inside document updates for unchanged, conflict and failure routes."
        )


def render_update_routing():
    st.subheader("Document updates and RBAC", icon=":material/edit_document:")
    st.markdown(
        "1. **Explicit edit intent.** Resolve the document from authorized metadata. The root uses "
        "`get_document_for_update` before `update_document`; a search excerpt cannot replace the full source.\n"
        "2. **Check current access.** Both tools require `READ_KNOWLEDGE | WRITE_KNOWLEDGE`, namespace access, "
        "and the catalog's stored reader and writer ACLs. Identity is bound by application code. "
        "Administrator document access still requires the action bits and namespace scope. "
        "An initial denial stops before staging, embedding or Pinecone calls.\n"
        "3. **Validate the replacement.** Submit complete content and the exact `expected_revision` from the read. "
        "A stale revision or in-progress update returns `revision_conflict`; read the latest source and reassess. "
        "Identical content returns `unchanged` without uploads or writes.\n"
        "4. **Stage and upload.** Reserve a new immutable revision transactionally; the old head remains active. "
        "Recheck access before opening providers, then embed and upload new chunks.\n"
        "5. **Verify and publish.** Fetch every chunk and compare text, metadata, revision and uploader identity. "
        "Reauthorize inside the publication transaction, then atomically switch the catalog head.\n"
        "6. **Return and follow the new revision.** Only `status=published` confirms publication. Return document "
        "and revision IDs. Future knowledge reads follow the new approved head and revision-based Mem0 context."
    )
    with st.container(border=True):
        st.markdown("**Failure and access boundaries**")
        st.write(
            "These tools cannot create unregistered documents, change ACLs or edit arbitrary local files. "
            "Upload, verification or publication failure preserves the old head until a commit succeeds, "
            "but may leave orphan chunks. After an interrupted or uncertain result, inspect the current head "
            "before retrying. Pinecone indexing may lag publication; old chunks and Mem0 entries are retained "
            "and excluded by current revision scope."
        )
    st.caption("Source: src/tools/document_updates.py · src/data_ingestion/updates.py · src/db/documents/repository.py")


def render_knowledge_routing():
    st.subheader("Routing inside search_knowledge", icon=":material/manage_search:")
    st.markdown(
        "**Mem0 hit:** previously seen question → validate saved Jev grading → return to the agent.\n\n"
        "**Mem0 miss:** Pinecone search → Jev document grading → save eligible sufficient reports in Mem0 → return."
    )
    st.markdown(
        "1. **Validate the call.** Check the query, current read access, namespace and approved document revisions. "
        "The agent supplies `query` and optional `use_memory`; identity and approvals stay application-bound.\n"
        "2. **Choose the route.** `use_memory=True` is the default. `False` goes straight to fresh retrieval "
        "and skips every Mem0 read and write.\n"
        "3. **Look for reusable evidence.** Mem0 searches within the current user and knowledge context. "
        "Exact repeats skip intent verification; paraphrases must pass a Jev query-equivalence check. "
        "Freshness, similarity where applicable, report digest, permissions and revisions must also pass.\n"
        "4. **Return a valid hit.** Return the saved retrieval and grading report to the agent. "
        "Pinecone, query embedding and document grading are skipped. The agent still generates the final answer.\n"
        "5. **Retrieve on a miss or bypass.** Pinecone runs dense semantic search and BM25 keyword search "
        "in parallel. External embedding mode uses OpenAI; an integrated index embeds the query itself. "
        "Searches filter by access and approved revisions, then returned chunks are checked again.\n"
        "6. **Merge and grade.** Weighted reciprocal rank fusion deduplicates chunks and selects the top results. "
        "Jev grades each chunk for relevance, coverage and whether that chunk alone can answer the entire query. "
        "Acceptance uses relevance and sufficient-evidence thresholds; coverage helps rank the results.\n"
        "7. **Return the report.** If any chunk is accepted, the status is `sufficient_evidence`; "
        "otherwise it is `insufficient_evidence`, or `no_documents` when retrieval is empty. "
        "With memory enabled, only sufficient, validated reports within the size limit are eligible for a Mem0 write."
    )
    with st.container(border=True):
        st.markdown("**What comes back to the agent**")
        st.write(
            "Documents and source/revision metadata, per-chunk grading, report status, cache source, "
            "and current execution metrics. A cache hit preserves the original report's grading and retrieval metrics; "
            "the outer execution metrics describe this call. The tool returns evidence, not a generated answer."
        )
    with st.expander("Failure and retry routes"):
        st.markdown(
            "- A Mem0 search/fetch error or failed paraphrase verifier causes fresh retrieval. "
            "A failed cache write preserves the fresh report. Missing provider configuration can still fail the call.\n"
            "- Dense and keyword search must both succeed. Retrieval or Jev failures return a sanitized tool error; "
            "document grading never silently returns a partial report.\n"
            "- Access denial or invalid input stops the call. Insufficient or empty evidence returns to the agent "
            "without caching. The agent can refine the query, bypass memory on a later call, or use Firecrawl "
            "within its run limits. These are agent decisions; the knowledge tool does not automatically search the web."
        )
    st.caption(
        "Source: src/query/pipeline.py · src/query/service.py · src/query/equivalence.py · "
        "src/retrieval/service.py · src/retrieval/fusion.py · src/grading/service.py"
    )


if __name__ == "__main__":
    render_architecture()
