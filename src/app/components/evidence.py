"""Source provenance, evidence grades and measured execution details."""

import streamlit as st


def render_evidence(turn):
    if turn.metrics:
        duration = turn.metrics.get("duration_ms")
        elapsed = f"{duration / 1000:.2f} s" if duration is not None else "Time unavailable"
        usage = turn.metrics.get("usage", {})
        st.caption(f"{elapsed} · {turn.metrics.get('tool_calls', 'Unknown')} tool calls · "
                   f"{usage.get('total_tokens', 'Unknown')} agent tokens")
        with st.expander("Usage details", type="compact"):
            st.caption("Agent token counts exclude tool providers. " +
                       ("Some model usage is unavailable. " if usage.get("complete") is False else "") +
                       "Cost is not measured.")
    if not turn.reports:
        return
    with st.expander("Sources & evidence", expanded=False, icon=":material/library_books:"):
        for index, report in enumerate(turn.reports, 1):
            cached = report.get("cache_source") == "mem0"
            st.write(f"**Knowledge lookup {index}** · {'Reused memory' if cached else 'Fresh retrieval'}")
            latency = report.get("metrics", {}).get("retrieval_latency_ms")
            st.caption("Retrieval latency: " + (f"{latency:.0f} ms" if latency is not None else "unknown") +
                       (" · original cached execution" if cached else ""))
            grading = report.get("grading", {})
            summary = grading.get("summary", {})
            st.write("Evidence assessment: " + summary.get("status", "unavailable").replace("_", " "))
            if summary:
                st.caption(f"{summary.get('accepted_chunks', 0)} of {summary.get('evaluated_chunks', 0)} chunks accepted")
            st.caption("This assesses document evidence; the final answer has not been independently graded.")
            graded = {row.get("chunk_id"): row for row in grading.get("documents", [])}
            for document in report.get("documents", []):
                metadata = document.get("metadata", {})
                with st.container(border=True):
                    st.text(metadata.get("file_name", "Unknown source"))
                    st.caption(f"Internal company source · Updated: {metadata.get('last_updated_at') or 'unknown'}")
                    st.text("Source: " + str(metadata.get("source", "unknown")))
                    st.caption(f"Last editor: {metadata.get('last_updated_by', 'unknown')} · "
                               f"Uploader: {metadata.get('ingested_by', 'unknown')}")
                    st.caption(f"Revision: {metadata.get('revision_id', 'unknown')} · "
                               f"Chunk: {document.get('chunk_id', 'unknown')}")
                    assessment = graded.get(document.get("chunk_id"), {})
                    if assessment:
                        st.caption("Evidence decision: " + assessment.get("decision", "unknown").replace("_", " "))
                    st.text(document.get("text", ""))
            with st.expander(f"Lookup {index} metrics and assessment"):
                st.json({"retrieval": report.get("metrics", {}), "current_execution": report.get("execution", {}),
                         "grading_summary": summary, "grading_settings": grading.get("settings", {}),
                         "grading_usage": grading.get("metrics", {})})
