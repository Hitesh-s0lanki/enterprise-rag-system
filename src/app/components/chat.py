"""Stream public agent text and tool lifecycle updates into native chat elements."""

import streamlit as st
from uuid import uuid4

from src.app.components.evidence import render_evidence
from src.app.services.conversation import Turn, history_context
from src.utils.logger import get_logger

logger = get_logger(__name__)


def tool_label(name):
    """Public progress labels; raw tool identifiers remain in diagnostics."""
    return {
        "search_knowledge": "Searching company knowledge",
        "retrieve_documents": "Finding relevant documents",
        "retrieve_and_grade_documents": "Checking document evidence",
        "get_document": "Opening the source document",
        "update_document": "Updating the document",
        "firecrawl_search": "Searching the web",
        "firecrawl_scrape": "Reading the web source",
        "firecrawl_map": "Finding web sources",
    }.get(name, "Working with a connected tool")


def render_activity(turn):
    with st.expander("Agent activity", type="compact"):
        for event in turn.events:
            if event.type in {"tool_started", "tool_completed", "tool_error"}:
                st.text(f"{event.timestamp} · {event.data.get('name', 'Tool')} · "
                        f"{event.type.removeprefix('tool_')}")
            elif event.type == "run_error":
                st.text(f"Request stopped · {event.data.get('code', 'error')}")


def render_turn(turn):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(turn.question)
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        if turn.status == "complete":
            st.markdown(turn.answer)
        else:
            st.error(turn.error or "This response was interrupted. Please try again.")
            if turn.run_id:
                st.caption(f"Reference: {turn.run_id}")
        render_activity(turn)
        render_evidence(turn)


def stream_turn(agent, question, turns):
    turn = Turn(question=question)
    context = history_context(turns, max_characters=max(0, min(6000,
        agent.config.max_input_characters - len(question))))
    turns.append(turn)
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        status = st.status("Working on your question…", expanded=False, type="compact")
        answer = st.empty()
        stream = None
        try:
            stream = agent.stream(question, context=context)
            for event in stream:
                turn.apply(event)
                if event.type == "text_delta":
                    answer.markdown(turn.answer + " ▌")
                elif event.type == "tool_started":
                    status.update(label=tool_label(event.data.get("name")) + "…")
                elif event.type in {"tool_completed", "tool_error"}:
                    status.update(label="Preparing your answer…" if event.type == "tool_completed"
                                  else "Checking the available results…")
            if turn.status == "running":
                turn.fail()
        except Exception:
            turn.run_id = turn.run_id or str(uuid4())
            logger.exception("ui.stream_failed", extra={"run_id": turn.run_id, "stage": "ui_stream"})
            # Provider/configuration exceptions may contain private request details.
            turn.fail()
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    logger.exception("ui.stream_cleanup_failed", extra={"run_id": turn.run_id, "stage": "ui_cleanup"})
                    turn.fail()
            if turn.status == "running":
                turn.fail()
        if turn.status == "complete":
            answer.markdown(turn.answer)
            status.update(label="Response complete", state="complete", expanded=False)
        else:
            answer.empty()
            st.error(turn.error)
            if turn.run_id:
                st.caption(f"Reference: {turn.run_id}")
            status.update(label="Request stopped", state="error", expanded=False)
        render_activity(turn)
        render_evidence(turn)
    return turn
