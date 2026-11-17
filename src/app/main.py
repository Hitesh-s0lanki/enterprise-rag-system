"""Launch from the repository root: uv run --locked streamlit run src/app/main.py."""

import sqlite3
import sys
from pathlib import Path

# Streamlit executes this file as a script, placing src/app on sys.path.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from dotenv import load_dotenv

from src.app.components.chat import render_turn, stream_turn
from src.app.components.sidebar import render_sidebar
from src.app.components.suggestions import render_suggestions
from src.app.services.conversation import restore_turns
from src.app.services.runtime import AppRuntime
from src.utils.logger import configure_logging


def render_copilot():
    load_dotenv(override=False)
    configure_logging()
    st.subheader("IT copilot", icon=":material/support_agent:")
    st.caption("NovaRetail · Company knowledge, clear answers.")
    st.session_state.setdefault("turns", [])
    try:
        runtime = AppRuntime.from_env()
    except (OSError, sqlite3.Error):
        st.error("Local database initialization failed. Check that AUTH_DB_PATH points to a writable "
                 "SQLite file location and that the database is accessible.")
        st.stop()
    except ValueError:
        st.error("Application settings are invalid. Review .env.example and your local configuration.")
        st.stop()
    try:
        user = render_sidebar(runtime)
        if user is None:
            st.stop()
        identity = (str(user.id), runtime.knowledge.namespace)
        if st.session_state.get("conversation_identity") != identity:
            st.session_state["turns"] = []
            st.session_state["conversation_identity"] = identity
        runtime.authorize(user)
        revisions = runtime.revisions()
        st.session_state["turns"] = restore_turns(st.session_state["turns"])
        try:
            runtime.check_history(user, st.session_state["turns"])
        except PermissionError:
            st.session_state["turns"] = []
            st.info("Conversation cleared because knowledge access or a source revision changed.")
        if not revisions:
            st.info("No published document revisions are registered for this namespace. "
                    "A maintainer must register verified documents in the catalog before you can ask questions.")
        elif len(revisions) > 100:
            st.warning("This showcase supports up to 100 published documents per namespace. "
                       "A maintainer must narrow the catalog before queries can run.")
        agent = runtime.agent(user)
    except PermissionError:
        st.session_state["turns"] = []
        st.error("This user does not have read access to this knowledge namespace.")
        st.stop()
    except (FileNotFoundError, sqlite3.Error):
        st.error("The local database could not be read. Check that the configured SQLite file is accessible.")
        st.stop()
    except (ValueError, OSError):
        st.error("Application settings are invalid. Review .env.example and your local configuration.")
        st.stop()

    turns = st.session_state["turns"]
    can_query = 1 <= len(revisions) <= 100
    suggested_question = None
    suggestions = st.empty()
    if not turns:
        with suggestions.container():
            suggested_question = render_suggestions(disabled=not can_query)
    for turn in turns:
        render_turn(turn)
    question = st.chat_input("Ask about VPN, passwords, devices, or IT support…", disabled=not can_query,
                             submit_mode="disable",
                             max_chars=min(agent.config.max_input_characters,
                                           agent.service.retrieval.max_query_characters,
                                           agent.service.grading.max_query_characters))
    question = question or suggested_question
    if can_query and question and question.strip():
        suggestions.empty()
        # Bound memory/rendering growth in this session-only showcase.
        del turns[:-19]
        stream_turn(agent, question.strip(), turns)


def main():
    st.set_page_config(page_title="NovaRetail · IT Copilot", page_icon=":material/support_agent:",
                       layout="centered", initial_sidebar_state="expanded")
    page = st.navigation([
        st.Page(render_copilot, title="IT copilot", icon=":material/support_agent:", default=True,
                help="Ask questions about company knowledge and inspect the supporting evidence."),
        st.Page("app_pages/architecture.py", title="Architecture", url_path="architecture",
                icon=":material/account_tree:",
                help="Explore the system design, agent tools, data flow and service boundaries."),
    ], position="top")
    st.set_page_config(layout="wide" if page.title == "Architecture" else "centered")
    page.run()


if __name__ == "__main__":
    main()
