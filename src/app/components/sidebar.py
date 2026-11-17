"""Session controls and database-derived demo identity."""

import streamlit as st

from src.db.auth import Permission


def render_sidebar(runtime):
    with st.sidebar:
        st.subheader("NovaRetail", icon=":material/hub:")
        st.caption("Your IT knowledge workspace")
        st.button("New conversation", key="new_conversation", on_click=clear_conversation,
                  width="stretch", icon=":material/add:")
        st.space("small")
        users = [user for user in runtime.users.list_users() if user.is_active]
        if not users:
            st.warning("No active users are available. Ask the database administrator to review access.")
            return None
        by_id = {str(user.id): user for user in users}
        if st.session_state.get("actor_selector") not in by_id:
            st.session_state.pop("actor_selector", None)
        selected = st.selectbox("Acting user", list(by_id), key="actor_selector",
            format_func=lambda identifier: by_id[identifier].display_name)
        user = by_id[selected]
        st.caption(f"{user.role.value.title()} · @{user.username}")
        permissions = runtime.users.get_permissions(user.role)
        st.badge("Knowledge access enabled" if permissions & Permission.READ_KNOWLEDGE else "Knowledge access denied",
                 color="blue" if permissions & Permission.READ_KNOWLEDGE else "red",
                 icon=":material/lock:")
        st.space("small")
        st.caption("WORKSPACE")
        st.markdown(":material/folder_open: Company IT knowledge")
        st.caption(f"Namespace · {runtime.knowledge.namespace}")
        with st.expander("About this workspace", icon=":material/info:"):
            st.caption("Answers use permitted company documents. Expand Sources & evidence to review source freshness and document grading.")
            st.caption("The user selector is for this demo; it is not production authentication.")
            st.caption("Agent usage excludes tool-provider usage. Spending totals and independent final-answer grading are not available.")
        return user


def clear_conversation():
    st.session_state["turns"] = []
