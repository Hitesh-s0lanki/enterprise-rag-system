"""Native starter questions for an empty IT-support conversation."""

import streamlit as st

SUGGESTED_QUESTIONS = (
    "How do I connect to the company VPN from home?",
    "What are the password requirements?",
    "Which MFA methods are approved?",
    "What ticket priority applies to a company-wide outage?",
)


def render_suggestions(*, disabled: bool) -> str | None:
    st.space("large")
    st.header("What can we help you with?")
    st.markdown("Find answers in the company knowledge you have access to, with sources you can check.")
    st.space("small")
    st.caption("START WITH A QUESTION")
    selected = None
    icons = ("vpn_lock", "password", "verified_user", "confirmation_number")
    topics = ("Remote access", "Account security", "Multi-factor authentication", "Service desk")
    for start in (0, 2):
        columns = st.columns(2, gap="small")
        for index in range(start, start + 2):
            with columns[index - start].container(border=True, height="stretch"):
                st.caption(topics[index])
                question = SUGGESTED_QUESTIONS[index]
                if st.button(question, key=f"suggested_question_{index}", disabled=disabled,
                             icon=f":material/{icons[index]}:", width="stretch", type="tertiary"):
                    selected = question
    return selected
