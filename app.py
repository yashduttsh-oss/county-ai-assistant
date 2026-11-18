"""Streamlit chat UI for the County AI Assistant.

A thin conversational front-end over rag.py: it takes a question, calls
rag.answer(), and shows the grounded answer plus the source files it cited.

Run:  streamlit run app.py
"""

import streamlit as st

import rag

# --- Page header ------------------------------------------------------------
st.set_page_config(page_title="County AI Assistant")
st.title("County AI Assistant")
st.caption(
    "Answers only from vetted public documents. Holds no personal information. "
    "Every answer cites its sources."
)

# --- Chat history (persisted across reruns via session_state) ---------------
# Each entry is {"role": "user"|"assistant", "content": str, "sources": list}.
if "history" not in st.session_state:
    st.session_state.history = []


def render_message(message):
    """Render one history entry as a chat bubble (+ sources for answers)."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message["content"] == rag.REFUSAL:
            # Make the "not in the documents" case look intentional, not broken.
            st.info(message["content"])
        else:
            st.markdown(message["content"])

        # Show the cited source files beneath assistant answers.
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")


# --- Replay the full conversation on every rerun ----------------------------
for message in st.session_state.history:
    render_message(message)

# --- Handle a new question --------------------------------------------------
question = st.chat_input("Ask a question about the county…")
if question:
    # Record and display the user's message.
    user_message = {"role": "user", "content": question, "sources": []}
    st.session_state.history.append(user_message)
    render_message(user_message)

    # Retrieve + answer, showing a spinner while the work happens.
    with st.spinner("Searching the documents…"):
        answer_text, sources = rag.answer(question)

    # Record and display the assistant's answer (with its sources).
    assistant_message = {
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
    }
    st.session_state.history.append(assistant_message)
    render_message(assistant_message)
