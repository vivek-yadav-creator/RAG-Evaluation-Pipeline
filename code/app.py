import streamlit as st
from backend import (
    create_session, list_sessions, load_messages, send_message,
    attach_document_to_session, rename_session, delete_session
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

if "active_session" not in st.session_state:
    sessions = list_sessions()
    st.session_state.active_session = sessions[0]["session_id"] if sessions else create_session()

with st.sidebar:
    st.title("Chats")

    if st.button("+ New chat", use_container_width=True):
        st.session_state.active_session = create_session()
        st.rerun()

    st.divider()

    for s in list_sessions():
        col1, col2 = st.columns([4, 1])
        with col1:
            label = s["title"] + (" 📄" if s.get("has_document") else "")
            if st.button(label, key=f"open_{s['session_id']}", use_container_width=True):
                st.session_state.active_session = s["session_id"]
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{s['session_id']}"):
                delete_session(s["session_id"])
                if st.session_state.active_session == s["session_id"]:
                    remaining = list_sessions()
                    st.session_state.active_session = remaining[0]["session_id"] if remaining else create_session()
                st.rerun()

    st.divider()
    st.subheader("Attach a document")
    st.caption("Required before you can chat — this session is RAG-only.")
    uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded_file is not None and st.button("Attach to this chat"):
        with st.spinner("Indexing document..."):
            try:
                attach_document_to_session(
                st.session_state.active_session, uploaded_file.getvalue(), uploaded_file.name
                )
                st.success(f"Attached: {uploaded_file.name}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        

session_id = st.session_state.active_session
messages = load_messages(session_id)
session_meta = next((s for s in list_sessions() if s["session_id"] == session_id), None)

st.header("RAG Chatbot")

if not session_meta or not session_meta.get("has_document"):
    st.info("Attach a PDF or TXT document in the sidebar to start chatting.")
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something about the document...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = send_message(session_id, user_input)
                st.write(result["answer"])
                st.caption(f"Faithfulness: {result['faithfulness']:.2f} | Retries: {result['retries']}")

        if len(messages) == 0:
            rename_session(session_id, user_input[:40])
            st.rerun()