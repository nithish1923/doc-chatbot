import streamlit as st
import uuid
from utils import process_files
from rag import build_vector_store, create_conversation_chain

st.set_page_config(page_title="Multi-Docs Chatbot", layout="centered")
st.title("📄 Multi-Docs Conversational Chatbot")

# ------------------ Session State ------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = None

# ------------------ Upload Section ------------------
uploaded_files = st.file_uploader(
    "Upload DOCX files",
    type=["docx", "DOCX"],
    accept_multiple_files=True
)

if st.button("✅ OK – Process Documents"):
    if uploaded_files:
        with st.spinner("Indexing documents..."):
            docs = process_files(uploaded_files)
            st.session_state.vectorstore = build_vector_store(docs)
            st.session_state.sessions = {}
            st.session_state.current_session = None
        st.success("Documents indexed successfully!")

st.divider()

# ------------------ Session Controls ------------------
if st.session_state.vectorstore:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("➕ New Chat Session"):
            session_id = str(uuid.uuid4())[:8]
            st.session_state.sessions[session_id] = {
                "chain": create_conversation_chain(st.session_state.vectorstore),
                "chat_history": []
            }
            st.session_state.current_session = session_id

    with col2:
        if st.session_state.sessions:
            st.session_state.current_session = st.selectbox(
                "Switch Session",
                options=list(st.session_state.sessions.keys()),
                index=0
            )

# ------------------ Chat UI ------------------
if st.session_state.current_session:
    session = st.session_state.sessions[st.session_state.current_session]

    user_question = st.chat_input("Ask from documentation")

    if user_question:
        response = session["chain"]({
            "question": user_question,
            "chat_history": session["chat_history"]
        })

        session["chat_history"].append(
            (user_question, response["answer"])
        )

        # Display conversation
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            st.markdown("### ✅ Answer")
            st.markdown(response["answer"])

            st.markdown("### 📌 Evidence")
            for doc in response["source_documents"]:
                st.markdown(
                    f"**File:** `{doc.metadata['source']}`\n\n"
                    f"> {doc.page_content}"
                )
else:
    st.info("Upload documents and start a new chat session.")
