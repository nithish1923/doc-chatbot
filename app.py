import streamlit as st
from utils import process_files
from rag import build_vector_store, create_conversation_chain

st.set_page_config(page_title="Doc Chatbot", layout="centered")

st.title("📄 Document Chatbot")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload DOCX files",
    type=["docx"],
    accept_multiple_files=True
)

if uploaded_files and st.button("OK"):
    with st.spinner("Processing documents..."):
        docs = process_files(uploaded_files)
        vectorstore = build_vector_store(docs)
        st.session_state.conversation = create_conversation_chain(vectorstore)
        st.success("Documents processed. Start chatting!")

# ---------------- DISPLAY CHAT HISTORY ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask a question...")

# ---------------- GREETING HANDLER ----------------
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "thanks", "thank you"]
    return text.lower().strip() in greetings

if user_input:
    # Show user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Greeting response
    if is_greeting(user_input):
        response = "Hi 👋 I'm ready! Ask me anything about your uploaded documents."
        st.session_state.chat_history.append(("assistant", response))
        with st.chat_message("assistant"):
            st.markdown(response)

    # Document-based QA
    elif st.session_state.conversation:
        with st.spinner("Thinking..."):
            result = st.session_state.conversation({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

            answer = result["answer"]

            # Source attribution
            sources = set()
            for doc in result.get("source_documents", []):
                sources.add(doc.metadata.get("source", "Unknown"))

            if sources:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

            st.session_state.chat_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)

    else:
        msg = "Please upload documents and click OK before asking questions."
        st.session_state.chat_history.append(("assistant", msg))
        with st.chat_message("assistant"):
            st.markdown(msg)
