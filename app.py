import streamlit as st
from utils import process_files
from rag import build_vector_store, create_conversation_chain

st.set_page_config(page_title="Document Chatbot", layout="centered")
st.title("📄 DOCX Chatbot")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# ---------------- HELPERS ----------------
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "thanks", "thank you"]

def is_conversation_question(text):
    keywords = [
        "last response",
        "previous response",
        "what did you say",
        "last answer",
        "previous answer"
    ]
    text = text.lower()
    return any(k in text for k in keywords)

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

# ---------------- DISPLAY CHAT ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask a question...")

if user_input:
    # Store user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1️⃣ Greeting
    if is_greeting(user_input):
        response = "Hi 👋 I’m ready! Ask me anything about your uploaded documents."

    # 2️⃣ Conversation-aware questions
    elif is_conversation_question(user_input):
        previous_answers = [
            msg for role, msg in st.session_state.chat_history
            if role == "assistant"
        ]
        if previous_answers:
            response = f"Here is my previous response:\n\n{previous_answers[-1]}"
        else:
            response = "There is no previous response yet."

    # 3️⃣ Document-based QA
    elif st.session_state.conversation:
        with st.spinner("Thinking..."):
            result = st.session_state.conversation({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

            response = result["answer"]

            sources = {
                doc.metadata.get("source", "Unknown")
                for doc in result.get("source_documents", [])
            }

            if sources:
                response += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    # 4️⃣ No documents uploaded
    else:
        response = "Please upload documents and click OK before asking questions."

    # Store assistant response
    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)
