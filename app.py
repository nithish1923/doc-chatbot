import streamlit as st
from utils import process_files
from rag import build_vector_store, create_conversation_chain
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Document Chatbot", layout="centered")
st.title("📄 DOCX Chatbot")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Small-talk LLM (separate from RAG)
chat_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# ---------------- HELPERS ----------------
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey"]

def is_small_talk(text):
    phrases = [
        "how are you",
        "how r you",
        "how's it going",
        "what's up",
        "good morning",
        "good evening",
        "good afternoon",
        "thanks",
        "thank you"
    ]
    text = text.lower()
    return any(p in text for p in phrases)

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

def chatgpt_style_reply(user_input):
    prompt = f"""
You are a friendly, polite, human-like assistant.
Respond naturally and briefly like ChatGPT.

User: {user_input}
Assistant:
"""
    return chat_llm.invoke(prompt).content

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
        st.success("Documents processed. You can start chatting!")

# ---------------- DISPLAY CHAT ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Type a message...")

if user_input:
    # Store user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1️⃣ Greeting
    if is_greeting(user_input):
        response = "Hi 👋 How can I help you today?"

    # 2️⃣ Small talk (ChatGPT-style)
    elif is_small_talk(user_input):
        response = chatgpt_style_reply(user_input)

    # 3️⃣ Conversation-aware questions
    elif is_conversation_question(user_input):
        previous_answers = [
            msg for role, msg in st.session_state.chat_history
            if role == "assistant"
        ]
        if previous_answers:
            response = f"Here’s my previous response:\n\n{previous_answers[-1]}"
        else:
            response = "There isn’t a previous response yet."

    # 4️⃣ Document-based QA (RAG)
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

    # 5️⃣ No docs uploaded
    else:
        response = "Please upload documents and click OK before asking document-related questions."

    # Store assistant response
    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)
