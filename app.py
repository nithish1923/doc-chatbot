import streamlit as st
import json
from utils import process_files
from rag import build_vector_store, create_conversation_chain
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="DOCX Chatbot", layout="centered")
st.title("📄 Document Chatbot")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# ---------------- LLMs ----------------
intent_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

chat_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# ---------------- INTENT DETECTION ----------------
def detect_intent(user_input):
    prompt = f"""
Classify the user message into one intent and confidence.

Allowed intents:
- small_talk
- conversation_meta
- document_question
- unknown

Return JSON only:
{{
  "intent": "<intent>",
  "confidence": <number between 0 and 1>
}}

User message:
"{user_input}"
"""
    raw = intent_llm.invoke(prompt).content

    try:
        data = json.loads(raw)
        return data["intent"], float(data["confidence"])
    except Exception:
        return "unknown", 0.0


def chatgpt_reply(user_input):
    prompt = f"""
You are a friendly, natural assistant like ChatGPT.
Respond politely and briefly.

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
        chunks = process_files(uploaded_files)
        vectorstore = build_vector_store(chunks)
        st.session_state.conversation = create_conversation_chain(vectorstore)
        st.success("Documents processed. Start chatting!")

# ---------------- DISPLAY CHAT ----------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Type a message...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    intent, confidence = detect_intent(user_input)

    # Uncomment for debugging
    # st.caption(f"DEBUG → intent={intent}, confidence={confidence}")

    # 1️⃣ Small talk
    if intent == "small_talk" and confidence >= 0.6:
        response = chatgpt_reply(user_input)

    # 2️⃣ Conversation memory
    elif intent == "conversation_meta" and confidence >= 0.6:
        previous = [
            msg for role, msg in st.session_state.chat_history
            if role == "assistant"
        ]
        response = (
            f"Here’s my previous response:\n\n{previous[-1]}"
            if previous else
            "There isn’t a previous response yet."
        )

    # 3️⃣ Document QA (default fallback)
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

    else:
        response = "Please upload documents and click OK first."

    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)
