import streamlit as st
import json
from utils import process_files
from rag import build_vector_store, create_conversation_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="DOCX Chatbot", layout="centered")
st.title("📄 Document Chatbot with Multi-Session & Streaming")

# ---------------- SESSION STATE ----------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = "default"

# Initialize default session
if st.session_state.current_session not in st.session_state.sessions:
    st.session_state.sessions[st.session_state.current_session] = {
        "chat_history": [],
        "conversation": None
    }

current = st.session_state.sessions[st.session_state.current_session]

# ---------------- LLMs ----------------
intent_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# ---------------- STREAMING CALLBACK ----------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

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
{{ "intent": "<intent>", "confidence": <number between 0 and 1> }}

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
You are a friendly, human-like assistant.
Respond politely and naturally.

User: {user_input}
Assistant:
"""
    return chat_llm.invoke(prompt).content

# ---------------- SESSION DROPDOWN ----------------
session_options = list(st.session_state.sessions.keys())
session_options.append("New Session")
selected_session = st.selectbox("Select session", session_options)

if selected_session == "New Session":
    new_session_id = f"session_{len(st.session_state.sessions)+1}"
    st.session_state.sessions[new_session_id] = {"chat_history": [], "conversation": None}
    st.session_state.current_session = new_session_id
else:
    st.session_state.current_session = selected_session

current = st.session_state.sessions[st.session_state.current_session]

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
        current["conversation"] = create_conversation_chain(vectorstore)
        st.success("Documents processed. You can start chatting!")

# ---------------- DISPLAY CHAT ----------------
for role, msg in current["chat_history"]:
    with st.chat_message(role):
        st.markdown(msg)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Type a message...")

if user_input:
    current["chat_history"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    intent, confidence = detect_intent(user_input)

    # ---------------- ROUTING ----------------
    # 1️⃣ Small talk
    if intent == "small_talk" and confidence >= 0.6:
        response = chatgpt_reply(user_input)

    # 2️⃣ Conversation memory
    elif intent == "conversation_meta" and confidence >= 0.6:
        previous = [msg for role, msg in current["chat_history"] if role=="assistant"]
        response = f"Here’s my previous response:\n\n{previous[-1]}" if previous else "No previous response yet."

    # 3️⃣ Document QA
    elif current["conversation"]:
        container = st.empty()  # streaming container
        callback = StreamlitCallbackHandler(container)
        result = current["conversation"](
            {"question": user_input, "chat_history": current["chat_history"]},
            callbacks=[callback]
        )
        # Final response (ensure sources + paragraph highlight)
        response = result["answer"]
        sources = []
        for doc in result.get("source_documents", []):
            para_index = doc.metadata.get("paragraph_index", "?")
            src = f"{doc.metadata.get('source','Unknown')} (Paragraph {para_index})"
            sources.append(src)
        if sources:
            response += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    # 4️⃣ Fallback
    else:
        response = "I’m not sure what you mean. Please upload documents or rephrase."

    current["chat_history"].append(("assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)
