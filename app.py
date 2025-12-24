import streamlit as st
import json
from datetime import datetime
from utils import process_files
from rag import build_vector_store, create_conversation_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="DOCX Chatbot", layout="centered")
st.title("📄 ChatGPT-like Document Chatbot")

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

def chatgpt_reply(user_input, chat_history=None):
    history_text = ""
    if chat_history:
        for role, msg in chat_history:
            if role == "user":
                history_text += f"User: {msg}\n"
            else:
                history_text += f"Assistant: {msg}\n"
    prompt = f"""
You are a friendly, human-like assistant.
Answer naturally and include references to previous conversation like "As I said before..." when relevant.

Conversation history:
{history_text}

Current user message:
{user_input}

Respond naturally, using:
- **Bold** for key points
- *Italics* for hints
- `code blocks` for code answers
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
def display_chat_history(history):
    for role, msg, timestamp in history:
        with st.chat_message(role):
            st.markdown(msg)
            st.caption(timestamp)

# Display existing chat
if current["chat_history"]:
    display_chat_history(current["chat_history"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Type a message...")

if user_input:
    timestamp = datetime.now().strftime("%I:%M %p")
    current["chat_history"].append(("user", user_input, timestamp))
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(timestamp)

    intent, confidence = detect_intent(user_input)

    # Prepare streaming container
    container = st.empty()
    callback = StreamlitCallbackHandler(container)

    # 1️⃣ Small talk
    if intent == "small_talk" and confidence >= 0.6:
        response = chatgpt_reply(user_input, chat_history=current["chat_history"])
        current["chat_history"].append(("assistant", response, timestamp))
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(timestamp)

    # 2️⃣ Conversation memory / previous responses
    elif intent == "conversation_meta" and confidence >= 0.6:
        previous = [msg for role, msg, _ in current["chat_history"] if role=="assistant"]
        response = previous[-1] if previous else "No previous response yet."
        current["chat_history"].append(("assistant", response, timestamp))
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(timestamp)

    # 3️⃣ Document QA
    elif current["conversation"]:
        result = current["conversation"](
            {"question": user_input, "chat_history": current["chat_history"]},
            callbacks=[callback]
        )
        response = result["answer"]

        # Collapsible sources
        sources = []
        for doc in result.get("source_documents", []):
            para_index = doc.metadata.get("paragraph_index", "?")
            src = f"{doc.metadata.get('source','Unknown')} (Paragraph {para_index})"
            sources.append(src)

        if sources:
            response += "\n\n<details><summary>Sources</summary>\n\n" + "\n".join(f"- {s}" for s in sources) + "\n</details>"

        current["chat_history"].append(("assistant", response, timestamp))
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(timestamp)

    # 4️⃣ Fallback
    else:
        response = "I’m not sure what you mean. Please upload documents or rephrase."
        current["chat_history"].append(("assistant", response, timestamp))
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(timestamp)

# ---------------- AUTO-SUMMARIZE LONG HISTORY ----------------
MAX_HISTORY = 20  # Max messages to keep
if len(current["chat_history"]) > MAX_HISTORY:
    summary_prompt = "\n".join([msg for _, msg, _ in current["chat_history"]])
    summary = chat_llm.invoke(f"Summarize the following conversation into a short context for future answers:\n\n{summary}").content
    # Keep only summary + last few messages
    current["chat_history"] = [("assistant", summary, timestamp)] + current["chat_history"][-10:]
