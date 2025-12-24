from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


def build_vector_store(documents):
    docs = [
        Document(
            page_content=d["text"],
            metadata={"source": d["source"]}
        )
        for d in documents
    ]

    embeddings = OpenAIEmbeddings()

    return FAISS.from_documents(docs, embeddings)


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.25
    )

    system_prompt = """
You are a helpful documentation assistant.

Rules:
- Speak naturally and clearly.
- Answer ONLY from the provided documents.
- If information is missing, say so.
"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt + """

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
