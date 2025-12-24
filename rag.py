from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
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

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.25
    )

    system_prompt = """
You are a helpful documentation assistant.

Guidelines:
- Answer in a natural, human, conversational tone.
- Be clear and concise.
- Explain like you are talking to a developer or analyst.
- Use ONLY the provided document context.
- If something is not present in the documents, say so clearly.
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
