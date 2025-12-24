import tempfile
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
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
    persist_dir = tempfile.mkdtemp()

    return Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful documentation assistant.

Rules:
- Answer only from the provided documents.
- Be clear and professional.
- If information is not present, say so.

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
        combine_docs_chain_kwargs={"prompt": prompt}
    )
