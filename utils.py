from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

def process_files(files):
    documents = []

    for file in files:
        text = read_docx(file)
        chunks = chunk_text(text)

        for chunk in chunks:
            documents.append({
                "text": chunk,
                "source": file.name
            })

    return documents
