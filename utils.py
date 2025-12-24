from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_files(files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for file in files:
        doc = Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)

        for chunk in splitter.split_text(text):
            chunks.append({
                "text": chunk,
                "source": file.name
            })

    return chunks
