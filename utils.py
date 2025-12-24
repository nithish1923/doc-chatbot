from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_files(files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    for file in files:
        doc = Document(file)
        full_text = "\n".join(p.text for p in doc.paragraphs)

        chunks = splitter.split_text(full_text)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": file.name
            })

    return all_chunks
