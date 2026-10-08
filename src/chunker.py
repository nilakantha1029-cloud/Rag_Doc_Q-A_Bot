import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def is_reference_chunk(text):
    urls = text.lower().count("http")
    refs = len(re.findall(r"\[\d{1,2}\]", text))
    years = len(re.findall(r"(?:19|20)\d\d", text))
    return urls >= 2 or (refs >= 3 and years >= 2)

def chunk_docs(docs, size=700, overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = []
    for d in docs:
        for i, text in enumerate(splitter.split_text(d["text"])):
            if is_reference_chunk(text):
                continue                      # skip bibliography / endnote chunks
            chunks.append({
                "id": f'{d["source"]}-p{d["page"]}-c{i}',
                "text": text,
                "source": d["source"],
                "page": d["page"],
            })
    return chunks