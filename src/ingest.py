from collections import Counter

from loader import load_pdfs
from chunker import chunk_docs
from vectorstore import add_chunks, collection

print("STEP 1: Loading PDFs from data/ ...")
docs = load_pdfs()
for name, n in Counter(d["source"] for d in docs).items():
    print(f"   loaded {name}: {n} pages")
print(f"   total: {len(docs)} pages (headers, footers and page numbers removed)\n")

print("STEP 2: Chunking (700 characters, 150 overlap, reference lists skipped) ...")
chunks = chunk_docs(docs)
for name, n in Counter(c["source"] for c in chunks).items():
    print(f"   {name}: {n} chunks")
print(f"   total: {len(chunks)} chunks")
sample = chunks[len(chunks) // 2]
preview = sample["text"][:150].replace("\n", " ")
print(f"   example chunk {sample['id']}:")
print(f'   "{preview}..."\n')

print("STEP 3: Embedding (all-MiniLM-L6-v2, batches of 100) and storing in ChromaDB ...")
add_chunks(chunks)
print(f"   stored {collection.count()} vectors in chroma_db/")

print("\nDone.")