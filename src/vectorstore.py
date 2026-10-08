import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

def add_chunks(chunks):
    # Chroma embeds the text itself (default model: all-MiniLM-L6-v2 via ONNX)
    batch = 100
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )

def search(query, k=4):
    res = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "meta": meta, "distance": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]

def search_diverse(query, k=8, per_doc=4, pool=30, max_gap=0.45):
    res = collection.query(query_texts=[query], n_results=pool)
    items = [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]  # best-first
    best = items[0]["distance"]
    items = [it for it in items if it["distance"] <= best + max_gap]

    picked, counts = [], {}
    # pass 1: best chunk from every relevant PDF, so each one gets a voice
    for it in items:
        src = it["meta"]["source"]
        if src not in counts:
            picked.append(it)
            counts[src] = 1
    # pass 2: fill remaining slots by rank, up to per_doc per PDF
    for it in items:
        if len(picked) >= k:
            break
        src = it["meta"]["source"]
        if it in picked or counts[src] >= per_doc:
            continue
        picked.append(it)
        counts[src] += 1

    # merge chunks that come from the same page into one entry
    merged = {}
    for it in sorted(picked, key=lambda x: x["distance"]):
        key = (it["meta"]["source"], it["meta"]["page"])
        if key in merged:
            merged[key]["text"] += "\n" + it["text"]
        else:
            merged[key] = dict(it)
    return list(merged.values())