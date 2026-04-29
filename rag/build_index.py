"""
Build FAISS index from zhongli_rag.json.
Run once on the server: python rag/build_index.py

Output:
  rag/zhongli.index  — FAISS flat L2 index
  rag/chunks.json    — list of chunk dicts {title, text}
"""

import json
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

RAG_JSON = os.path.join(os.path.dirname(__file__), "..", "zhongli_rag.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "zhongli.index")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.json")

CHUNK_MAX = 400   # max chars per chunk
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def split_sentences(text):
    """Split on Chinese/English sentence-ending punctuation, keeping the delimiter."""
    return re.split(r'(?<=[。！？…\.\!\?])\s*', text.strip())


def split_text(text):
    """Group sentences into chunks not exceeding CHUNK_MAX chars."""
    sentences = [s for s in split_sentences(text) if s]
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) <= CHUNK_MAX:
            current += sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


def main():
    with open(RAG_JSON, encoding="utf-8") as f:
        entries = json.load(f)

    chunks = []
    for entry in entries:
        title = entry["title"]
        content = entry["content"]
        if len(content) <= CHUNK_MAX:
            chunks.append({"title": title, "text": f"【{title}】\n{content}"})
        else:
            for part in split_text(content):
                chunks.append({"title": title, "text": f"【{title}】\n{part}"})

    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine sim after normalization
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
