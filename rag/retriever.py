"""
RAG retriever — loads FAISS index and retrieves top-k chunks.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(_DIR, "zhongli.index")
CHUNKS_PATH = os.path.join(_DIR, "chunks.json")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None
_index = None
_chunks = None


def _load():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            _chunks = json.load(f)


def retrieve(query: str, k: int = 3) -> list[str]:
    """Return top-k relevant chunk texts for the given query."""
    _load()
    emb = _model.encode([query], normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    _, indices = _index.search(emb, k)
    return [_chunks[i]["text"] for i in indices[0] if i < len(_chunks)]
