"""
Embeds chunks from data/chunks.json using sentence-transformers and builds
a FAISS index. Outputs data/faiss.index and data/chunks_meta.json.
"""

import os

# Force transformers to use only PyTorch, skip TensorFlow entirely.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunks_meta.json"

EMBED_MODEL = "all-MiniLM-L6-v2"


def main():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found. Run ingest.py first.")
        return

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to {INDEX_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()