"""
Core RAG logic: embed query -> retrieve top-k chunks from FAISS ->
send chunks + question to Mistral-7B-Instruct via HF Inference API ->
return grounded answer with sources.
"""

import os

# Force transformers to use only PyTorch, skip TensorFlow entirely.
# Avoids a Keras 3 / transformers incompatibility crash on machines that
# happen to have TensorFlow installed. Must be set before sentence_transformers
# (which pulls in transformers) is imported.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunks_meta.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_embed_model = None
_index = None
_meta = None
_client = None


def _load():
    global _embed_model, _index, _meta, _client
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _meta is None:
        with open(META_PATH) as f:
            _meta = json.load(f)
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN environment variable not set.")
        _client = InferenceClient(model=LLM_MODEL, token=token)


def retrieve(query, k=5):
    _load()
    q_emb = _embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = _index.search(q_emb, k)
    results = [_meta[i] for i in indices[0] if i != -1]
    return results


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c['title']} ({c['arxiv_id']}), section: {c.get('section', 'N/A')}]\n{c['text']}"
        for c in chunks
    )
    prompt = (
        "You are a research assistant answering questions about efficient LLM "
        "inference (quantization, KV-cache optimization, speculative decoding). "
        "Answer ONLY using the provided context. If the context does not contain "
        "the answer, say so clearly instead of guessing. Cite the paper title for "
        "each claim.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return prompt


def answer_query(query, k=5):
    _load()
    chunks = retrieve(query, k=k)
    prompt = build_prompt(query, chunks)

    response = _client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    sources = list({(c["title"], c["arxiv_id"]) for c in chunks})
    return answer, sources