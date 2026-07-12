---
title: Efficient LLM Inference RAG
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Efficient LLM Inference — Research Assistant (RAG)

A RAG system for querying ~30 arXiv papers on efficient LLM inference
(quantization, KV-cache optimization, speculative decoding). Answers are
grounded in retrieved paper excerpts, with sources cited.

Full source code and write-up: _[add your GitHub repo link here]_

## How to deploy this yourself

1. Create a new Space on Hugging Face (SDK: Gradio)
2. Push this repo's contents to the Space repo (this file becomes the
   Space's `README.md`, replacing the GitHub one)
3. In the Space's **Settings → Repository secrets**, add `HF_TOKEN` with a
   valid Hugging Face access token (needed for the Inference API calls)
4. Also push the pre-built `data/faiss.index` and `data/chunks_meta.json`
   (run `src/ingest.py` and `src/build_index.py` locally first — the Space
   itself doesn't need to re-process PDFs on every boot)
5. The Space will install from `requirements.txt` and launch `app.py`
   automatically
