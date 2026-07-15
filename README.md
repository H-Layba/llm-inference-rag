# Efficient LLM Inference - Research Assistant (RAG)

A retrieval-augmentated generation system for querying a curated repository
of ~30 arXiv papers on LLM inference — quantization, KV-cache optimization, 
and speculative decoding. Answers are based exclusively on the information in
the papers retrieved and the sources are listed with each answer.

**Live demo:** https://llm-inference-rag-kzkn44ma7w4ki2hiujnqjy.streamlit.app/

<img width="800" height="600" alt="Screenshot 2026-07-15 221204" src="https://github.com/user-attachments/assets/f792a87c-2d97-4b91-8fcd-985c764387b3" />
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/bb311a54-2e2d-4645-89fd-1e34ae0f2f14" />

## Why this project

General purpose search or asking the LLM directly gives scattered
pointers or ungrounded (sometimes hallucinated) summaries. The system
fetches the actual relevant passage from a fixed set of documents first,
then formulate an answer that is limited to this context – the answers 
can be traced back to a specific document and the specific context.


## Architecture

```
PDFs (arXiv) → PyMuPDF extraction → section-aware chunking → strip references
                                                                    ↓
                                              sentence-transformers embeddings
                                                                    ↓
                                                            FAISS vector index
                                                                    ↓
User question → embed → retrieve top-k chunks → Qwen2.5-7B-Instruct (HF Inference API)
                                                                    ↓
                                                    grounded answer using cited sources
```

## Engineering decisions worth knowing about

- **Section-aware chunking.** Naive fixed-length chunks are generated in
  equations, tables, and section boundaries; impacts retrieval qualities for
  dense technical texts. Where possible chunks are created which preserve
  recognized section boundaries, such as Abstract, Introduction, Method, etc.
- **References are removed prior to chunking** Each bibliographic entry is no
  use in retrieval and would pollute the vector index.
- **Each chunk carries metadata** (paper title, arXiv ID, section, category)
  so that an answer can refer to where it was retrieved from; and retrieval
  could be specialized by category: (quantiation / kv_cache / speculative_decoding / survey).
- **Fully free/open-source stack.** Qwen2.5-7B-Instruct via the Hugging Face
   Inference API, sentence-transformers for embeddings and FAISS for vector
   search.

## Project structure

```
data/
  paper_corpus.csv     # source-of-truth list of all papers (title, arxiv_id, category, pdf_url)
  raw_pdfs/             # downloaded PDFs (gitignored, not committed — see setup below)
  faiss.index           # built vector index (committed and for deployment)
  chunks_meta.json       # chunk metadata (committed, needed for deployment)
src/
  ingest.py             # PDF -> chunks - per section -> data/chunks.json
  build_index.py        # Chunks -> Embeddings -> FAISS index.
  rag_pipeline.py        # retrieval + generation logic.
scripts/
  download_papers.py    # used to download all papers stored in the paper_corpus.csv.
streamlit_app.py         # (streamlit UI / deployed version — see Deployment below)
app.py                   # Gradio UI (local-only alternative)
.streamlit/config.toml    # custom dark theme for the Streamlit UI
```

## Setup (local)

```bash
pip install –r requirements.txt

# 1. Download papers (or place your own PDFs, named <arxiv_id>.pdf, in data/raw_pdfs/)
python scripts/download_papers.py

# 2. Extract + chunk
python src/ingest.py

# 3. Compiles the vector index
python src/build_index.py

# 4. Place your Hugging Face token
cp .env.example .env  #then edit .env using your token
export HF_TOKEN=your_token_here     # Windows PowerShell: $env:HF_TOKEN = "your_token_here"

# 5. Run the app
streamlit run streamlit_app.py      # or python app.py (Gradio, only on local machine)
```

## Deployment

Deployed on **Streamlit Community Cloud**, connected directly to
this GitHub repo:

1. Push the repo to GitHub, including the pre-built `data/faiss.index` and
   `data/chunks_meta.json`
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, branch `main`, main file `streamlit_app.py`
3. Add `HF_TOKEN` under the app's Secrets (TOML format:
   `HF_TOKEN = "your_token"`)
4. Deploy

### A note on the free LLM tier

Not all models are permanently available on Hugging Face's free serverless
Inference API and the availability of models varies. This project aimed to
deploy `Mistral-7B-Instruct` but ended up addressing issues related to availability
and provider-routing on the free tier launched by Mistral AI and thus proceeds 
with `Qwen2.5-7B-Instruct` instead. If your fork of this project combined with the 
model you've specified in `src/rag_pipeline.py` doesn't work, make sure to check the
status of a candidate model's page on huggingface.co under the "Inference Providers"
section before installing it.


## Data provenance

All papers can be found on arXiv; you can find their original arXiv ID and
pdf link in `data/paper_corpus.csv`. The PDFs contained in this repo are NOT
raw PDFs – use `scripts/download_papers.py` to obtain raw PDFs directly from
arXiv.

## Known limitations

- Corpus is intentionally narrow, limited to ~30 papers —
  this is a demonstrative, not a full literature database.
- Not all paper layouts provide a perfect heuristic (font-size-based) section
  detection.

