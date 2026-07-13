# Efficient LLM Inference — Research Assistant (RAG)

A retrieval-augmented generation system for querying a curated corpus of ~30
arXiv papers on efficient LLM inference — quantization, KV-cache
optimization, and speculative decoding. Answers are grounded strictly in the
retrieved paper excerpts, with sources shown alongside every answer.

**Live demo:** https://llm-inference-rag-kzkn44ma7w4ki2hiujnqjy.streamlit.app/

## Why this project

General-purpose search or asking an LLM directly gives you scattered
pointers or ungrounded (sometimes hallucinated) summaries. This system
retrieves the actual relevant passages from a fixed set of papers first,
then generates an answer constrained to that context — so answers are
traceable back to a specific paper and section.

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
                                                    grounded answer + cited sources
```

## Engineering decisions worth knowing about

- **PDF parsing: PyMuPDF, not GROBID.** GROBID gives more reliable section
  detection for scientific papers but requires running a separate Java
  service. For a ~30-paper corpus, PyMuPDF with font-size-based header
  heuristics was a better time/quality tradeoff. This is the natural first
  upgrade if the corpus grows.
- **Section-aware chunking.** Naive fixed-length chunking cuts through
  equations, tables, and section boundaries, which hurts retrieval quality
  on dense technical text. Chunks here are built to respect detected section
  boundaries (Abstract, Introduction, Method, etc.) where possible.
- **References are stripped before chunking.** Bibliography entries are pure
  noise for retrieval and would otherwise pollute the vector index.
- **Every chunk carries metadata** (paper title, arXiv ID, section, category)
  so answers can cite their source, and retrieval can later be filtered by
  category (quantization / kv_cache / speculative_decoding / survey).
- **Fully free/open-source stack.** Qwen2.5-7B-Instruct via the Hugging Face
  Inference API, sentence-transformers for embeddings, FAISS for vector
  search — no paid API dependency. (Note: the specific model was chosen
  based on live availability on HF's free serverless tier at deployment
  time — see "A note on the free LLM tier" below.)

## Project structure

```
data/
  paper_corpus.csv     # source-of-truth list of all papers (title, arxiv_id, category, pdf_url)
  raw_pdfs/             # downloaded PDFs (gitignored, not committed — see setup below)
  faiss.index            # built vector index (committed, needed for deployment)
  chunks_meta.json       # chunk metadata (committed, needed for deployment)
src/
  ingest.py             # PDF -> section-aware chunks -> data/chunks.json
  build_index.py        # chunks -> embeddings -> FAISS index
  rag_pipeline.py        # retrieval + generation logic
scripts/
  download_papers.py    # downloads all papers listed in paper_corpus.csv
streamlit_app.py          # Streamlit UI (deployed version — see Deployment below)
app.py                   # Gradio UI (local-only alternative)
.streamlit/config.toml    # custom dark theme for the Streamlit UI
```

## Setup (local)

```bash
pip install -r requirements.txt

# 1. Download papers (or place your own PDFs, named <arxiv_id>.pdf, in data/raw_pdfs/)
python scripts/download_papers.py

# 2. Extract + chunk
python src/ingest.py

# 3. Build the vector index
python src/build_index.py

# 4. Set your Hugging Face token
cp .env.example .env   # then edit .env with your token
export HF_TOKEN=your_token_here     # Windows PowerShell: $env:HF_TOKEN = "your_token_here"

# 5. Run the app
streamlit run streamlit_app.py      # or: python app.py (Gradio, local only)
```

## Deployment

Deployed on **Streamlit Community Cloud** (free tier), connected directly to
this GitHub repo:

1. Push the repo to GitHub, including the pre-built `data/faiss.index` and
   `data/chunks_meta.json` (Streamlit Cloud doesn't rebuild these — the app
   loads them directly at runtime)
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, branch `main`, main file `streamlit_app.py`
3. Add `HF_TOKEN` under the app's Secrets (TOML format:
   `HF_TOKEN = "your_token"`)
4. Deploy

### A note on the free LLM tier

Hugging Face's free serverless Inference API doesn't host every model at all
times, and which models are available can change. This project originally
targeted Mistral-7B-Instruct but switched to Qwen2.5-7B-Instruct after
hitting availability/provider-routing errors on the free tier during
deployment. If you fork this project and the configured model in
`src/rag_pipeline.py` stops working, check a candidate model's page on
huggingface.co for its "Inference Providers" status before swapping it in.

Separately, Hugging Face Spaces' free tier no longer reliably supports the
Gradio SDK at the time of writing (new Spaces are steered toward paid tiers
or ZeroGPU hardware) — this is why the deployed version uses Streamlit
Community Cloud instead. `app.py` (Gradio) is kept in the repo as a local
alternative UI.

## Data provenance

All papers are sourced from arXiv and listed with their original arXiv ID
and PDF link in `data/paper_corpus.csv`. Raw PDFs are not committed to this
repo — run `scripts/download_papers.py` to fetch them directly from arXiv.

## Known limitations

- Corpus is intentionally narrow (~30 papers, one subfield) — this is a
  focused demo, not a comprehensive literature database.
- Section detection is heuristic (font-size based) and not perfect on all
  paper layouts.
- Answers are only as good as retrieval — if a question falls outside the
  corpus's coverage, the model is instructed to say so rather than guess,
  but this isn't foolproof.
- **Grounding is prompt-enforced, not hard-constrained.** In testing, the
  model sometimes correctly notes the context doesn't cover a question, then
  supplements with general knowledge anyway (e.g. answering "What's the
  capital of France?" after noting it wasn't in the corpus). This is an
  accepted tradeoff for this demo — it keeps answers useful rather than
  uselessly blank — but means grounding isn't strictly guaranteed the way a
  hard-filtered RAG system would enforce it.

## Future improvements

- Swap PyMuPDF for GROBID for more reliable structural parsing at scale
- Add category-filtered retrieval (e.g., "only search quantization papers")
- Expand corpus and add incremental re-indexing