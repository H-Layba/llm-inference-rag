# Efficient LLM Inference — Research Assistant (RAG)

A retrieval-augmented generation system for querying a curated corpus of ~30
arXiv papers on efficient LLM inference — quantization, KV-cache
optimization, and speculative decoding. Answers are grounded strictly in the
retrieved paper excerpts, with sources shown alongside every answer.

**Live demo:** _[add your Hugging Face Space link here once deployed]_

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
User question → embed → retrieve top-k chunks → Mistral-7B-Instruct (HF Inference API)
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
- **Fully free/open-source stack.** Mistral-7B-Instruct via the Hugging Face
  Inference API, sentence-transformers for embeddings, FAISS for vector
  search — no paid API dependency.

## Project structure

```
data/
  paper_corpus.csv     # source-of-truth list of all papers (title, arxiv_id, category, pdf_url)
  raw_pdfs/             # downloaded PDFs (gitignored — see setup below)
src/
  ingest.py             # PDF -> section-aware chunks -> data/chunks.json
  build_index.py        # chunks -> embeddings -> FAISS index
  rag_pipeline.py        # retrieval + generation logic
scripts/
  download_papers.py    # downloads all papers listed in paper_corpus.csv
app.py                   # Gradio UI (local + Hugging Face Spaces entry point)
```

## Setup

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
export HF_TOKEN=your_token_here

# 5. Run the app
python app.py
```

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