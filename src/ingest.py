"""
Ingests PDFs from data/raw_pdfs/, extracts text with PyMuPDF, splits into
section-aware chunks, strips references, and attaches metadata from
paper_corpus.csv. Outputs data/chunks.json for build_index.py to embed.

Naming convention: raw_pdfs/<arxiv_id>.pdf  (e.g. 2210.17323.pdf)
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"
CSV_PATH = Path(__file__).parent.parent / "data" / "paper_corpus.csv"
OUT_PATH = Path(__file__).parent.parent / "data" / "chunks.json"

CHUNK_SIZE = 450       # approx tokens (we use words as a cheap proxy)
CHUNK_OVERLAP = 60
REFERENCES_HEADERS = {"references", "bibliography", "acknowledgments", "acknowledgements"}


def extract_text_blocks(pdf_path):
    """Extract text with rough section-header detection based on font size."""
    doc = fitz.open(pdf_path)
    blocks = []
    font_sizes = []

    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span["size"])

    if not font_sizes:
        return []

    body_size = max(set(font_sizes), key=font_sizes.count)  # most common size = body text
    header_threshold = body_size + 1.5  # headers are noticeably larger/bolder

    current_section = "Abstract"
    stop = False

    for page in doc:
        if stop:
            break
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                if not line_text:
                    continue
                max_span_size = max((s["size"] for s in line.get("spans", [])), default=body_size)

                # detect section headers
                if max_span_size >= header_threshold and len(line_text.split()) <= 8:
                    header_norm = line_text.lower().strip(" .0123456789")
                    if header_norm in REFERENCES_HEADERS:
                        stop = True  # drop everything from References onward
                        break
                    current_section = line_text
                    continue

                blocks.append({"section": current_section, "text": line_text})
            if stop:
                break
    return blocks


def chunk_text(blocks, meta):
    """Group blocks into ~CHUNK_SIZE-word chunks, staying within a section where possible."""
    chunks = []
    buffer = []
    buffer_section = None
    word_count = 0

    def flush():
        nonlocal buffer, word_count
        if buffer:
            chunks.append({
                "text": " ".join(buffer),
                "section": buffer_section,
                **meta
            })
        buffer = []
        word_count = 0

    for block in blocks:
        if buffer_section is not None and block["section"] != buffer_section and word_count >= CHUNK_SIZE * 0.5:
            flush()
        buffer_section = block["section"]
        words = block["text"].split()
        buffer.append(block["text"])
        word_count += len(words)

        if word_count >= CHUNK_SIZE:
            flush()

    flush()

    # add simple word-overlap between consecutive chunks of the same paper
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1]["text"].split()
        overlap = " ".join(prev_words[-CHUNK_OVERLAP:])
        chunks[i]["text"] = overlap + " " + chunks[i]["text"]

    return chunks


def main():
    df = pd.read_csv(CSV_PATH, dtype={"arxiv_id": str})
    all_chunks = []

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}. Place downloaded papers there, named <arxiv_id>.pdf")
        return

    for pdf_path in pdf_files:
        # strip arXiv version suffix, e.g. "2210.17323v2" -> "2210.17323"
        arxiv_id = re.sub(r"v\d+$", "", pdf_path.stem)
        row = df[df["arxiv_id"] == arxiv_id]
        if row.empty:
            print(f"WARNING: {pdf_path.name} not found in paper_corpus.csv, skipping metadata match")
            meta = {"title": arxiv_id, "arxiv_id": arxiv_id, "category": "unknown"}
        else:
            r = row.iloc[0]
            meta = {"title": r["title"], "arxiv_id": r["arxiv_id"], "category": r["category"]}

        print(f"Processing {pdf_path.name} ...")
        blocks = extract_text_blocks(pdf_path)
        chunks = chunk_text(blocks, meta)
        all_chunks.extend(chunks)

    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Done. {len(all_chunks)} chunks written to {OUT_PATH}")


if __name__ == "__main__":
    main()