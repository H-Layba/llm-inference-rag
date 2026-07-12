"""
Run this locally to download all papers listed in data/paper_corpus.csv
into data/raw_pdfs/, named <arxiv_id>.pdf so ingest.py can match metadata.
"""

import pandas as pd
import requests
from pathlib import Path
import time

CSV_PATH = Path(__file__).parent.parent / "data" / "paper_corpus.csv"
OUT_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        out_path = OUT_DIR / f"{row['arxiv_id']}.pdf"
        if out_path.exists():
            print(f"Skipping {row['arxiv_id']} (already downloaded)")
            continue
        print(f"Downloading {row['arxiv_id']}: {row['title']}")
        try:
            resp = requests.get(row["pdf_url"], timeout=30)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(1)  # be polite to arXiv's servers

    print("Done. Verify data/paper_corpus.csv 'downloaded' column manually if desired.")


if __name__ == "__main__":
    main()
