# src/ingest.py
import os
from typing import List, Dict
from src.utils import (
    load_any,
    chunk_text,
    is_text_file,
    is_pdf_file,
    is_image_file,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ingest_dir(data_dir: str = DATA_DIR) -> List[Dict]:
    """
    Walk through data_dir and create chunks from:
    - .txt files
    - .pdf files
    - image files (png, jpg, etc., via OCR)
    """
    docs: List[Dict] = []

    for fn in os.listdir(data_dir):
        path = os.path.join(data_dir, fn)

        if not os.path.isfile(path):
            continue  # skip subdirs etc.

        # We only handle txt, pdf, image types
        if not (is_text_file(path) or is_pdf_file(path) or is_image_file(path)):
            continue

        raw_text = load_any(path)
        if not raw_text or not raw_text.strip():
            # nothing useful extracted
            print(f"[WARN] No text extracted from {fn}, skipping.")
            continue

        chunks = chunk_text(raw_text)

        for i, c in enumerate(chunks):
            docs.append({
                "id": f"{fn}__{i}",
                "text": c,
                "meta": {
                    "source": fn,
                    "chunk": i
                }
            })

    print(f"Ingested {len(docs)} chunks from directory: {data_dir}")
    return docs


if __name__ == "__main__":
    docs = ingest_dir()
    print(f"Example chunks:\n")
    for d in docs[:3]:
        print(d["id"], "=>", d["text"][:200].replace("\n", " "), "...\n")
