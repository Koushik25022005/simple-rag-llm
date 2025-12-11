# src/ingest.py
import os
from pathlib import Path
from typing import List, Dict
from src.utils import load_any, chunk_text, is_text_file, is_pdf_file, is_image_file, get_ext
import logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

def list_data_files(data_dir: Path = DATA_DIR) -> List[Path]:
    files = []
    for p in data_dir.iterdir():
        if p.is_file():
            files.append(p)
    # include uploads/ if exists
    up = data_dir / "uploads"
    if up.exists():
        for p in up.iterdir():
            if p.is_file():
                files.append(p)
    return sorted(files)

def ingest_dir(data_dir: Path = DATA_DIR, ocr_for_pdf: bool = True, lang: str = "eng") -> List[Dict]:
    """
    Walk data_dir and data/upload/ and return list of chunk dicts:
    {id, text, meta: {source, chunk, page?, is_scanned_pdf}}
    """
    docs = []
    files = list_data_files(data_dir)
    for p in files:
        try:
            text, meta = load_any(p, ocr_for_pdf=ocr_for_pdf, lang=lang)
        except Exception as e:
            logger.warning("Failed to read %s: %s", p, e)
            continue
        if not text or not text.strip():
            logger.info("No text from %s; skipping", p)
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append({
                "id": f"{meta['source']}__chunk_{i}",
                "text": c,
                "meta": {
                    "source": meta['source'],
                    "chunk": i,
                    "is_scanned_pdf": meta.get("is_scanned_pdf", False),
                    "orig_path": str(p)
                }
            })
    return docs

if __name__ == "__main__":
    docs = ingest_dir()
    print(f"Ingested {len(docs)} chunks from data/")
    if docs:
        print(docs[0])
