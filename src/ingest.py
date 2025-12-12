# src/ingest.py
import os
from pathlib import Path
from typing import List, Dict
from src.utils import load_any, chunk_text, is_text_file, is_pdf_file, is_image_file, get_ext
import logging
# src/ingest.py  (replace ingest_dir function with this)
from pathlib import Path
from typing import List, Dict
from src.utils import load_any, chunk_text, get_ext
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

def ingest_dir(data_dir: Path = DATA_DIR, ocr_for_pdf: bool = True, lang: str = "eng") -> List[Dict]:
    """
    Walk data_dir and data/uploads and return a list of chunk dicts:
    {id, text, meta: {source, page(optional), chunk, is_scanned_pdf, orig_path}}
    - For PDFs: process per page (digital text if available else OCR per page).
    - For images: process per frame (multi-page TIFF/GIF frames).
    - For txt: chunk the full text (as before).
    """
    docs: List[Dict] = []
    # collect files from data_dir and data/uploads
    files = []
    up = data_dir / "uploads"
    if data_dir.exists():
        for p in sorted(data_dir.iterdir()):
            if p.is_file():
                files.append(p)
    if up.exists():
        for p in sorted(up.iterdir()):
            if p.is_file():
                files.append(p)

    for p in files:
        ext = get_ext(p)
        try:
            # For PDFs we want page-by-page chunking
            if ext == ".pdf":
                # use load_any to get digital text first; if digital text exists,
                # split it into pages where possible (pypdf extract). Otherwise use OCR fallback
                text, meta = load_any(p, ocr_for_pdf=False, lang=lang)
                if text and text.strip():
                    # attempt to split into pages heuristically using newline markers:
                    # pypdf extract_text often gives pages concatenated; better approach:
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(p))
                        # try extracting per page
                        for page_idx, page in enumerate(reader.pages):
                            page_text = page.extract_text() or ""
                            if not page_text.strip():
                                continue
                            page_chunks = chunk_text(page_text)
                            for i, c in enumerate(page_chunks):
                                docs.append({
                                    "id": f"{meta['source']}__page{page_idx}__chunk{i}",
                                    "text": c,
                                    "meta": {
                                        "source": meta['source'],
                                        "page": page_idx,
                                        "chunk": i,
                                        "is_scanned_pdf": False,
                                        "orig_path": str(p)
                                    }
                                })
                    except Exception:
                        # fallback: chunk the whole extracted text
                        page_chunks = chunk_text(text)
                        for i, c in enumerate(page_chunks):
                            docs.append({
                                "id": f"{meta['source']}__chunk{i}",
                                "text": c,
                                "meta": {
                                    "source": meta['source'],
                                    "chunk": i,
                                    "is_scanned_pdf": False,
                                    "orig_path": str(p)
                                }
                            })
                else:
                    # no digital text - use OCR per page (load_any with ocr_for_pdf True)
                    # load_any will return concatenated OCR text; we instead convert PDF pages to images
                    try:
                        from pdf2image import convert_from_path
                        pil_pages = convert_from_path(str(p), dpi=300)
                        for page_idx, pil in enumerate(pil_pages):
                            page_text, _m = load_any(pil, ocr_for_pdf=False, lang=lang)
                            if not page_text or not page_text.strip():
                                continue
                            page_chunks = chunk_text(page_text)
                            for i, c in enumerate(page_chunks):
                                docs.append({
                                    "id": f"{p.name}__page{page_idx}__chunk{i}",
                                    "text": c,
                                    "meta": {
                                        "source": p.name,
                                        "page": page_idx,
                                        "chunk": i,
                                        "is_scanned_pdf": True,
                                        "orig_path": str(p)
                                    }
                                })
                    except Exception:
                        # last fallback: use load_any full text and chunk
                        full_text, meta = load_any(p, ocr_for_pdf=True, lang=lang)
                        if full_text and full_text.strip():
                            page_chunks = chunk_text(full_text)
                            for i, c in enumerate(page_chunks):
                                docs.append({
                                    "id": f"{p.name}__chunk{i}",
                                    "text": c,
                                    "meta": {
                                        "source": p.name,
                                        "chunk": i,
                                        "is_scanned_pdf": True,
                                        "orig_path": str(p)
                                    }
                                })

            # Images: treat each frame/page separately (multi-page TIFF/GIF)
            elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}:
                text, meta = load_any(p, ocr_for_pdf=False, lang=lang)
                if not text or not text.strip():
                    continue
                # If multi-page, load_any may return concatenated text; attempt to split by page markers.
                # We just chunk the returned text — chunk_text will break it down into multiple chunks.
                page_chunks = chunk_text(text)
                for i, c in enumerate(page_chunks):
                    docs.append({
                        "id": f"{meta['source']}__chunk{i}",
                        "text": c,
                        "meta": {
                            "source": meta['source'],
                            "chunk": i,
                            "orig_path": str(p)
                        }
                    })

            # Plain text
            elif ext == ".txt":
                text, meta = load_any(p, ocr_for_pdf=False, lang=lang)
                if not text or not text.strip():
                    continue
                page_chunks = chunk_text(text)
                for i, c in enumerate(page_chunks):
                    docs.append({
                        "id": f"{meta['source']}__chunk{i}",
                        "text": c,
                        "meta": {
                            "source": meta['source'],
                            "chunk": i,
                            "orig_path": str(p)
                        }
                    })

            else:
                # unknown extension: attempt to load_any and chunk
                text, meta = load_any(p, ocr_for_pdf=True, lang=lang)
                if not text or not text.strip():
                    continue
                page_chunks = chunk_text(text)
                for i, c in enumerate(page_chunks):
                    docs.append({
                        "id": f"{meta.get('source', p.name)}__chunk{i}",
                        "text": c,
                        "meta": {
                            "source": meta.get('source', p.name),
                            "chunk": i,
                            "orig_path": str(p)
                        }
                    })
        except Exception as e:
            logger.warning("Failed ingesting %s: %s", p, e)
            continue

    logger.info("Ingested %d chunks from %s", len(docs), data_dir)
    return docs
