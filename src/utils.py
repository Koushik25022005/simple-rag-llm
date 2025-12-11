# src/utils.py
import os
from typing import List, Tuple, Optional
from pypdf import PdfReader
from PIL import Image
import pytesseract
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

TEXT_EXTS = {".txt"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
PDF_EXTS = {".pdf"}

def get_ext(path: str) -> str:
    return Path(path).suffix.lower()

def is_text_file(path: str) -> bool:
    return get_ext(path) in TEXT_EXTS

def is_pdf_file(path: str) -> bool:
    return get_ext(path) in PDF_EXTS

def is_image_file(path: str) -> bool:
    return get_ext(path) in IMAGE_EXTS

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf_text(path: str) -> str:
    # Extract digital text from PDF pages (fast)
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)

def load_image_ocr(path: str, lang: str = "eng") -> str:
    try:
        image = Image.open(path)
    except Exception as e:
        logger.exception("PIL cannot open image %s: %s", path, e)
        return ""
    try:
        text = pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        logger.exception("pytesseract failed on %s: %s", path, e)
        return ""
    return text

def pdf_to_images(path: str, dpi: int = 300):
    """
    Convert PDF pages to PIL images using pdf2image if available.
    Returns list of PIL.Image objects.
    """
    try:
        from pdf2image import convert_from_path
    except Exception:
        raise RuntimeError("pdf2image not installed or poppler not available")

    images = convert_from_path(path, dpi=dpi)
    return images

def load_pdf_with_ocr(path: str, lang: str = "eng") -> str:
    """
    For scanned PDFs: convert pages to images and run OCR per page.
    """
    try:
        imgs = pdf_to_images(path)
    except Exception as e:
        raise RuntimeError(f"Cannot convert PDF to images: {e}")

    texts = []
    for img in imgs:
        texts.append(pytesseract.image_to_string(img, lang=lang))
    return "\n".join(texts)


def load_any(path: str, ocr_for_pdf: bool = True, lang: str = "eng") -> Tuple[str, dict]:
    """
    Returns (extracted_text, metadata) where metadata includes:
      - source (filename)
      - is_scanned_pdf (bool)
    """
    path = str(path)
    meta = {"source": os.path.basename(path), "is_scanned_pdf": False}
    ext = get_ext(path)
    if is_text_file(path):
        return load_txt(path), meta
    if is_image_file(path):
        return load_image_ocr(path, lang=lang), meta
    if is_pdf_file(path):
        # try digital text first
        text = load_pdf_text(path)
        if text.strip():
            return text, meta
        # fallback to OCR if requested
        if ocr_for_pdf:
            try:
                ocr_text = load_pdf_with_ocr(path, lang=lang)
                meta["is_scanned_pdf"] = True
                return ocr_text, meta
            except Exception as e:
                logger.warning("PDF OCR failed for %s: %s", path, e)
                return "", meta
    return "", meta


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """
    Word-based chunking; returns list of strings.
    """
    if not text:
        return []
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks
