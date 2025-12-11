# src/utils.py
import os
from typing import List
from pypdf import PdfReader
from PIL import Image
import pytesseract

TEXT_EXTS = {".txt"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
PDF_EXTS = {".pdf"}


def get_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_text_file(path: str) -> bool:
    return get_ext(path) in TEXT_EXTS


def is_pdf_file(path: str) -> bool:
    return get_ext(path) in PDF_EXTS


def is_image_file(path: str) -> bool:
    return get_ext(path) in IMAGE_EXTS


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """
    Extract text from a PDF using pypdf.
    Note: This works well for "digital" PDFs, not always for scanned ones.
    """
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def load_image_ocr(path: str, lang: str = "eng") -> str:
    """
    Extract text from an image using Tesseract OCR.
    """
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text


def load_any(path: str) -> str:
    """
    Dispatch helper: decide how to read a file based on extension.
    Returns text (possibly empty string).
    """
    if is_text_file(path):
        return load_txt(path)
    if is_pdf_file(path):
        return load_pdf(path)
    if is_image_file(path):
        return load_image_ocr(path)
    # unsupported type
    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple word-based chunking.
    """
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks
