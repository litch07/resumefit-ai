"""File parsers for ResumeFit AI."""

from __future__ import annotations

from io import BytesIO
from typing import Union

import pdfplumber
from PyPDF2 import PdfReader
from docx import Document


FileLike = Union[bytes, bytearray, BytesIO]


def _to_bytes_io(file: object) -> BytesIO:
    """Normalize file inputs into a BytesIO stream for parsers."""
    # Normalize varying file input types into a standard BytesIO stream.
    try:
        if isinstance(file, (bytes, bytearray)):
            return BytesIO(file)
        if hasattr(file, "read"):
            data = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
            if isinstance(data, str):
                data = data.encode("utf-8", errors="ignore")
            return BytesIO(data)
        if hasattr(file, "getvalue"):
            data = file.getvalue()
            if isinstance(data, str):
                data = data.encode("utf-8", errors="ignore")
            return BytesIO(data)
    except Exception:
        return BytesIO(b"")

    return BytesIO(b"")


def extract_text_from_pdf(file: object) -> str:
    """Extract text from a PDF using pdfplumber, fallback to PyPDF2."""
    # Attempt pdfplumber for accurate layout parsing; fallback to PyPDF2.
    try:
        file_stream = _to_bytes_io(file)
        with pdfplumber.open(file_stream) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages).strip()
    except Exception:
        pass

    try:
        file_stream = _to_bytes_io(file)
        reader = PdfReader(file_stream)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""


def extract_text_from_docx(file: object) -> str:
    """Extract text from a DOCX file using python-docx."""
    # Extract DOCX text iteratively by paragraph.
    try:
        file_stream = _to_bytes_io(file)
        doc = Document(file_stream)
        paragraphs = [para.text for para in doc.paragraphs if para.text]
        return "\n".join(paragraphs).strip()
    except Exception:
        return ""


def extract_text_from_txt(file: object) -> str:
    """Extract text from a plain text file."""
    try:
        if isinstance(file, (bytes, bytearray)):
            return file.decode("utf-8", errors="ignore").strip()
        if hasattr(file, "read"):
            data = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="ignore").strip()
            return str(data).strip()
        if hasattr(file, "getvalue"):
            data = file.getvalue()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="ignore").strip()
            return str(data).strip()
    except Exception:
        return ""

    return ""


def extract_text(file: object, filename: str) -> str:
    """Route extraction based on file extension (pdf, docx, txt)."""
    # Route extraction to the appropriate parser based on file extension.
    try:
        extension = filename.lower().rsplit(".", 1)[-1]
    except Exception:
        return ""

    if extension == "pdf":
        return extract_text_from_pdf(file)
    if extension == "docx":
        return extract_text_from_docx(file)
    if extension == "txt":
        return extract_text_from_txt(file)

    return ""
