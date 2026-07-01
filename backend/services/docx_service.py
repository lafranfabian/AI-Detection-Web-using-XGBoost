"""
docx_service.py
----------------
Membaca file Microsoft Word (.docx)
"""

from io import BytesIO
from docx import Document


def extract_docx(file_bytes: bytes):
    """
    Extract seluruh text dari file DOCX.

    Returns
    -------
    dict
    {
        "text": "...",
        "pages": None,
        "ocr_used": False
    }
    """

    document = Document(BytesIO(file_bytes))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    document_text = "\n".join(paragraphs)

    return {

        "text": document_text,

        "pages": None,

        "ocr_used": False

    }