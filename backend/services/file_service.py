"""
file_service.py
---------------
Dispatcher untuk membaca berbagai jenis dokumen.
"""

from pathlib import Path

from .pdf_service import extract_pdf
from .docx_service import extract_docx
from .txt_service import extract_txt


SUPPORTED_EXTENSIONS = {

    ".pdf",

    ".docx",

    ".txt"

}


def extract_document(filename: str, file_bytes: bytes):

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(

            f"Unsupported file type: {extension}"

        )

    if extension == ".pdf":

        return extract_pdf(file_bytes)

    if extension == ".docx":

        return extract_docx(file_bytes)

    if extension == ".txt":

        return extract_txt(file_bytes)

    raise ValueError(

        "Unknown document format."

    )