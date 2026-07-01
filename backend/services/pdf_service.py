# backend/services/pdf_service.py

import fitz
from PIL import Image

from .ocr_service import image_to_text


def extract_pdf(file_bytes: bytes):

    doc = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    full_text = []

    total_pages = len(doc)

    ocr_used = False

    for page in doc:

        text = page.get_text().strip()

        if text:

            full_text.append(text)

            continue

        ocr_used = True

        pix = page.get_pixmap(
            dpi=300
        )

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        ocr_text = image_to_text(img)

        full_text.append(ocr_text)

    document_text = "\n".join(full_text)

    return {

        "text": document_text,

        "pages": total_pages,

        "ocr_used": ocr_used

    }