"""
txt_service.py
--------------
Reader file TXT.
Support UTF-8 dan fallback latin1.
"""


def extract_txt(file_bytes: bytes):

    try:

        text = file_bytes.decode("utf-8")

    except UnicodeDecodeError:

        text = file_bytes.decode(
            "latin-1",
            errors="ignore"
        )

    return {

        "text": text,

        "pages": None,

        "ocr_used": False

    }