"""
ocr_service.py
--------------
OCR fallback untuk halaman PDF yang tidak memiliki teks (scan-only PDF).

Catatan: OCR penuh memerlukan tesseract atau easyocr yang berat.
Implementasi ini menggunakan Pillow untuk konversi dasar dan
mengembalikan string kosong jika tidak ada teks dapat diekstrak.
Untuk OCR production yang akurat, install pytesseract + tesseract-ocr.
"""

from PIL import Image


def image_to_text(img: Image.Image) -> str:
    """
    Konversi gambar halaman PDF ke teks.

    Saat ini mengembalikan string kosong sebagai fallback ringan.
    Untuk OCR penuh, uncomment kode pytesseract di bawah dan
    install: pip install pytesseract + tesseract-ocr binary.
    """
    # ── Optional: Full OCR dengan pytesseract ──
    # Uncomment baris berikut jika tesseract sudah terinstall:
    #
    # import pytesseract
    # return pytesseract.image_to_string(img, lang='ind+eng').strip()

    # Fallback: tidak ada teks dari halaman scan
    return ""