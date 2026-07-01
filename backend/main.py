"""
main.py
-------
FastAPI backend untuk AI Text Detector.

Endpoints:
    POST /predict        → Prediksi dari teks langsung (JSON)
    POST /predict-file   → Prediksi dari file upload (PDF/DOCX/TXT)
    GET  /health         → Health check
    GET  /               → Info API

Jalankan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from predictor import predict, _load_artifacts
from services.file_service import extract_document

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LIFESPAN: Pre-load artifacts saat startup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load semua ML artifacts saat server startup."""
    logger.info("Loading ML artifacts...")
    try:
        _load_artifacts()
        logger.info("ML artifacts loaded successfully.")
    except RuntimeError as e:
        logger.error("FATAL: %s", e)
    yield
    logger.info("Shutting down AI Text Detector API.")


# ─────────────────────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Text Detector API",
    description=(
        "Deteksi apakah sebuah teks ditulis oleh AI atau manusia "
        "menggunakan model XGBoost. Mendukung input teks langsung "
        "maupun upload file (PDF, DOCX, TXT)."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────
# CORS MIDDLEWARE
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Ganti ke domain spesifik di production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Teks tidak boleh kosong.")
        if len(v) < 20:
            raise ValueError("Teks terlalu pendek. Minimal 20 karakter untuk hasil yang akurat.")
        if len(v) > 50_000:
            raise ValueError("Teks terlalu panjang. Maksimal 50.000 karakter.")
        return v


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


class DocumentInfo(BaseModel):
    filename: str
    pages: int | None
    words: int
    characters: int
    ocr_used: bool


class PredictFileResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    document: DocumentInfo


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "AI Text Detector API",
        "version": "1.1.0",
        "endpoints": {
            "POST /predict":      "Deteksi teks dari JSON body",
            "POST /predict-file": "Deteksi teks dari file upload (PDF/DOCX/TXT)",
            "GET  /health":       "Health check",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    return HealthResponse(
        status="ok",
        version="1.1.0",
        model="XGBoost (aiidentification.pkl)",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictRequest):
    """
    Prediksi apakah teks ditulis oleh AI atau manusia.

    **Input:**
    ```json
    { "text": "teks yang ingin dicek..." }
    ```

    **Output:**
    ```json
    {
        "prediction": "AI Generated",
        "confidence": 96.52,
        "probabilities": { "human": 3.48, "ai": 96.52 }
    }
    ```
    """
    start_time = time.perf_counter()

    try:
        result = predict(request.text)
    except RuntimeError as e:
        logger.error("Artifact error: %s", e)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model artifacts belum tersedia.",
                "solution": "Jalankan: python3 scripts/build_preprocessor.py",
            },
        )
    except Exception as e:
        logger.exception("Unexpected error during prediction: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "Terjadi kesalahan internal saat memproses teks."},
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "Prediction: %s (%.2f%%) | %.1f ms | chars=%d",
        result["prediction"], result["confidence"], elapsed_ms, len(request.text)
    )

    return PredictResponse(**result)


@app.post("/predict-file", response_model=PredictFileResponse, tags=["Prediction"])
async def predict_file_endpoint(file: UploadFile = File(...)):
    """
    Prediksi apakah konten file ditulis oleh AI atau manusia.

    Format yang didukung: **PDF**, **DOCX**, **TXT**

    **Output:**
    ```json
    {
        "prediction": "AI Generated",
        "confidence": 78.3,
        "probabilities": { "human": 21.7, "ai": 78.3 },
        "document": {
            "filename": "essay.pdf",
            "pages": 5,
            "words": 1200,
            "characters": 7800,
            "ocr_used": false
        }
    }
    ```
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")

    start_time = time.perf_counter()

    try:
        file_bytes = await file.read()
        document   = extract_document(filename=file.filename, file_bytes=file_bytes)
        text       = document["text"].strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Dokumen tidak memiliki teks yang dapat dibaca. "
                       "Pastikan file tidak kosong atau ter-scan tanpa OCR.",
            )

        result = predict(text)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.exception("Artifact error: %s", e)
        raise HTTPException(status_code=503, detail="Model belum tersedia.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing file: %s", e)
        raise HTTPException(status_code=500, detail="Gagal memproses dokumen.")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "Prediction File: %s | %.2f%% | %s | %.1f ms",
        result["prediction"], result["confidence"], file.filename, elapsed_ms
    )

    return PredictFileResponse(
        **result,
        document=DocumentInfo(
            filename=file.filename,
            pages=document.get("pages"),
            words=len(text.split()),
            characters=len(text),
            ocr_used=document.get("ocr_used", False),
        ),
    )


# ─────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."},
    )
