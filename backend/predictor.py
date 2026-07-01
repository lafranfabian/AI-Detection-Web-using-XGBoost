"""
predictor.py
------------
Modul inferensi utama. Memuat model XGBoost dan preprocessor,
lalu menghasilkan prediksi dari teks mentah.

Pipeline inferensi:
    text (str)
        → feature_extractor.extract_features()   → dict (15 fitur + text_content)
        → pd.DataFrame                            → DataFrame (1 row, 16 cols)
        → preprocessor.transform()               → sparse/dense matrix (315 features)
        → xgb_model.predict_proba()              → [P(Human), P(AI)]
        → result dict                             → {"prediction": ..., "confidence": ...}
"""

import os
import logging
import warnings
from functools import lru_cache
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from feature_extractor import extract_features

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BACKEND_DIR, "models")
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

MODEL_PATH = os.path.join(_PROJECT_ROOT, "aiidentification.pkl")
PREPROCESSOR_PATH = os.path.join(_MODELS_DIR, "preprocessor.pkl")
LE_CONTENT_PATH = os.path.join(_MODELS_DIR, "label_encoder_content.pkl")


# ─────────────────────────────────────────────────────────────
# LAZY LOADING (singleton via lru_cache)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_artifacts():
    """
    Load semua artefak ML sekali saja (lazy + cached).
    Melempar RuntimeError jika file tidak ditemukan.
    """
    missing = []
    for name, path in [
        ("XGBoost model", MODEL_PATH),
        ("Preprocessor", PREPROCESSOR_PATH),
        ("LabelEncoder content_type", LE_CONTENT_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"  - {name}: {path}")

    if missing:
        raise RuntimeError(
            "Artifact tidak ditemukan. Jalankan dulu:\n"
            "  python3 scripts/build_preprocessor.py\n\n"
            "File yang hilang:\n" + "\n".join(missing)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        le_content = joblib.load(LE_CONTENT_PATH)

    logger.info("Artifacts loaded: model=%s, preprocessor=%s, le_content=%s",
                MODEL_PATH, PREPROCESSOR_PATH, LE_CONTENT_PATH)
    return model, preprocessor, le_content


# ─────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────

# Urutan kolom HARUS identik dengan urutan X = data.drop(columns=['label'])
# yang ada di notebook. Diverifikasi dari CSV asli:
# ['text_content', 'content_type', 'word_count', 'character_count',
#  'sentence_count', 'lexical_diversity', 'avg_sentence_length',
#  'avg_word_length', 'punctuation_ratio', 'flesch_reading_ease',
#  'gunning_fog_index', 'grammar_errors', 'passive_voice_ratio',
#  'predictability_score', 'burstiness', 'sentiment_score']
_COLUMN_ORDER = [
    'text_content',
    'content_type',
    'word_count',
    'character_count',
    'sentence_count',
    'lexical_diversity',
    'avg_sentence_length',
    'avg_word_length',
    'punctuation_ratio',
    'flesch_reading_ease',
    'gunning_fog_index',
    'grammar_errors',
    'passive_voice_ratio',
    'predictability_score',
    'burstiness',
    'sentiment_score',
]


def predict(text: str) -> Dict[str, Any]:
    """
    Predict apakah teks ditulis oleh AI atau manusia.

    Args:
        text: Teks mentah dari user.

    Returns:
        Dict dengan keys:
            - prediction (str): "AI Generated" atau "Human Original"
            - confidence (float): Persentase keyakinan model (0–100)
            - label_index (int): 0 = Human, 1 = AI (raw model output)
            - probabilities (dict): P(Human) dan P(AI) mentah
    """
    model, preprocessor, le_content = _load_artifacts()

    # ── Step 1: Hitung fitur linguistik dari teks ──────────────
    features = extract_features(text)

    # ── Step 2: Tentukan content_type → encoded integer ────────
    # Kita gunakan kelas pertama yang tersedia dari LabelEncoder.
    # Semua kelas yang pernah dilihat saat training:
    available_classes = list(le_content.classes_)
    # Default: "blog_post" jika ada, fallback ke kelas pertama
    default_class = 'blog_post' if 'blog_post' in available_classes else available_classes[0]
    features['content_type'] = int(le_content.transform([default_class])[0])

    # ── Step 3: Buat DataFrame dengan urutan kolom yang benar ──
    df_input = pd.DataFrame([features], columns=_COLUMN_ORDER)

    # ── Step 4: Transform menggunakan preprocessor yang sudah di-fit ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_transformed = preprocessor.transform(df_input)

    # ── Step 5: Predict ────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = model.predict_proba(X_transformed)[0]  # [P(class_0), P(class_1)]
        pred_class = int(model.predict(X_transformed)[0])

    # ── Step 6: Map label ──────────────────────────────────────
    # Dari notebook: 1 = AI, 0 = Human (hasil LabelEncoder pada kolom 'label')
    # Konfirmasi dari kode inference di notebook:
    #   status_deteksi = "AI" if prediksi_angka[0] == 1 else "MANUSIA"
    p_human = float(proba[0])
    p_ai = float(proba[1])

    if pred_class == 1:
        prediction_label = "AI Generated"
        confidence = round(p_ai * 100, 2)
    else:
        prediction_label = "Human Original"
        confidence = round(p_human * 100, 2)

    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "probabilities": {
            "human": round(p_human * 100, 2),
            "ai": round(p_ai * 100, 2),
        }
    }
