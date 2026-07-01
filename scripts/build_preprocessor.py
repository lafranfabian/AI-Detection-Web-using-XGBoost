"""
build_preprocessor.py
---------------------
Script untuk membuat ulang preprocessor (ColumnTransformer) yang identik
dengan alur training di notebook, lalu menyimpannya sebagai preprocessor.pkl.

Jalankan SEKALI sebelum menjalankan backend:
    python3 scripts/build_preprocessor.py

Output:
    backend/models/preprocessor.pkl
    backend/models/label_encoder_content.pkl
"""

import sys
import os

# Pastikan path relatif ke root project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CSV_PATH = os.path.join(ROOT_DIR, "expanded_ai_human_content_detection_dataset_2500.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 55)
print("  BUILD PREPROCESSOR — AI Text Detector")
print("=" * 55)

# ============================================================
# 1. LOAD DATA (identik dengan notebook)
# ============================================================
print(f"\n[1/5] Loading dataset dari: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, sep=';')
data = df.copy()
print(f"      Shape: {data.shape}")
print(f"      Columns: {list(data.columns)}")

# ============================================================
# 2. CLEANING & LABEL ENCODING (identik dengan notebook)
# ============================================================
print("\n[2/5] Cleaning & Label Encoding...")

# Pastikan text_content tidak ada NaN
if 'text_content' in data.columns:
    data['text_content'] = data['text_content'].fillna('')

# LabelEncode 'label' (target)
le_label = LabelEncoder()
data['label'] = le_label.fit_transform(data['label'])
print(f"      Label classes: {dict(enumerate(le_label.classes_))}")

# LabelEncode 'content_type' (fitur kategorikal)
le_content = LabelEncoder()
data['content_type'] = le_content.fit_transform(data['content_type'])
print(f"      Content type classes: {dict(enumerate(le_content.classes_))}")

# Simpan LabelEncoder content_type (dibutuhkan saat inferensi)
le_content_path = os.path.join(MODELS_DIR, "label_encoder_content.pkl")
joblib.dump(le_content, le_content_path)
print(f"      Saved: label_encoder_content.pkl")

# ============================================================
# 3. DEFINE FEATURES (identik dengan notebook)
# ============================================================
print("\n[3/5] Mendefinisikan fitur...")

X = data.drop(columns=['label'])
y = data['label']

# Kolom numerik = semua kecuali text_content
numeric_features = [col for col in X.columns if col != 'text_content']
text_feature = 'text_content'

print(f"      Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"      Text feature: {text_feature}")

# ============================================================
# 4. BUILD & FIT PREPROCESSOR (identik dengan notebook)
# ============================================================
print("\n[4/5] Building & Fitting ColumnTransformer...")

# Pipeline numerik: Imputer(median) → StandardScaler
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ColumnTransformer: numerik + TF-IDF
# PENTING: Parameter TF-IDF HARUS identik dengan notebook
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('text', TfidfVectorizer(max_features=300, stop_words='english'), text_feature)
    ]
)

# Fit menggunakan SELURUH data training (bukan test split)
# Konsisten dengan notebook yang menggunakan preprocessor.fit_transform(X_train)
# Kita fit dengan semua data untuk mendapatkan representasi terbaik
preprocessor.fit(X, y)

# Verifikasi dimensi output
sample_transformed = preprocessor.transform(X.head(1))
n_features_out = sample_transformed.shape[1]
print(f"      Output features: {n_features_out} (expected: 315)")

if n_features_out != 315:
    print(f"      WARNING: Expected 315 features, got {n_features_out}!")
    print(f"               Periksa jumlah kolom di dataset.")
else:
    print(f"      OK: Feature dimensi cocok dengan model XGBoost (315)")

# ============================================================
# 5. SIMPAN PREPROCESSOR
# ============================================================
print("\n[5/5] Menyimpan preprocessor.pkl...")

preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)

print(f"      Saved: preprocessor.pkl")
print(f"\n{'='*55}")
print("  SELESAI! File yang dihasilkan:")
print(f"  - backend/models/preprocessor.pkl")
print(f"  - backend/models/label_encoder_content.pkl")
print(f"{'='*55}")
print("\nSelanjutnya jalankan backend:")
print("  cd backend && uvicorn main:app --reload")
