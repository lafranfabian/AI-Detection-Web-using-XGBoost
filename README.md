# Installation Guide

## Prerequisites

Pastikan perangkat telah memenuhi kebutuhan berikut sebelum menjalankan aplikasi:

- Python 3.10 atau lebih baru
- Git
- Visual Studio Code (direkomendasikan)
- Browser modern (Google Chrome, Microsoft Edge, atau Mozilla Firefox)

---

# Project Setup

## 1. Clone Repository

Clone repository ke komputer lokal menggunakan perintah berikut:

```bash
git clone https://github.com/lafranfabian/AI-Detection-Web-using-XGBoost.git
```

Masuk ke direktori proyek.

```bash
cd AI-Detection
```

---

## 2. Create Virtual Environment

Buat virtual environment untuk mengisolasi dependency proyek.

```bash
python -m venv .venv
```

---

## 3. Activate Virtual Environment

### Windows (Command Prompt)

```bash
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

Apabila aktivasi berhasil, terminal akan menampilkan awalan berikut.

```text
(.venv)
```

---

## 4. Install Project Dependencies

Install seluruh dependency yang dibutuhkan menggunakan file `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Masuk ke direktori backend.

```bash
cd backend
```

Jalankan aplikasi FastAPI menggunakan Uvicorn.

```bash
python -m uvicorn main:app --reload
```

Apabila server berhasil dijalankan, akan muncul informasi sebagai berikut.

```text
INFO: Uvicorn running on http://127.0.0.1:8000
```

Dokumentasi API dapat diakses melalui:

```
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Buka terminal baru, kemudian masuk ke direktori frontend.

```bash
cd frontend
```

Aplikasi frontend dapat dijalankan menggunakan salah satu metode berikut.

### Option 1 (Recommended)

Menggunakan **Visual Studio Code Live Server**.

- Klik kanan pada file `index.html`
- Pilih **Open with Live Server**

### Option 2

Menggunakan HTTP Server bawaan Python.

```bash
python -m http.server 5500
```

Kemudian buka browser dan akses alamat berikut.

```
http://127.0.0.1:5500
```

---

# Application Workflow

1. Jalankan backend menggunakan FastAPI.
2. Jalankan frontend menggunakan Live Server atau HTTP Server.
3. Buka aplikasi melalui browser.
4. Pilih metode input:
   - Paste Text
   - Upload File
5. Klik tombol **Analyze**.
6. Sistem akan menampilkan hasil prediksi beserta nilai confidence.

---

# Supported File Formats

Aplikasi mendukung analisis dokumen dengan format berikut.

- PDF (.pdf)
- Microsoft Word (.docx)
- Plain Text (.txt)

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | API Information |
| GET | `/health` | API Health Check |
| POST | `/predict` | Text Classification |
| POST | `/predict-file` | Document Classification |

---

# Notes

- Pastikan backend telah berjalan sebelum membuka frontend.
- Pastikan virtual environment telah diaktifkan sebelum menjalankan aplikasi.
- Seluruh dependency harus berhasil diinstal menggunakan `requirements.txt`.
- Browser harus dapat mengakses `http://127.0.0.1:8000` agar frontend dapat berkomunikasi dengan backend.
