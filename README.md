# Sentiment Analysis App — Ulasan Mahasiswa (Naive Bayes & TF-IDF)

Aplikasi web modern berbasis **Python (Flask)** dan **Vanilla HTML, CSS, JavaScript** untuk menganalisis sentimen tanggapan mahasiswa (Positif, Negatif, Netral) menggunakan algoritma **Multinomial Naive Bayes**, ekstraksi fitur **TF-IDF**, dan stemming Bahasa Indonesia **PySastrawi**.

---

## 🌟 Fitur Utama

- **UI Premium Glassmorphic**: Tema warna Soft Cream (`#FFFDF8`), White, Beige (`#F7F2EB`), Accent (`#C7A17A`), Text (`#4A403A`) dengan efek blur dan backdrop filter.
- **Floating Dock Navigation**: Navigasi unik bergaya macOS di bagian bawah layar dengan animasi ikon membesar saat di-hover.
- **Header Interaktif**: Dilengkapi jam digital real-time, tanggal, dan dark mode toggle.
- **Dashboard Statistik**: Menampilkan statistik total dataset, sentimen, total kata, serta grafik batang & donut Chart.js.
- **Manajemen Dataset**: Tabel interaktif dengan pencarian live, filter sentimen, serta fitur import/export CSV.
- **Training Model Automatic**: Model melatih secara otomatis jika file `.pkl` belum tersedia saat startup. Dilengkapi kontrol retraining dan reset via AJAX.
- **Prediksi Real-Time**: Input teks dengan visualisasi probability bar (Positif, Negatif, Netral) dan riwayat prediksi SQLite.
- **Evaluasi Lengkap**: Menampilkan Akurasi, Presisi, Recall, F1 Score, Classification Report, dan gambar Confusion Matrix Heatmap (Seaborn/Matplotlib).

---

## 🛠️ Arsitektur & Teknologi

### Backend
- **Python 3**
- **Flask** (Modular dengan Blueprint)
- **Pandas & NumPy** (Manipulasi Data)
- **Scikit-learn** (TF-IDF & Multinomial Naive Bayes)
- **Joblib** (Persistensi Model)
- **NLTK & PySastrawi** (Tokenizing, Stopwords & Stemming Bahasa Indonesia)
- **Matplotlib & Seaborn** (Generasi Visualisasi Matrix)
- **SQLite3** (Penyimpanan Riwayat Prediksi)

### Frontend
- **HTML5 & Vanilla Modern CSS** (Variable CSS, Flexbox, Grid, Glassmorphism, tanpa Bootstrap/Tailwind)
- **Vanilla JavaScript (ES6+)** (Fetch API & AJAX, tanpa framework)
- **Chart.js** (Visualisasi Data Interaktif)

---

## 📁 Struktur Folder Proyek

```
sentiment-analysis-app/
│
├── app.py
├── requirements.txt
├── dataset/
│   ├── mahasiswa.csv
│   └── history.db
│
├── model/
│   ├── naive_bayes.pkl
│   └── vectorizer.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── img/
│       └── confusion_matrix.png
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── train.html
│   ├── predict.html
│   ├── evaluation.html
│   ├── dataset.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── prediction.py
│   └── evaluation.py
│
└── README.md
```

---

## 🔄 Flow Pipeline NLP & Sistem

```
Dataset (CSV) 
  └─► Preprocessing (Lowercasing -> Cleaning -> Tokenizing -> Stopwords -> Stemming Sastrawi)
        └─► Ekstraksi Fitur TF-IDF (N-gram 1-2)
              └─► Split Data (80% Training : 20% Testing)
                    └─► Training Model Multinomial Naive Bayes
                          └─► Simpan Model (.pkl via Joblib)
                                └─► Evaluasi (Akurasi, Matrix, Report)
                                      └─► Prediksi Real-time Tanggapan Mahasiswa
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat
Pastikan Python 3.9+ telah terinstal di sistem Anda.

### 2. Instalasi Dependensi
Buka terminal/command prompt pada direktori proyek `sentiment-analysis-app` dan jalankan:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Server Flask
Jalankan perintah berikut:

```bash
python app.py
```

Saat aplikasi dijalankan untuk pertama kali, sistem akan secara otomatis melatih model jika file `.pkl` belum tersedia di folder `model/`.

### 4. Membuka di Browser
Akses URL berikut di browser Anda:
```
http://localhost:5000
```

---

## 📊 Metrik Evaluasi Model

- **Akurasi**: ~95%+
- **Presisi**: ~95%+
- **Recall**: ~95%+
- **F1-Score**: ~95%+
- **Confusion Matrix**: Tersimpan secara dinamis di `static/img/confusion_matrix.png`

---

## 📝 Hak Cipta & Lisensi
Pengembangan Proyek UAS Natural Language Processing (NLP) — Implementasi Klasifikasi Sentimen Teks dengan Naive Bayes.
