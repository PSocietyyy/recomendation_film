
<h1 align="center">🎬 Movie Recommendation System</h1>
<p align="center">Proyek Machine Learning untuk merekomendasikan film menggunakan algoritma dasar seperti <strong>Linear Regression</strong>, <strong>Logistic Regression</strong>, dan <strong>K-Nearest Neighbors</strong>.</p>
<p align="center"><i>A Machine Learning project for recommending movies using basic algorithms like <strong>Linear Regression</strong>, <strong>Logistic Regression</strong>, and <strong>K-Nearest Neighbors</strong>.</i></p>

---

## 📂 Struktur Folder / Project Structure


```notebook/
├── exploration\_data.ipynb       # Eksplorasi data dan visualisasi / Data exploration & visualization
├── preprocessing\_data.ipynb     # Pembersihan & transformasi data / Data preprocessing
├── training.ipynb               # Training model dan evaluasi / Model training & evaluation
```

---

## 🧠 Algoritma yang Digunakan / Algorithms Used

### 1. Linear Regression
- **ID:** Digunakan untuk memprediksi skor rating film berdasarkan fitur numerik seperti durasi, jumlah penonton, dan tahun rilis.
- **EN:** Used to predict movie rating scores based on numeric features like duration, viewer count, and release year.

### 2. Logistic Regression
- **ID:** Digunakan untuk klasifikasi biner, contohnya apakah pengguna akan menyukai sebuah film atau tidak.
- **EN:** Used for binary classification, for example whether a user will like a movie or not.

### 3. K-Nearest Neighbors (KNN)
- **ID:** Digunakan untuk merekomendasikan film berdasarkan kemiripan fitur antar film atau antar pengguna.
- **EN:** Used to recommend movies based on similarity between movie features or between users.

---

## 📊 Alur Kerja / Workflow

### 📌 1. Eksplorasi Data
- Membaca dataset film mentah (misal: `movies.csv`)
- Visualisasi distribusi genre, rating, dan popularitas
- Pengecekan data hilang dan anomali

### 📌 2. Pra-pemrosesan Data
- Pembersihan data: menghapus nilai kosong, encoding kategori (genre, sutradara, dll)
- Normalisasi dan feature scaling

### 📌 3. Training dan Evaluasi
- Melatih model Machine Learning menggunakan dataset terproses
- Mengevaluasi akurasi dan performa menggunakan metrik (MAE, RMSE, F1-Score, dll)
- Visualisasi hasil prediksi dan rekomendasi

---

## 🚀 Cara Menjalankan / How to Run

1. Clone repo:
```bash
   git clone https://github.com/username/movie-recommender.git
   cd movie-recommender/notebook
```

2. Jalankan `exploration_data.ipynb` → lalu `preprocessing_data.ipynb` → terakhir `training.ipynb`.

3. Pastikan library Python seperti `pandas`, `scikit-learn`, `matplotlib`, dan `seaborn` sudah terinstal.

---

## 🏷️ Teknologi yang Digunakan / Tech Stack

* Python
* Jupyter Notebook
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn (untuk ML)
* Visualisasi interaktif (opsional: Plotly)

---

## ✨ Output

* Model ML yang mampu memprediksi atau merekomendasikan film
* Grafik evaluasi (akurasi, confusion matrix, MAE, dll)
* Sistem sederhana untuk rekomendasi film berdasarkan input pengguna

---

## 📌 Catatan

* Ini adalah proyek pembelajaran. Akurasi dan performa bisa ditingkatkan lebih lanjut.

## 🧑‍💻 Kontributor / Contributors

* **Ferdiansyah Pratama** - *Top Contributor & Developer* 🚀
