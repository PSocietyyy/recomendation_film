# Dokumentasi Analisis Data Film - Sistem Rekomendasi

## Deskripsi Proyek
Proyek ini merupakan implementasi sistem rekomendasi film menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)** dan **Cosine Similarity**. Sistem ini dapat memberikan rekomendasi film berdasarkan kesamaan konten seperti genre, kata kunci, dan deskripsi film.

## Struktur Proyek
```
project/
├── dataset/
│   ├── film/
│   │   ├── tmdb_5000_credits.csv
│   │   └── tmdb_5000_movies.csv
│   └── preprocessing_data/
│       └── processed_data.pkl
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── cosine_similarity.pkl
├── notebooks/
│   ├── exploration_data.ipynb
│   └── preprocessing_data.ipynb
└── main.py
```

## Dataset
Proyek ini menggunakan dataset **TMDb 5000 Movies** yang terdiri dari:
- **tmdb_5000_movies.csv**: Data utama film (4809 baris, 23 kolom)
- **tmdb_5000_credits.csv**: Data kredit film (cast dan crew)

### Kolom Dataset Utama
- `movie_id`: ID unik film
- `title`: Judul film
- `overview`: Deskripsi singkat film
- `genres`: Genre film (format JSON)
- `keywords`: Kata kunci film (format JSON)
- `cast`: Pemeran film (format JSON)
- `crew`: Kru film (format JSON)

## Tahapan Preprocessing Data

### 1. Eksplorasi Data (`exploration_data.ipynb`)

#### Import Library
```python
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
```

#### Memuat Dataset
```python
credits = pd.read_csv("../dataset/film/tmdb_5000_credits.csv")
movies = pd.read_csv("../dataset/film/tmdb_5000_movies.csv")
```

#### Penggabungan Dataset
Dataset `movies` dan `credits` digabungkan berdasarkan `movie_id` untuk mendapatkan informasi lengkap tentang setiap film.

```python
movies = movies.merge(credits, on='movie_id')
```

### 2. Pembersihan Data (`preprocessing_data.ipynb`)

#### Seleksi Fitur
Memilih fitur yang relevan untuk sistem rekomendasi dan menghapus fitur yang tidak penting:

**Fitur yang Dipertahankan:**
- `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`

**Fitur yang Dihapus:**
- `budget`, `homepage`, `original_language`, `original_title`, `popularity`
- `production_companies`, `production_countries`, `revenue`, `runtime`
- `spoken_language`, `tagline`, `status`, `vote_average`, `vote_count`

#### Penanganan Data Kosong
```python
# Menghitung nilai kosong
movies.isnull().sum()

# Menghapus baris dengan data kosong
movies.dropna(inplace=True)

# Memeriksa duplikasi
movies.duplicated().sum()
```

#### Konversi Format Data JSON

**Fungsi untuk Mengkonversi Genres dan Keywords:**
```python
import ast

def convert_genres_and_keyword(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l
```

**Fungsi untuk Mengkonversi Cast (3 aktor utama):**
```python
def convert_cast(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l
```

**Fungsi untuk Mengambil Sutradara:**
```python
def fetch_director(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == "Director":
            l.append(i['name'])
        else:
            break
    return l
```

#### Preprocessing Teks
```python
# Memecah overview menjadi list kata
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Menghapus spasi dari semua teks
def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))
    return l

# Mengaplikasikan preprocessing pada semua kolom teks
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
```

#### Penggabungan Konten
```python
def create_content(row):
    content = row['overview'] + row['genres'] + row['keywords'] + row['cast'] + row['crew']
    return " ".join(content)

movies['content'] = movies.apply(create_content, axis=1)
```

### 3. Vectorization dengan TF-IDF

#### Implementasi TF-IDF Vectorizer
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matriks = tfidf_vectorizer.fit_transform(movies['content'])
```

**Hasil Vectorization:**
- Dimensi matriks: (4806, 27452)
- 4806 film dengan 27452 fitur unik

### 4. Sistem Rekomendasi (`main.py`)

#### Struktur Kelas

**MovieProcessor (Kelas Dasar):**
```python
class MovieProcessor:
    """Kelas untuk memproses data film, termasuk preprocessing teks."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.movies = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def load_data(self):
        """Memuat dataset film dan membersihkan data."""
        
    def save_processed_data(self, data_path):
        """Menyimpan data setelah preprocessing."""
        
    def get_movies(self):
        """Mengembalikan data film yang telah diproses."""
```

**MovieRecommender (Kelas Turunan):**
```python
class MovieRecommender(MovieProcessor):
    """Kelas untuk sistem rekomendasi film berbasis TF-IDF & Cosine Similarity."""
    
    def train_model(self):
        """Melatih model TF-IDF dan menyimpan hasilnya."""
        
    def load_model(self):
        """Memuat model TF-IDF dan cosine similarity."""
        
    def recommend_movies(self, title, top_n=3):
        """Memberikan rekomendasi film berdasarkan judul."""
        
    def visualize_recommendations(self, title, top_n=3):
        """Menampilkan hasil rekomendasi dalam bentuk grafik."""
```

## Algoritma Rekomendasi

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- **Term Frequency (TF)**: Frekuensi kemunculan kata dalam dokumen
- **Inverse Document Frequency (IDF)**: Kebalikan dari frekuensi dokumen yang mengandung kata tersebut
- **Formula**: TF-IDF = TF × IDF

### 2. Cosine Similarity
- Mengukur kesamaan antara dua vektor dalam ruang multidimensi
- Nilai berkisar antara 0 (tidak mirip) hingga 1 (sangat mirip)
- **Formula**: cosine_similarity = (A · B) / (||A|| × ||B||)

## Fitur-Fitur Sistem

### 1. Preprocessing Data
- Konversi format JSON ke list Python
- Penghapusan spasi dan standardisasi teks
- Penggabungan multiple fitur menjadi satu konten

### 2. Pelatihan Model
- Vectorization menggunakan TF-IDF
- Perhitungan cosine similarity matrix
- Penyimpanan model untuk penggunaan selanjutnya

### 3. Rekomendasi Film
- Input: Judul film
- Output: List film yang mirip dengan skor similarity
- Dapat disesuaikan jumlah rekomendasi (parameter `top_n`)

### 4. Visualisasi
- Grafik hasil rekomendasi
- Analisis distribusi similarity score

## Cara Penggunaan

### 1. Preprocessing Data
```python
# Menjalankan eksplorasi data
jupyter notebook exploration_data.ipynb

# Menjalankan preprocessing
jupyter notebook preprocessing_data.ipynb
```

### 2. Menggunakan Sistem Rekomendasi
```python
# Inisialisasi recommender
recommender = MovieRecommender(
    dataset_path="dataset/film/tmdb_5000_movies.csv",
    model_path="models/tfidf_vectorizer.pkl",
    similarity_path="models/cosine_similarity.pkl"
)

# Melatih model (jika belum ada)
recommender.train_model()

# Atau memuat model yang sudah ada
recommender.load_model()

# Mendapatkan rekomendasi
recommendations = recommender.recommend_movies("Avatar", top_n=5)
print(recommendations)
```

## Hasil dan Evaluasi

### Kualitas Data
- **Total Film**: 4806 film (setelah pembersihan)
- **Missing Values**: Ditangani dengan penghapusan baris
- **Duplicates**: Tidak ada duplikasi data

### Performa Model
- **Dimensi Vektor**: 27452 fitur unik
- **Metode Similarity**: Cosine Similarity
- **Akurasi**: Berdasarkan relevansi konten (genre, keywords, cast, overview)

## Kelebihan dan Kekurangan

### Kelebihan ✅
- **Content-based filtering**: Tidak memerlukan data user interaction
- **Cold start problem**: Dapat memberikan rekomendasi untuk film baru
- **Interpretable**: Hasil rekomendasi dapat dijelaskan berdasarkan kesamaan konten
- **Scalable**: Efisien untuk dataset besar

### Kekurangan ❌
- **Limited diversity**: Cenderung merekomendasikan film yang sangat mirip
- **No user preference**: Tidak mempertimbangkan preferensi personal user
- **Text dependency**: Kualitas rekomendasi bergantung pada kualitas deskripsi teks
- **No popularity factor**: Tidak mempertimbangkan popularitas atau rating film

## Optimasi dan Pengembangan Lanjutan

### 1. Perbaikan Preprocessing
- Implementasi stemming/lemmatization
- Penanganan synonyms dan word embeddings
- Feature engineering yang lebih canggih

### 2. Model Enhancement
- Hybrid recommendation (content + collaborative filtering)
- Deep learning approaches (neural networks)
- Ensemble methods

### 3. Evaluasi Model
- Implementasi metrics evaluasi (precision, recall, F1-score)
- A/B testing untuk mengukur user satisfaction
- Cross-validation untuk model stability

## Dependencies

### Library Python yang Digunakan
```python
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.3.0
```

### Instalasi
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Struktur File Output

### Model Files
- `tfidf_vectorizer.pkl`: Model TF-IDF vectorizer
- `cosine_similarity.pkl`: Matriks cosine similarity
- `processed_data.pkl`: Data film yang telah dipreprocessing

### Ukuran File
- Dataset asli: ~3-5 MB
- Model TF-IDF: ~10-15 MB
- Cosine similarity matrix: ~200-300 MB (tergantung ukuran dataset)

## Troubleshooting

### Common Issues
1. **Memory Error**: Cosine similarity matrix memerlukan memory yang besar
   - **Solusi**: Gunakan sparse matrix atau batch processing

2. **JSON Parsing Error**: Format data JSON yang tidak konsisten
   - **Solusi**: Tambahkan error handling dalam fungsi konversi

3. **Performance Issue**: Waktu komputasi yang lama
   - **Solusi**: Implementasi caching dan optimasi algoritma

## Kesimpulan
Sistem rekomendasi film ini menggunakan pendekatan content-based filtering yang efektif untuk memberikan rekomendasi berdasarkan kesamaan konten. Meskipun memiliki beberapa keterbatasan, sistem ini dapat menjadi foundation yang baik untuk pengembangan sistem rekomendasi yang lebih kompleks.
