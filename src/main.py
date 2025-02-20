import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieProcessor:
    """Kelas untuk memproses data film, termasuk preprocessing teks."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.movies = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def load_data(self):
        """Memuat dataset film dan membersihkan data."""
        self.movies = pd.read_csv(self.dataset_path)
        self.movies['content'] = (
            self.movies['overview'].fillna('') + " " +
            self.movies['genres'].fillna('') + " " +
            self.movies['keywords'].fillna('')
        )
    
    def save_processed_data(self, data_path):
        """Menyimpan data setelah preprocessing."""
        joblib.dump(self.movies, data_path)
    
    def get_movies(self):
        """Mengembalikan data film yang telah diproses."""
        return self.movies

class MovieRecommender(MovieProcessor):
    """Kelas untuk sistem rekomendasi film berbasis TF-IDF & Cosine Similarity."""
    
    def __init__(self, dataset_path, model_path, similarity_path):
        super().__init__(dataset_path)
        self.model_path = model_path
        self.similarity_path = similarity_path
        self.cosine_sim = None
    
    def train_model(self):
        """Melatih model TF-IDF dan menyimpan hasilnya."""
        self.load_data()
        tfidf_matrix = self.vectorizer.fit_transform(self.movies['content'])
        
        # Simpan model dan vektorisasi
        joblib.dump(self.vectorizer, self.model_path)
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        joblib.dump(self.cosine_sim, self.similarity_path)
    
    def load_model(self):
        """Memuat model TF-IDF dan cosine similarity."""
        self.load_data()
        self.vectorizer = joblib.load(self.model_path)
        self.cosine_sim = joblib.load(self.similarity_path)
    
    def recommend_movies(self, title, top_n=3):
        """Memberikan rekomendasi film berdasarkan judul."""
        if title not in self.movies['title'].values:
            return "Film tidak ditemukan dalam database."
        
        idx = self.movies[self.movies['title'] == title].index[0]
        similarity_scores = list(enumerate(self.cosine_sim[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommended_movies = [(self.movies.iloc[index]['title'], score) for index, score in similarity_scores]
        return recommended_movies

    def visualize_recommendations(self, title, top_n=3):
        """Menampilkan hasil rekomendasi dalam bentuk grafik."""
        recommended = self.recommend_movies(title, top_n)
        
        if isinstance(recommended, str):
            print(recommended)
            return
        
        recommended_titles = [movie[0] for movie in recommended]
        recommended_scores = [movie[1] for movie in recommended]

        # Heatmap Cosine Similarity
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cosine_sim[:10, :10], annot=True, cmap="coolwarm",
                    xticklabels=self.movies['title'][:10], yticklabels=self.movies['title'][:10])
        plt.title("Cosine Similarity antar Film")
        plt.show()

        # Barplot Rekomendasi
        plt.figure(figsize=(8, 4))
        sns.barplot(x=recommended_scores, y=recommended_titles, palette="viridis")
        plt.xlabel("Similarity Score")
        plt.ylabel("Recommended Movies")
        plt.title(f"Rekomendasi Film untuk '{title}'")
        plt.show()

if __name__ == "__main__":
    recommender = MovieRecommender(
        dataset_path="../dataset/clean_film/cleaned_movies.csv",
        model_path="../models/tfidf_vectorizer.pkl",
        similarity_path="../models/cosine_similarity.pkl"
    )

    recommender.visualize_recommendations("Interstellar", top_n=3)
