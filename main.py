import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("⚠️ File 'movies.csv' not found. Please make sure it is in the same directory.")
    exit()

required_columns = {'title', 'description'}
if not required_columns.issubset(movies.columns):
    print("⚠️ The dataset must contain the columns: 'title' and 'description'.")
    exit()
movies['description'] = movies['description'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(movies['description'])

# print("✅ TF-IDF matrix created. Shape:", tfidf_matrix.shape) >> printing the amount of descriptions and the amount of "important" vectors

movie_title = input("\n🎬 Enter a movie title: ").strip().lower()


if movie_title not in movies['title'].str.lower().values:
    print(f"❌ Movie '{movie_title}' not found in the dataset.")
    exit()

movie_index = movies[movies['title'].str.lower() == movie_title].index[0]

cosine_similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()

similar_indices = cosine_similarities.argsort()[::-1][1:6]

print("\n🎯 Recommended movies:")
for idx in similar_indices:
    print(f"🔹 {movies.iloc[idx]['title']}")