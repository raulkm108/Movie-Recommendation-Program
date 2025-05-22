import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    movies = load_data("movies.csv")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies['description'])

    input_title = input("\n Enter a movie title: ").strip()
    titles = movies['title'].tolist()

    if input_title.lower() not in [t.lower() for t in titles]:
        suggestion = suggest_similar_title(input_title, titles)
        if suggestion:
            input_title = suggestion
        else:
            print("❌ No match found or cancelled.")
            return

movie_index = movies[movies['title'].str.lower() == input_title.lower()].index[0]
recommendation = get_recommendations(movie_index, tfidf_matrix, movies)

print(f"\n Recommended movies based on '{input_title}':")
for title, score in recommendation:
    print(f" {title} (score: {score:.6f})")


def load_data(path_csv)
    try:
        movies = pd.read_csv(path_csv)
    except FileNotFoundError:
        print(f"⚠️ File '{path_csv}' not found.")
        exit()
    if not {'title', 'description'}.issubset(movies.columns):
        print(f"⚠️ The dataset must contain 'title' and 'description'.")
        exit()
    movies['description'] = movies['description'].fillna('')

tfidf_matrix = vectorizer.fit_transform(movies['description'])

# print("✅ TF-IDF matrix created. Shape:", tfidf_matrix.shape) >> printing the amount of descriptions and the amount of "important" vectors

movie_title = input("\n🎬 Enter a movie title: ").strip().lower()


if movie_title not in movies['title'].str.lower().values:
    print(f"❌ Movie '{movie_title}' not found in the dataset.")
    exit()

movie_index = movies[movies['title'].str.lower() == movie_title].index[0]

cosine_similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
cosine_similarities = np.round(cosine_similarities, 6)

similar_indices = cosine_similarities.argsort()[::-1][1:6]

print("\n🎯 Recommended movies:")
for idx in similar_indices:
    title = movies.iloc[idx]['title']
    score = cosine_similarities[idx]
    print(f"🔹 {title} (score: {score:.6f})")