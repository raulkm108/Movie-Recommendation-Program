import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

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

def load_data(path_csv):
    try:
        movies = pd.read_csv(path_csv)
    except FileNotFoundError:
        print(f"⚠️ File '{path_csv}' not found.")
        exit()
    if not {'title', 'description'}.issubset(movies.columns):
        print(f"⚠️ The dataset must contain 'title' and 'description'.")
        exit()
    movies['description'] = movies['description'].fillna('')
    return movies

def suggest_similar_title(input_title, title_list):
    matches = difflib.get_close_matches(input_title, title_list, n=5, cutoff=0.5)
    if matches:
        print("\n Movie not found, Did you mean one of these?")
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match}")
        try:
            choice = int(input("\nEnter the number of the correct movie (0 to cancel): "))
            if 1 <= choice <= len(matches):
                return matches[choice - 1]
        except ValueError:
            pass
    return None

def get_recommendations(movie_index, tfidf_matrix, movies, n=5):
    cosine_similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    cosine_similarities = np.round(cosine_similarities, 6)
    similar_index = cosine_similarities.argsort()[::-1][1:n + 1]
    return [(movies.iloc[i]['title'], cosine_similarities[i]) for i in similar_index]

if __name__ == "__main__":
    main()