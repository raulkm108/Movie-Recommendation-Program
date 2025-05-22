import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

def main():
    API_KEY = '8e5204ead69396cef3240e5175bb098e'
    input_title = input("🎬 Type the movie title: ").strip()

    recommendations = get_recommendations(input_title, API_KEY)
    if recommendations:
        print(f"\n🔎 Recomendations based on: '{input_title}':")
        for title, score in recommendations:
            print(f"🔹 {title} (similarity: {score:.6f})")
    else:
        print("⚠️ No recommendations found!")

def search_tdmb_movie(title, API_KEY):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.jason().get("results")
        if results:
            return results[0]["id"]
    return None     
    
def get_movie_description(movie_id, API_KEY):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("overview", "")
    return ""

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