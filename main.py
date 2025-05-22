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

def get_popular_movies(API_KEY, pages=10):
    movies = []
    for page in range (1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular"
        params = {"api_key": API_KEY, "page": page}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            movies.extend(results)
    return movies

def get_recommendations(input_title, API_KEY, n=5):
    movie_id = search_tdmb_movie(input_title, API_KEY)
    if not movie_id:
        print("❌ Movie not Found!")
        return []
    
    input_description = get_movie_description(movie_id, API_KEY)
    if not input_description:
        print("❌ Movie description not Found!")
        return []
    
    candidate_movies = get_popular_movies(API_KEY)
    description = []
    titles = []
    





    cosine_similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    cosine_similarities = np.round(cosine_similarities, 6)
    similar_index = cosine_similarities.argsort()[::-1][1:n + 1]
    return [(movies.iloc[i]['title'], cosine_similarities[i]) for i in similar_index]

if __name__ == "__main__":
    main()