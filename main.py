import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
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
        results = response.json().get("results")
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
    descriptions = []
    titles = []

    for movie in candidate_movies:
        overview = movie.get("overview")
        title = movie.get("title")
        if overview and title:
            descriptions.append(overview)
            titles.append(title)
    
    if not descriptions:
        print ("❌ No movie with description avaiable")
        return []

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=3)
    tfidf_matrix = vectorizer.fit_transform([input_description] + descriptions)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    cosine_similarities = np.round(cosine_similarities, 6)
    
    top_indexes = cosine_similarities.argsort()[::-1][:n]
    return [(titles[i], cosine_similarities[i]) for i in top_indexes]

if __name__ == "__main__":
    main()