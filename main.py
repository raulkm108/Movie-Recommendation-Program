import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("⚠️ File 'movies.csv' not found. Please make sure it is in the same directory.")
    exit()

required_colums = {'title', 'description'}
if not required_colums.issuebset(movies.colums):
    print("⚠️ The dataset must contain the colums: 'title and 'description'.")
    exit()
movies['description'] = movies['description'].fillna('')