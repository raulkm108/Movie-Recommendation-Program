import requests

API_KEY = '8e5204ead69396cef3240e5175bb098e'
movie_title = 'inception'
url = f'https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}'

response = requests.get(url)
data = response.json()

print(f"{response}")

print(f"\n{data}")

if data['results']:
    movie = data['results'][0]
    print(f"Title: {movie['title']}")
    print(f"Description: {movie['overview']}")
else:
    print("Movie not Found.")