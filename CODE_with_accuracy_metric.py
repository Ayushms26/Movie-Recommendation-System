import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast
import pickle

# Load datasets
movies = pd.read_csv('/content/tmdb_5000_movies.csv.zip')
credits = pd.read_csv('/content/tmdb_5000_credits.csv.zip')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['genres', 'id', 'keywords', 'overview', 'title', 'cast', 'crew']]

# Drop rows with missing values
movies = movies.dropna()

# Extract genres
def genres(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(genres)

# Extract keywords
def keywords(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['keywords'] = movies['keywords'].apply(keywords)

# Extract top 3 cast members
def cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)][:3]

movies['cast'] = movies['cast'].apply(cast)

# Extract director from crew
def crew(obj):
    return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director'][:1]

movies['crew'] = movies['crew'].apply(crew)

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces in names
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])

# Combine all features into tags
movies['tags'] = movies['cast'] + movies['crew'] + movies['genres'] + movies['keywords'] + movies['overview']
movies = movies.drop(['genres', 'keywords', 'overview', 'cast', 'crew'], axis=1)

# Convert tags list to string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Stemming
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(i) for i in text.split()])

movies['tags'] = movies['tags'].apply(stem)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Save data for later use
pickle.dump(movies.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Evaluate system performance
def evaluate_system():
    # Example: Test recommendations for a few movies
    test_movies = ['Avatar', 'The Dark Knight', 'Inception']
    hit_count = 0
    total_recommendations = 0

    for movie in test_movies:
        recommendations = recommend(movie)
        print(f"Recommendations for {movie}: {recommendations}")
        total_recommendations += len(recommendations)
        hit_count += 1 if movie in recommendations else 0

    # Hit Rate@5
    hit_rate = hit_count / len(test_movies)
    print("Hit Rate@5: ",hit_rate)

    # F1-score (assuming binary relevance)
    precision = hit_count / total_recommendations
    recall = hit_count / len(test_movies)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("F1-score: ",f1_score)

    # Diversity Score (unique recommendations across all test movies)
    all_recommendations = [recommend(movie) for movie in test_movies]
    unique_recommendations = set([item for sublist in all_recommendations for item in sublist])
    diversity_score = len(unique_recommendations) / (len(test_movies) * 5)  # 5 recommendations per movie
    print("Diversity Score: ",diversity_score)

    # Personalization Score (average pairwise dissimilarity between user recommendations)
    personalization_score = 1 - np.mean([cosine_similarity([vectors[movies[movies['title'] == movie].index[0]] for movie in test_movies])])
    print("Personalization Score: ",personalization_score)

# Run evaluation
evaluate_system()