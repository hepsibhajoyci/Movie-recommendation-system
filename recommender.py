# ============================================
# STEP 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# STEP 2: Load Dataset
# ============================================
movies = pd.read_csv('movies.csv')

# ============================================
# STEP 3: Preprocessing
# ============================================
movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['title'] + ' ' + movies['genres']

# ============================================
# 📊 GRAPH 1: Genre Distribution
# ============================================
all_genres = ' '.join(movies['genres']).split()
genre_counts = pd.Series(all_genres).value_counts().head(10)

plt.figure()
genre_counts.plot(kind='bar')
plt.title("Top 10 Movie Genres")
plt.xlabel("Genres")
plt.ylabel("Count")
plt.show()

# ============================================
# STEP 4: TF-IDF
# ============================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# ============================================
# STEP 5: Cosine Similarity
# ============================================
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ============================================
# STEP 6: Index Mapping
# ============================================
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ============================================
# STEP 7: Recommendation Function
# ============================================
def recommend_movies(title, num_recommendations=10):
    
    if title not in indices:
        return None, None
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices], scores


# ============================================
# 📊 GRAPH 2: Recommendation Scores
# ============================================
movie_name = 'Toy Story (1995)'
recommended_movies, scores = recommend_movies(movie_name)

if recommended_movies is not None:
    plt.figure()
    plt.barh(recommended_movies[::-1], scores[::-1])
    plt.title(f"Top Recommendations for {movie_name}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Movies")
    plt.show()

# ============================================
# 📊 GRAPH 3: Cosine Similarity Heatmap
# ============================================
# Take small subset (first 20 movies)
subset = cosine_sim[:20, :20]

plt.figure()
sns.heatmap(subset)
plt.title("Cosine Similarity Heatmap (Sample)")
plt.show()

# ============================================
# STEP 8: Interactive Mode
# ============================================
while True:
    user_input = input("\nEnter movie name (or 'exit'): ")
    
    if user_input.lower() == 'exit':
        break
    
    recs, scores = recommend_movies(user_input)
    
    if recs is None:
        print("Movie not found!")
        continue
    
    print("\nRecommended Movies:\n")
    print(recs)
    
    # Plot for user input
    plt.figure()
    plt.barh(recs[::-1], scores[::-1])
    plt.title(f"Recommendations for {user_input}")
    plt.show()