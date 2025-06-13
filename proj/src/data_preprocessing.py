import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MovieDataPreprocessor:
    """
    Comprehensive data preprocessing for movie recommendation system
    """
    
    def __init__(self):
        self.genre_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
    def preprocess_movies(self, movies_df):
        """Preprocess movies dataset"""
        
        logger.info("Preprocessing movies data...")
        movies = movies_df.copy()
        
        # Extract year from title
        movies['year'] = movies['title'].str.extract(r'(\d{4})$')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        movies['year'] = movies['year'].fillna(movies['year'].median())
        
        # Clean title (remove year)
        movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Process genres
        movies['genres'] = movies['genres'].fillna('Unknown')
        movies['genre_list'] = movies['genres'].str.split('|')
        movies['num_genres'] = movies['genre_list'].str.len()
        
        # Create genre features
        all_genres = set()
        for genres in movies['genre_list']:
            all_genres.update(genres)
        
        for genre in all_genres:
            movies[f'genre_{genre}'] = movies['genre_list'].apply(lambda x: 1 if genre in x else 0)
        
        # Calculate movie age
        current_year = datetime.now().year
        movies['movie_age'] = current_year - movies['year']
        
        logger.info(f"Processed {len(movies)} movies with {len(all_genres)} unique genres")
        
        return movies
    
    def preprocess_ratings(self, ratings_df):
        """Preprocess ratings dataset"""
        
        logger.info("Preprocessing ratings data...")
        ratings = ratings_df.copy()
        
        # Convert timestamp to datetime
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year'] = ratings['datetime'].dt.year
        ratings['month'] = ratings['datetime'].dt.month
        ratings['day_of_week'] = ratings['datetime'].dt.dayofweek
        ratings['hour'] = ratings['datetime'].dt.hour
        
        # User stats
        user_stats = ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        }).round(3)
        user_stats.columns = ['user_rating_count', 'user_avg_rating', 'user_rating_std', 'user_unique_movies']
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Movie stats
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        }).round(3)
        movie_stats.columns = ['movie_rating_count', 'movie_avg_rating', 'movie_rating_std', 'movie_unique_users']
        movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)
        
        # Merge
        ratings = ratings.merge(user_stats, on='userId', how='left')
        ratings = ratings.merge(movie_stats, on='movieId', how='left')
        
        # Rating deviation
        ratings['rating_deviation'] = ratings['rating'] - ratings['user_avg_rating']
        
        logger.info(f"Processed {len(ratings)} ratings")
        
        return ratings, user_stats, movie_stats
    
    def preprocess_tags(self, tags_df):
        """Preprocess tags dataset"""
        
        logger.info("Preprocessing tags data...")
        tags = tags_df.copy()
        
        # Clean tags
        tags['tag'] = tags['tag'].str.lower().str.strip()
        tags['tag'] = tags['tag'].str.replace(r'[^\w\s]', '', regex=True)
        tags = tags[tags['tag'].str.len() >= 3]
        
        # Convert timestamp
        tags['datetime'] = pd.to_datetime(tags['timestamp'], unit='s')
        
        # Aggregate tags
        movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movie_tags.columns = ['movieId', 'all_tags']
        
        tag_counts = tags['tag'].value_counts()
        popular_tags = tag_counts.head(100).index.tolist()
        
        logger.info(f"Processed {len(tags)} tags, found {len(popular_tags)} popular tags")
        
        return tags, movie_tags, popular_tags
    
    def create_content_features(self, movies, movie_tags):
        """Create content-based features"""
        
        logger.info("Creating content-based features...")
        
        content_df = movies.merge(movie_tags, on='movieId', how='left')
        content_df['all_tags'] = content_df['all_tags'].fillna('')
        
        # Combine text
        content_df['content_text'] = (
            content_df['clean_title'] + ' ' + 
            content_df['genres'].str.replace('|', ' ') + ' ' + 
            content_df['all_tags']
        )
        
        # TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_df['content_text'])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f'tfidf_{name}' for name in feature_names],
            index=content_df.index
        )
        
        content_features = pd.concat([
            content_df[['movieId', 'year', 'num_genres', 'movie_age']],
            content_df[[col for col in content_df.columns if col.startswith('genre_')]],
            tfidf_df
        ], axis=1)
        
        logger.info(f"Created content features with {content_features.shape[1]} dimensions")
        
        return content_features
    
    def create_user_item_matrix(self, ratings):
        """Create user-item interaction matrix"""
        
        logger.info("Creating user-item matrix...")
        user_item_matrix = ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        logger.info(f"Created user-item matrix: {user_item_matrix.shape}")
        
        return user_item_matrix
    
    def process_all_data(self, movies_df, ratings_df, tags_df):
        """Process all datasets and return preprocessed data"""
        
        logger.info("Starting comprehensive data preprocessing...")
        
        movies_processed = self.preprocess_movies(movies_df)
        ratings_processed, user_stats, movie_stats = self.preprocess_ratings(ratings_df)
        tags_processed, movie_tags, popular_tags = self.preprocess_tags(tags_df)
        content_features = self.create_content_features(movies_processed, movie_tags)
        user_item_matrix = self.create_user_item_matrix(ratings_processed)
        
        processed_data = {
            'movies': movies_processed,
            'ratings': ratings_processed,
            'tags': tags_processed,
            'user_stats': user_stats,
            'movie_stats': movie_stats,
            'movie_tags': movie_tags,
            'popular_tags': popular_tags,
            'content_features': content_features,
            'user_item_matrix': user_item_matrix
        }
        
        logger.info("Data preprocessing completed successfully!")
        
        return processed_data

# ---------------------
# Main Execution Block
# ---------------------
if __name__ == "__main__":
    from data_loader import KaggleDataLoader
    
    loader = KaggleDataLoader()
    try:
        movies, ratings, tags = loader.load_data()
    except:
        movies, ratings, tags = loader.create_sample_csv()
    
    preprocessor = MovieDataPreprocessor()
    processed_data = preprocessor.process_all_data(movies, ratings, tags)
    
    print("Preprocessing completed!")
    for key, value in processed_data.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
