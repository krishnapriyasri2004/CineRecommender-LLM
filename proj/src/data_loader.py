import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDataLoader:
    """
    Downloads and loads the MovieLens dataset from Kaggle
    Dataset: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_movielens_data(self):
        """Download MovieLens dataset (using a public mirror since we can't use Kaggle API directly)"""
        
        # URLs for MovieLens dataset files
        base_url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        
        zip_path = self.data_dir / "ml-25m.zip"
        
        if not zip_path.exists():
            logger.info("Downloading MovieLens 25M dataset...")
            response = requests.get(base_url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Download completed!")
        
        # Extract the zip file
        extract_dir = self.data_dir / "ml-25m"
        if not extract_dir.exists():
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info("Extraction completed!")
        
        return extract_dir
    
    def load_data(self):
        """Load and return the main datasets"""
        
        # Download data if not exists
        data_path = self.download_movielens_data()
        
        # Load datasets
        logger.info("Loading datasets...")
        
        movies = pd.read_csv(data_path / "movies.csv")
        ratings = pd.read_csv(data_path / "ratings.csv")
        tags = pd.read_csv(data_path / "tags.csv")
        
        # Sample data for faster processing (you can remove this for full dataset)
        logger.info("Sampling data for demo purposes...")
        
        # Take top 10000 most rated movies
        movie_counts = ratings['movieId'].value_counts().head(10000)
        popular_movies = movie_counts.index.tolist()
        
        # Filter datasets
        movies_filtered = movies[movies['movieId'].isin(popular_movies)]
        ratings_filtered = ratings[ratings['movieId'].isin(popular_movies)]
        tags_filtered = tags[tags['movieId'].isin(popular_movies)]
        
        # Take sample of users (top 5000 most active users)
        user_counts = ratings_filtered['userId'].value_counts().head(5000)
        active_users = user_counts.index.tolist()
        
        ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(active_users)]
        tags_filtered = tags_filtered[tags_filtered['userId'].isin(active_users)]
        
        logger.info(f"Loaded {len(movies_filtered)} movies, {len(ratings_filtered)} ratings, {len(tags_filtered)} tags")
        
        return movies_filtered, ratings_filtered, tags_filtered
    
    def create_sample_csv(self):
        """Create sample CSV files for demonstration"""
        
        # Create sample data if real data download fails
        logger.info("Creating sample movie data...")
        
        # Sample movies data
        sample_movies = pd.DataFrame({
            'movieId': range(1, 1001),
            'title': [f"Movie {i} ({1990 + i%30})" for i in range(1, 1001)],
            'genres': np.random.choice([
                'Action|Adventure', 'Comedy|Romance', 'Drama', 'Horror|Thriller',
                'Sci-Fi|Fantasy', 'Documentary', 'Animation|Children', 'Crime|Mystery'
            ], 1000)
        })
        
        # Sample ratings data
        np.random.seed(42)
        n_ratings = 50000
        sample_ratings = pd.DataFrame({
            'userId': np.random.randint(1, 1001, n_ratings),
            'movieId': np.random.randint(1, 1001, n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
            'timestamp': np.random.randint(946684800, 1640995200, n_ratings)  # 2000-2022
        })
        
        # Sample tags data
        sample_tags = pd.DataFrame({
            'userId': np.random.randint(1, 1001, 5000),
            'movieId': np.random.randint(1, 1001, 5000),
            'tag': np.random.choice([
                'funny', 'action-packed', 'romantic', 'scary', 'thought-provoking',
                'family-friendly', 'classic', 'underrated', 'must-watch', 'boring'
            ], 5000),
            'timestamp': np.random.randint(946684800, 1640995200, 5000)
        })
        
        # Save to CSV
        sample_movies.to_csv(self.data_dir / "movies.csv", index=False)
        sample_ratings.to_csv(self.data_dir / "ratings.csv", index=False)
        sample_tags.to_csv(self.data_dir / "tags.csv", index=False)
        
        logger.info("Sample CSV files created!")
        
        return sample_movies, sample_ratings, sample_tags

if __name__ == "__main__":
    loader = KaggleDataLoader()
    try:
        movies, ratings, tags = loader.load_data()
    except Exception as e:
        logger.warning(f"Failed to download real data: {e}")
        logger.info("Using sample data instead...")
        movies, ratings, tags = loader.create_sample_csv()
    
    print("Data loaded successfully!")
    print(f"Movies shape: {movies.shape}")
    print(f"Ratings shape: {ratings.shape}")
    print(f"Tags shape: {tags.shape}")
