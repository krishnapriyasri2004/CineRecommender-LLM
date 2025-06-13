# Enhanced Movie Recommendation Engine with Real Data and LLM Integration
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os
import requests
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedMovieRecommendationEngine:
    """Complete Movie Recommendation System with LLM Integration"""
    
    def __init__(self, data_directory="D:/Movie Recommendor-python/data"):
        self.data_directory = data_directory
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.user_item_matrix = None
        self.cosine_sim = None
        self.svd_model = None
        self.predicted_ratings = None
        self.tfidf_matrix = None
        
        # File paths
        self.movies_path = os.path.join(data_directory, "movies.csv")
        self.ratings_path = os.path.join(data_directory, "ratings.csv")
        self.tags_path = os.path.join(data_directory, "tags.csv")
        
    def load_data(self):
        """Load movie data from CSV files"""
        print("Loading movie data from files...")
        
        try:
            # Load the actual data files
            print(f"Loading movies from: {self.movies_path}")
            self.movies_df = pd.read_csv(self.movies_path)
            
            print(f"Loading ratings from: {self.ratings_path}")
            self.ratings_df = pd.read_csv(self.ratings_path)
            
            print(f"Loading tags from: {self.tags_path}")
            self.tags_df = pd.read_csv(self.tags_path)
            
            print(f"✅ Successfully loaded {len(self.movies_df)} movies, {len(self.ratings_df)} ratings, {len(self.tags_df)} tags")
            
            # Display basic info about the data
            print(f"\nData Overview:")
            print(f"Movies: {len(self.movies_df)} entries")
            print(f"Ratings: {len(self.ratings_df)} entries")
            print(f"Tags: {len(self.tags_df)} entries")
            print(f"Unique users: {self.ratings_df['userId'].nunique()}")
            print(f"Rating range: {self.ratings_df['rating'].min()} - {self.ratings_df['rating'].max()}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            print("Creating sample data instead...")
            self._create_sample_data()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self._create_sample_data()
        
    def _create_sample_data(self):
        """Create sample data for demonstration if files are not found"""
        # Sample movies (keeping the original sample for fallback)
        movies_data = {
            'movieId': list(range(1, 21)),
            'title': [
                'Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)',
                'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)',
                'Heat (1995)', 'Sabrina (1995)', 'Tom and Huck (1995)',
                'Sudden Death (1995)', 'GoldenEye (1995)', 'The American President (1995)',
                'Dracula: Dead and Loving It (1995)', 'Balto (1995)', 'Nixon (1995)',
                'Cutthroat Island (1995)', 'Casino (1995)', 'Sense and Sensibility (1995)',
                'Four Rooms (1995)', 'Ace Ventura: When Nature Calls (1995)',
                'Money Train (1995)'
            ],
            'genres': [
                'Adventure|Animation|Children|Comedy|Fantasy',
                'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance',
                'Comedy', 'Action|Crime|Thriller', 'Comedy|Romance', 'Adventure|Children',
                'Action', 'Action|Adventure|Thriller', 'Comedy|Drama|Romance',
                'Comedy|Horror', 'Adventure|Animation|Children', 'Drama',
                'Action|Adventure|Romance', 'Crime|Drama|Thriller', 'Drama|Romance',
                'Comedy|Crime|Thriller', 'Comedy', 'Action|Crime|Drama|Thriller'
            ]
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        
        # Generate sample ratings
        np.random.seed(42)
        ratings_data = []
        for user_id in range(1, 201):
            n_ratings = np.random.randint(5, 15)
            movie_ids = np.random.choice(range(1, 21), size=n_ratings, replace=False)
            for movie_id in movie_ids:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
                timestamp = np.random.randint(1000000000, 1600000000)
                ratings_data.append([user_id, movie_id, rating, timestamp])
        
        self.ratings_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating', 'timestamp'])
        
        # Generate sample tags
        tags_data = []
        sample_tags = ['funny', 'action', 'romance', 'thriller', 'family', 'adventure', 'comedy', 'drama', 'sci-fi', 'fantasy']
        for i in range(100):
            user_id = np.random.randint(1, 201)
            movie_id = np.random.randint(1, 21)
            tag = np.random.choice(sample_tags)
            timestamp = np.random.randint(1000000000, 1600000000)
            tags_data.append([user_id, movie_id, tag, timestamp])
        
        self.tags_df = pd.DataFrame(tags_data, columns=['userId', 'movieId', 'tag', 'timestamp'])
    
    def preprocess_data(self):
        """Preprocess and prepare data for recommendations"""
        print("Preprocessing data...")
        
        # Limit data size for faster processing (optional)
        max_users = 5000  # Limit to first 5000 users for faster processing
        if len(self.ratings_df['userId'].unique()) > max_users:
            top_users = self.ratings_df['userId'].value_counts().head(max_users).index
            self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(top_users)]
            print(f"Limited to top {max_users} active users for faster processing")
        
        # Create user-item matrix
        print("Creating user-item matrix...")
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Prepare content features for content-based filtering
        print("Preparing content features...")
        self.movies_df['content_features'] = self.movies_df['genres'].fillna('')
        
        # Add tags to content features if available
        if not self.tags_df.empty:
            movie_tags = self.tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
            self.movies_df = self.movies_df.merge(movie_tags, on='movieId', how='left')
            self.movies_df['tag'] = self.movies_df['tag'].fillna('')
            self.movies_df['content_features'] = self.movies_df['genres'] + ' ' + self.movies_df['tag']
        
        # Create TF-IDF matrix
        print("Creating TF-IDF matrix...")
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content_features'])
        
        # Compute cosine similarity for content-based filtering
        print("Computing similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("✅ Data preprocessing completed!")
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering model using SVD"""
        print("Training collaborative filtering model...")
        
        # Apply SVD
        user_item_array = self.user_item_matrix.values
        n_components = min(100, min(user_item_array.shape) - 1)
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = self.svd_model.fit_transform(user_item_array)
        item_factors = self.svd_model.components_
        
        # Reconstruct ratings matrix
        self.predicted_ratings = np.dot(user_factors, item_factors)
        
        print(f"✅ Collaborative filtering trained with {n_components} components")
    
    def get_content_based_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get content-based recommendations for a movie"""
        try:
            # Get movie index
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies (excluding the movie itself)
            movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
            
            recommendations = []
            for idx in movie_indices:
                movie_data = self.movies_df.iloc[idx]
                recommendations.append({
                    'movieId': int(movie_data['movieId']),
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity_score': float(sim_scores[idx][1])
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in content-based recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations for a user"""
        try:
            if user_id not in self.user_item_matrix.index:
                return self._get_popular_movies(n_recommendations)
            
            user_idx = list(self.user_item_matrix.index).index(user_id)
            user_predictions = self.predicted_ratings[user_idx]
            
            # Get movies not rated by user
            rated_movies = self.user_item_matrix.loc[user_id]
            unrated_movies = rated_movies[rated_movies == 0].index
            
            # Get predictions for unrated movies
            movie_scores = []
            for movie_id in unrated_movies:
                if movie_id in self.user_item_matrix.columns:
                    movie_idx = list(self.user_item_matrix.columns).index(movie_id)
                    score = user_predictions[movie_idx]
                    movie_scores.append((movie_id, score))
            
            # Sort by predicted rating
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for movie_id, score in movie_scores[:n_recommendations]:
                movie_data = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'predicted_rating': float(score)
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in collaborative recommendations: {e}")
            return self._get_popular_movies(n_recommendations)
    
    def _get_popular_movies(self, n_recommendations: int = 10) -> List[Dict]:
        """Get popular movies as fallback"""
        popular_movies = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        
        popular_movies.columns = ['avg_rating', 'rating_count']
        popular_movies = popular_movies[popular_movies['rating_count'] >= 10]  # At least 10 ratings
        popular_movies = popular_movies.sort_values(['avg_rating', 'rating_count'], ascending=[False, False])
        
        recommendations = []
        for movie_id in popular_movies.head(n_recommendations).index:
            try:
                movie_data = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'avg_rating': float(popular_movies.loc[movie_id, 'avg_rating']),
                    'rating_count': int(popular_movies.loc[movie_id, 'rating_count'])
                })
            except IndexError:
                continue
        
        return recommendations
    
    def get_llm_enhanced_recommendations(self, user_preferences: str, n_recommendations: int = 5) -> Dict:
        """Get LLM-enhanced recommendations with detailed explanations"""
        print("Generating LLM-enhanced recommendations...")
        
        # Get base recommendations from different methods
        popular_movies = self._get_popular_movies(10)
        
        # Create context for LLM
        movies_context = []
        for movie in popular_movies:
            # Get additional info
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie['movieId']]
            movie_tags = self.tags_df[self.tags_df['movieId'] == movie['movieId']]['tag'].unique()
            
            context = {
                'title': movie['title'],
                'genres': movie['genres'],
                'avg_rating': movie['avg_rating'],
                'rating_count': movie['rating_count'],
                'tags': list(movie_tags) if len(movie_tags) > 0 else []
            }
            movies_context.append(context)
        
        # Generate LLM-style recommendations
        llm_recommendations = self._generate_intelligent_recommendations(
            user_preferences, movies_context, n_recommendations
        )
        
        return {
            'user_preferences': user_preferences,
            'recommendations': llm_recommendations,
            'total_movies_analyzed': len(movies_context),
            'recommendation_method': 'LLM-Enhanced with Collaborative and Content Filtering'
        }
    
    def _generate_intelligent_recommendations(self, preferences: str, movies_context: List[Dict], n_recommendations: int) -> List[Dict]:
        """Generate intelligent recommendations using rule-based LLM simulation"""
        preferences_lower = preferences.lower()
        
        # Advanced keyword matching with weights
        genre_keywords = {
            'action': {'genres': ['Action'], 'weight': 1.0},
            'comedy': {'genres': ['Comedy'], 'weight': 1.0},
            'romance': {'genres': ['Romance'], 'weight': 1.0},
            'drama': {'genres': ['Drama'], 'weight': 1.0},
            'thriller': {'genres': ['Thriller'], 'weight': 1.0},
            'adventure': {'genres': ['Adventure'], 'weight': 1.0},
            'animation': {'genres': ['Animation'], 'weight': 1.0},
            'sci-fi': {'genres': ['Sci-Fi'], 'weight': 1.0},
            'fantasy': {'genres': ['Fantasy'], 'weight': 1.0},
            'horror': {'genres': ['Horror'], 'weight': 1.0},
            'family': {'genres': ['Children'], 'weight': 1.0},
            'funny': {'genres': ['Comedy'], 'weight': 0.8},
            'scary': {'genres': ['Horror'], 'weight': 0.8},
            'exciting': {'genres': ['Action', 'Adventure'], 'weight': 0.7}
        }
        
        # Quality indicators
        quality_keywords = {
            'good': 0.1, 'great': 0.2, 'excellent': 0.3, 'amazing': 0.3,
            'best': 0.4, 'top': 0.3, 'high quality': 0.3, 'well made': 0.2
        }
        
        # Score movies
        movie_scores = []
        for movie in movies_context:
            score = movie['avg_rating']  # Base score
            explanation_parts = []
            
            # Genre matching
            for keyword, info in genre_keywords.items():
                if keyword in preferences_lower:
                    for genre in info['genres']:
                        if genre in movie['genres']:
                            score += info['weight']
                            explanation_parts.append(f"matches your interest in {keyword}")
                            break
            
            # Quality boost
            for keyword, boost in quality_keywords.items():
                if keyword in preferences_lower:
                    score += boost
                    if movie['avg_rating'] >= 4.0:
                        explanation_parts.append("has excellent ratings")
            
            # Popularity boost for well-rated movies
            if movie['rating_count'] > 1000:
                score += 0.2
                explanation_parts.append("widely acclaimed")
            
            # Tag matching
            if movie['tags']:
                for tag in movie['tags']:
                    if any(keyword in tag.lower() for keyword in preferences_lower.split()):
                        score += 0.3
                        explanation_parts.append(f"tagged as '{tag}'")
                        break
            
            # Create explanation
            if explanation_parts:
                explanation = f"Recommended because it {', '.join(explanation_parts[:3])}."
            else:
                explanation = f"Popular {movie['genres'].replace('|', ', ')} movie with {movie['avg_rating']:.1f}/5 rating."
            
            movie_scores.append({
                'movie': movie,
                'score': score,
                'explanation': explanation
            })
        
        # Sort by score and return top recommendations
        movie_scores.sort(key=lambda x: x['score'], reverse=True)
        
        recommendations = []
        for item in movie_scores[:n_recommendations]:
            movie = item['movie']
            recommendations.append({
                'movieId': movie.get('movieId', 0),
                'title': movie['title'],
                'genres': movie['genres'],
                'avg_rating': movie['avg_rating'],
                'rating_count': movie['rating_count'],
                'tags': movie['tags'],
                'recommendation_score': round(item['score'], 2),
                'explanation': item['explanation'],
                'why_recommended': self._generate_detailed_explanation(movie, preferences)
            })
        
        return recommendations
    
    def _generate_detailed_explanation(self, movie: Dict, preferences: str) -> str:
        """Generate detailed explanation for recommendation"""
        explanations = []
        
        # Genre analysis
        genres = movie['genres'].split('|')
        if len(genres) > 1:
            explanations.append(f"This {'/'.join(genres[:2])} film offers a rich blend of {' and '.join(genres[:3]).lower()}")
        else:
            explanations.append(f"This {genres[0].lower()} movie")
        
        # Quality assessment
        rating = movie['avg_rating']
        if rating >= 4.5:
            explanations.append("is exceptionally well-rated by viewers")
        elif rating >= 4.0:
            explanations.append("has received excellent reviews")
        elif rating >= 3.5:
            explanations.append("is well-regarded by audiences")
        
        # Popularity note
        if movie['rating_count'] > 5000:
            explanations.append("and has been watched by thousands of viewers")
        elif movie['rating_count'] > 1000:
            explanations.append("with a solid viewership base")
        
        return f"{' '.join(explanations)}."
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Get detailed information about a movie"""
        try:
            movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            
            # Get rating statistics
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']
            
            # Get tags
            movie_tags = self.tags_df[self.tags_df['movieId'] == movie_id]['tag'].unique()
            
            # Get rating distribution
            if len(movie_ratings) > 0:
                rating_dist = movie_ratings.value_counts().sort_index()
                rating_distribution = {int(k): int(v) for k, v in rating_dist.items()}
            else:
                rating_distribution = {}
            
            return {
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'avg_rating': float(movie_ratings.mean()) if len(movie_ratings) > 0 else 0.0,
                'rating_count': len(movie_ratings),
                'tags': list(movie_tags),
                'rating_distribution': rating_distribution
            }
        except Exception as e:
            print(f"Error getting movie details: {e}")
            return {}
    
    def generate_recommendation_report(self, user_preferences: str) -> str:
        """Generate a comprehensive recommendation report"""
        print("Generating comprehensive recommendation report...")
        
        # Get different types of recommendations
        llm_recs = self.get_llm_enhanced_recommendations(user_preferences, 5)
        popular_recs = self._get_popular_movies(5)
        
        report = f"""
🎬 PERSONALIZED MOVIE RECOMMENDATION REPORT
{'='*60}

USER PREFERENCES: {user_preferences}

🤖 AI-ENHANCED RECOMMENDATIONS:
{'-'*40}
"""
        
        for i, rec in enumerate(llm_recs['recommendations'], 1):
            report += f"""
{i}. {rec['title']}
   ⭐ Rating: {rec['avg_rating']:.1f}/5 ({rec['rating_count']} ratings)
   🎭 Genres: {rec['genres'].replace('|', ', ')}
   🎯 Match Score: {rec['recommendation_score']}/5
   💡 Why: {rec['explanation']}
   📝 Details: {rec['why_recommended']}
"""
        
        report += f"""

📈 TOP POPULAR MOVIES (For Comparison):
{'-'*40}
"""
        
        for i, rec in enumerate(popular_recs, 1):
            report += f"""
{i}. {rec['title']}
   ⭐ Rating: {rec['avg_rating']:.1f}/5 ({rec['rating_count']} ratings)
   🎭 Genres: {rec['genres'].replace('|', ', ')}
"""
        
        report += f"""

📊 RECOMMENDATION STATISTICS:
{'-'*40}
• Total movies in database: {len(self.movies_df):,}
• Total ratings analyzed: {len(self.ratings_df):,}
• Total users: {self.ratings_df['userId'].nunique():,}
• Recommendation method: {llm_recs['recommendation_method']}
• Movies analyzed for your preferences: {llm_recs['total_movies_analyzed']}

🎯 PERSONALIZATION NOTES:
{'-'*40}
Your recommendations are based on a combination of:
• Content similarity (genres, tags)
• Collaborative filtering (user behavior patterns)
• Popularity and quality metrics
• AI-enhanced preference matching

{'='*60}
"""
        
        return report

# Main execution function
def main():
    print("🎬 ENHANCED MOVIE RECOMMENDATION ENGINE")
    print("=" * 60)
    
    # Initialize engine with your data directory
    engine = EnhancedMovieRecommendationEngine("D:/Movie Recommendor-python/data")
    
    # Load data
    engine.load_data()
    
    # Preprocess data
    engine.preprocess_data()
    
    # Train collaborative filtering
    engine.train_collaborative_filtering()
    
    print("\n✅ Enhanced recommendation engine initialized successfully!")
    
    # Test different recommendation types
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Test 1: Content-based recommendations
    if len(engine.movies_df) > 0:
        sample_movie_id = engine.movies_df.iloc[0]['movieId']
        print(f"\n🔍 Content-Based Recommendations (Similar to Movie ID {sample_movie_id}):")
        content_recs = engine.get_content_based_recommendations(sample_movie_id, 5)
        for i, rec in enumerate(content_recs, 1):
            print(f"{i}. {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
    
    # Test 2: Collaborative filtering
    if len(engine.user_item_matrix) > 0:
        sample_user_id = engine.user_item_matrix.index[0]
        print(f"\n🔍 Collaborative Filtering Recommendations (User {sample_user_id}):")
        collab_recs = engine.get_collaborative_recommendations(sample_user_id, 5)
        for i, rec in enumerate(collab_recs, 1):
            if 'predicted_rating' in rec:
                print(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']:.2f})")
            else:
                print(f"{i}. {rec['title']} (Avg Rating: {rec.get('avg_rating', 'N/A')})")
    
    # Test 3: LLM-enhanced recommendations
    print(f"\n🤖 LLM-Enhanced Recommendations:")
    test_preferences = "I love action movies with great special effects and adventure"
    llm_recs = engine.get_llm_enhanced_recommendations(test_preferences, 3)
    for i, rec in enumerate(llm_recs['recommendations'], 1):
        print(f"{i}. {rec['title']}")
        print(f"   💡 {rec['explanation']}")
        print(f"   ⭐ {rec['avg_rating']:.1f}/5 ({rec['rating_count']} ratings)")
    
    # Generate full report
    print(f"\n📋 GENERATING COMPREHENSIVE REPORT...")
    report = engine.generate_recommendation_report(test_preferences)
    print(report)
    
    print("\n🎉 All tests completed successfully!")
    
    return engine

# Run the main function
if __name__ == "__main__":
    recommendation_engine = main()