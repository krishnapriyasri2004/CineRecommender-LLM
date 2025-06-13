import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """
    Comprehensive evaluation metrics for recommendation systems
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def split_data_temporal(self, ratings_df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally (more realistic for recommendation systems)"""
        
        # Sort by timestamp
        ratings_sorted = ratings_df.sort_values('timestamp')
        
        # Split point
        split_point = int(len(ratings_sorted) * (1 - test_ratio))
        
        train_data = ratings_sorted.iloc[:split_point]
        test_data = ratings_sorted.iloc[split_point:]
        
        logger.info(f"Temporal split: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_precision_at_k(self, recommended_items: List, relevant_items: List, k: int = 10) -> float:
        """Calculate Precision@K"""
        
        if not recommended_items:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / min(k, len(recommended_k))
    
    def calculate_recall_at_k(self, recommended_items: List, relevant_items: List, k: int = 10) -> float:
        """Calculate Recall@K"""
        
        if not relevant_items:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def calculate_f1_at_k(self, recommended_items: List, relevant_items: List, k: int = 10) -> float:
        """Calculate F1@K"""
        
        precision = self.calculate_precision_at_k(recommended_items, relevant_items, k)
        recall = self.calculate_recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_ndcg_at_k(self, recommended_items: List, relevant_items: Dict, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        
        def dcg_at_k(scores: List[float], k: int) -> float:
            scores_k = scores[:k]
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores_k))
        
        # Get relevance scores for recommended items
        recommended_k = recommended_items[:k]
        relevance_scores = [relevant_items.get(item, 0) for item in recommended_k]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevant_items.values(), reverse=True)
        idcg = dcg_at_k(ideal_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_coverage(self, all_recommendations: List[List], total_items: int) -> float:
        """Calculate catalog coverage"""
        
        unique_recommended = set()
        for recommendations in all_recommendations:
            unique_recommended.update(recommendations)
        
        return len(unique_recommended) / total_items
    
    def calculate_diversity(self, recommendations: List, item_features: pd.DataFrame) -> float:
        """Calculate intra-list diversity using cosine similarity"""
        
        if len(recommendations) < 2:
            return 0.0
        
        # Get features for recommended items
        rec_features = item_features[item_features.index.isin(recommendations)]
        
        if len(rec_features) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(rec_features)
        
        # Calculate average similarity (excluding diagonal)
        n = len(similarities)
        total_similarity = np.sum(similarities) - np.trace(similarities)
        avg_similarity = total_similarity / (n * (n - 1))
        
        # Diversity is 1 - similarity
        return 1 - avg_similarity
    
    def calculate_novelty(self, recommendations: List, item_popularity: Dict) -> float:
        """Calculate novelty based on item popularity"""
        
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 1)
            # Novelty is inverse of popularity (log scale)
            novelty = -np.log2(popularity / sum(item_popularity.values()))
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def evaluate_model_performance(self, model, test_data: pd.DataFrame, 
                                 user_item_matrix: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data"""
        
        results = {
            'rmse': [],
            'mae': [],
            'precision_at_5': [],
            'precision_at_10': [],
            'recall_at_5': [],
            'recall_at_10': [],
            'f1_at_5': [],
            'f1_at_10': [],
            'ndcg_at_5': [],
            'ndcg_at_10': []
        }
        
        # Group test data by user
        user_groups = test_data.groupby('userId')
        
        for user_id, user_test_data in user_groups:
            
            # Get user's test movies and ratings
            test_movies = user_test_data['movieId'].tolist()
            test_ratings = user_test_data['rating'].tolist()
            
            # Get relevant items (highly rated movies)
            relevant_items = [movie for movie, rating in zip(test_movies, test_ratings) if rating >= 4]
            
            try:
                # Get recommendations from model
                if hasattr(model, 'get_collaborative_recommendations'):
                    recommendations = model.get_collaborative_recommendations(user_id, n_recommendations=20)
                    recommended_items = [rec['movieId'] for rec in recommendations]
                else:
                    recommended_items = []
                
                if recommended_items and relevant_items:
                    # Calculate metrics
                    results['precision_at_5'].append(
                        self.calculate_precision_at_k(recommended_items, relevant_items, 5)
                    )
                    results['precision_at_10'].append(
                        self.calculate_precision_at_k(recommended_items, relevant_items, 10)
                    )
                    results['recall_at_5'].append(
                        self.calculate_recall_at_k(recommended_items, relevant_items, 5)
                    )
                    results['recall_at_10'].append(
                        self.calculate_recall_at_k(recommended_items, relevant_items, 10)
                    )
                    results['f1_at_5'].append(
                        self.calculate_f1_at_k(recommended_items, relevant_items, 5)
                    )
                    results['f1_at_10'].append(
                        self.calculate_f1_at_k(recommended_items, relevant_items, 10)
                    )
                    
                    # For NDCG, create relevance scores
                    relevance_dict = {movie: rating for movie, rating in zip(test_movies, test_ratings)}
                    results['ndcg_at_5'].append(
                        self.calculate_ndcg_at_k(recommended_items, relevance_dict, 5)
                    )
                    results['ndcg_at_10'].append(
                        self.calculate_ndcg_at_k(recommended_items, relevance_dict, 10)
                    )
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        avg_results = {}
        for metric, values in results.items():
            if values:
                avg_results[metric] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
            else:
                avg_results[metric] = 0.0
                avg_results[f'{metric}_std'] = 0.0
        
        return avg_results
    
    def evaluate_recommendation_quality(self, all_recommendations: List[List], 
                                      movies_df: pd.DataFrame, 
                                      ratings_df: pd.DataFrame) -> Dict:
        """Evaluate overall recommendation quality"""
        
        # Calculate item popularity
        item_popularity = ratings_df['movieId'].value_counts().to_dict()
        
        # Calculate coverage
        total_movies = len(movies_df)
        coverage = self.calculate_coverage(all_recommendations, total_movies)
        
        # Calculate average novelty
        novelty_scores = []
        for recommendations in all_recommendations:
            if recommendations:
                novelty = self.calculate_novelty(recommendations, item_popularity)
                novelty_scores.append(novelty)
        
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
        
        # Calculate diversity (if content features available)
        diversity_scores = []
        if 'genres' in movies_df.columns:
            # Create simple genre features for diversity calculation
            genre_features = movies_df['genres'].str.get_dummies(sep='|')
            genre_features.index = movies_df['movieId']
            
            for recommendations in all_recommendations:
                if len(recommendations) > 1:
                    diversity = self.calculate_diversity(recommendations, genre_features)
                    diversity_scores.append(diversity)
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        
        return {
            'coverage': coverage,
            'avg_novelty': avg_novelty,
            'avg_diversity': avg_diversity,
            'total_recommendations': len(all_recommendations),
            'avg_recommendations_per_user': np.mean([len(recs) for recs in all_recommendations])
        }
    
    def create_evaluation_report(self, model_results: Dict, quality_results: Dict) -> str:
        """Create comprehensive evaluation report"""
        
        report = f"""
# Movie Recommendation System Evaluation Report

## Model Performance Metrics

### Accuracy Metrics
- **Precision@5**: {model_results.get('precision_at_5', 0):.4f} ± {model_results.get('precision_at_5_std', 0):.4f}
- **Precision@10**: {model_results.get('precision_at_10', 0):.4f} ± {model_results.get('precision_at_10_std', 0):.4f}
- **Recall@5**: {model_results.get('recall_at_5', 0):.4f} ± {model_results.get('recall_at_5_std', 0):.4f}
- **Recall@10**: {model_results.get('recall_at_10', 0):.4f} ± {model_results.get('recall_at_10_std', 0):.4f}

### Ranking Metrics
- **F1@5**: {model_results.get('f1_at_5', 0):.4f} ± {model_results.get('f1_at_5_std', 0):.4f}
- **F1@10**: {model_results.get('f1_at_10', 0):.4f} ± {model_results.get('f1_at_10_std', 0):.4f}
- **NDCG@5**: {model_results.get('ndcg_at_5', 0):.4f} ± {model_results.get('ndcg_at_5_std', 0):.4f}
- **NDCG@10**: {model_results.get('ndcg_at_10', 0):.4f} ± {model_results.get('ndcg_at_10_std', 0):.4f}

## Recommendation Quality Metrics

### Coverage and Diversity
- **Catalog Coverage**: {quality_results.get('coverage', 0):.4f} ({quality_results.get('coverage', 0)*100:.1f}% of movies recommended)
- **Average Novelty**: {quality_results.get('avg_novelty', 0):.4f}
- **Average Diversity**: {quality_results.get('avg_diversity', 0):.4f}

### System Statistics
- **Total Recommendations Generated**: {quality_results.get('total_recommendations', 0)}
- **Average Recommendations per User**: {quality_results.get('avg_recommendations_per_user', 0):.1f}

## Performance Interpretation

### Strengths
- High precision indicates relevant recommendations
- Good coverage ensures diverse movie discovery
- Balanced novelty promotes serendipitous discoveries

### Areas for Improvement
- Monitor recall to ensure we're not missing relevant items
- Balance between accuracy and diversity
- Consider user feedback for continuous improvement

## Recommendations for Production

1. **A/B Testing**: Implement A/B testing to compare different algorithms
2. **Real-time Monitoring**: Track user engagement and satisfaction metrics
3. **Feedback Loop**: Incorporate user ratings and implicit feedback
4. **Cold Start**: Develop strategies for new users and new movies
5. **Scalability**: Optimize for larger datasets and real-time inference

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        return report
    
    def plot_evaluation_metrics(self, results: Dict):
        """Create visualization of evaluation metrics"""
        
        # Prepare data for plotting
        metrics = ['precision_at_5', 'precision_at_10', 'recall_at_5', 'recall_at_10', 
                  'f1_at_5', 'f1_at_10', 'ndcg_at_5', 'ndcg_at_10']
        
        values = [results.get(metric, 0) for metric in metrics]
        errors = [results.get(f'{metric}_std', 0) for metric in metrics]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Bar plot with error bars
        bars = plt.bar(range(len(metrics)), values, yerr=errors, capsize=5, alpha=0.7)
        
        # Customize plot
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Score')
        plt.title('Recommendation System Performance Metrics')
        plt.xticks(range(len(metrics)), [m.replace('_', '@').title() for m in metrics], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return plt

if __name__ == "__main__":
    print("Evaluation metrics module loaded successfully!")
