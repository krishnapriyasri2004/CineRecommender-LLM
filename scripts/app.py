import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from typing import List, Dict, Optional
import time
from datetime import datetime

# Import your recommendation engine
# Assuming your code is in a file called 'recommendation_models.py'
try:
    from recommendation_models import EnhancedMovieRecommendationEngine
except ImportError:
    st.error("Please make sure 'recommendation_models.py' is in the same directory!")
    st.stop()

class LLMIntegration:
    """LLM Integration class supporting multiple providers"""
    
    def __init__(self):
        self.providers = {
            "OpenAI": self._call_openai,
            "Groq": self._call_groq,
            "Hugging Face": self._call_huggingface,
            "Ollama (Local)": self._call_ollama
        }
    
    def _call_openai(self, prompt: str, api_key: str, model: str = "gpt-3.5-turbo") -> str:
        """Call OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def _call_groq(self, prompt: str, api_key: str, model: str = "mixtral-8x7b-32768") -> str:
        """Call Groq API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model,
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Groq: {str(e)}"
    
    def _call_huggingface(self, prompt: str, api_key: str, model: str = "microsoft/DialoGPT-large") -> str:
        """Call Hugging Face API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated")
                return str(result)
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Hugging Face: {str(e)}"
    
    def _call_ollama(self, prompt: str, api_key: str = "", model: str = "llama2") -> str:
        """Call local Ollama API"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Ollama: {str(e)}. Make sure Ollama is running locally."
    
    def generate_response(self, prompt: str, provider: str, api_key: str, model: str) -> str:
        """Generate response using specified LLM provider"""
        if provider in self.providers:
            return self.providers[provider](prompt, api_key, model)
        else:
            return "Unsupported LLM provider"

def create_movie_prompt(user_preferences: str, movies_data: List[Dict]) -> str:
    """Create a detailed prompt for LLM movie recommendations"""
    
    movies_info = []
    for movie in movies_data[:20]:  # Limit to prevent token overflow
        movies_info.append(
            f"- {movie['title']} | Genres: {movie['genres']} | "
            f"Rating: {movie.get('avg_rating', 'N/A')}/5 | "
            f"Votes: {movie.get('rating_count', 'N/A')}"
        )
    
    prompt = f"""You are an expert movie recommendation assistant. Based on the user's preferences and the available movies, provide personalized movie recommendations with detailed explanations.

User Preferences: "{user_preferences}"

Available Movies:
{chr(10).join(movies_info)}

Please provide:
1. Top 5 movie recommendations that best match the user's preferences
2. For each recommendation, explain WHY it matches their preferences
3. Mention specific aspects like genre, themes, or style that align with their taste
4. Rate how well each movie matches their preferences (1-10 scale)

Format your response clearly with movie titles, explanations, and match scores."""

    return prompt

@st.cache_resource
def load_recommendation_engine():
    """Load and cache the recommendation engine"""
    engine = EnhancedMovieRecommendationEngine()
    engine.load_data()
    engine.preprocess_data()
    engine.train_collaborative_filtering()
    return engine

def main():
    st.set_page_config(
        page_title="🎬 AI Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .recommendation-score {
        background: #4ecdc4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎬 AI-Powered Movie Recommender</h1>', unsafe_allow_html=True)
    
    # Initialize LLM integration
    llm = LLMIntegration()
    
    # Sidebar Configuration
    st.sidebar.header("🔧 Configuration")
    
    # LLM Provider Selection
    provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        ["OpenAI", "Groq", "Hugging Face", "Ollama (Local)"]
    )
    
    # API Key input (if needed)
    api_key = ""
    if provider != "Ollama (Local)":
        api_key = st.sidebar.text_input(
            f"{provider} API Key:",
            type="password",
            help=f"Enter your {provider} API key"
        )
    
    # Model selection
    model_options = {
        "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        "Groq": ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
        "Hugging Face": ["microsoft/DialoGPT-large", "microsoft/DialoGPT-medium", "facebook/blenderbot-400M-distill"],
        "Ollama (Local)": ["llama2", "mistral", "codellama", "vicuna"]
    }
    
    model = st.sidebar.selectbox(
        "Choose Model:",
        model_options[provider]
    )
    
    # Data Directory
    data_directory = st.sidebar.text_input(
        "Data Directory:",
        value="D:/Movie Recommendor-python/data",
        help="Path to your movie data files"
    )
    
    # Load recommendation engine
    try:
        with st.spinner("Loading recommendation engine..."):
            if 'engine' not in st.session_state:
                engine = EnhancedMovieRecommendationEngine(data_directory)
                engine.load_data()
                engine.preprocess_data()
                engine.train_collaborative_filtering()
                st.session_state.engine = engine
            else:
                engine = st.session_state.engine
        
        st.sidebar.success("✅ Engine loaded successfully!")
        
        # Display data statistics
        st.sidebar.metric("Total Movies", len(engine.movies_df))
        st.sidebar.metric("Total Ratings", len(engine.ratings_df))
        st.sidebar.metric("Total Users", engine.ratings_df['userId'].nunique())
        
    except Exception as e:
        st.error(f"Error loading recommendation engine: {str(e)}")
        st.stop()
    
    # Main Interface
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Recommendations", "🔍 Movie Search", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Get AI-Powered Movie Recommendations")
        
        # User input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_preferences = st.text_area(
                "Tell me about your movie preferences:",
                placeholder="E.g., I love action movies with great special effects, adventure, and strong characters. I also enjoy sci-fi and fantasy genres.",
                height=100
            )
        
        with col2:
            st.write("**Quick Options:**")
            if st.button("🎬 Action & Adventure"):
                user_preferences = "I love action movies with adventure, thrills, and exciting sequences"
            if st.button("😂 Comedy & Fun"):
                user_preferences = "I enjoy comedy movies that are funny, light-hearted, and entertaining"
            if st.button("💕 Romance & Drama"):
                user_preferences = "I like romantic movies with drama, emotional stories, and character development"
            if st.button("🚀 Sci-Fi & Fantasy"):
                user_preferences = "I'm interested in science fiction and fantasy movies with creative worlds and concepts"
        
        # Recommendation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            n_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
        with col2:
            use_llm = st.checkbox("Use LLM Enhancement", value=True)
        with col3:
            show_explanations = st.checkbox("Show detailed explanations", value=True)
        
        # Generate recommendations button
        if st.button("🎯 Get Recommendations", type="primary"):
            if not user_preferences.strip():
                st.warning("Please enter your movie preferences!")
            elif use_llm and provider != "Ollama (Local)" and not api_key.strip():
                st.warning(f"Please enter your {provider} API key!")
            else:
                with st.spinner("Generating personalized recommendations..."):
                    try:
                        # Get base recommendations
                        base_recs = engine.get_llm_enhanced_recommendations(user_preferences, n_recommendations * 2)
                        
                        if use_llm:
                            # Create LLM prompt
                            prompt = create_movie_prompt(user_preferences, base_recs['recommendations'])
                            
                            # Get LLM response
                            llm_response = llm.generate_response(prompt, provider, api_key, model)
                            
                            # Display LLM response
                            st.subheader("🤖 AI-Enhanced Recommendations")
                            st.markdown(f"""
                            <div class="movie-card">
                            <h4>🎭 Personalized Analysis</h4>
                            {llm_response}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display standard recommendations
                        st.subheader("📋 Detailed Recommendations")
                        
                        for i, rec in enumerate(base_recs['recommendations'][:n_recommendations], 1):
                            with st.expander(f"{i}. {rec['title']} ⭐ {rec['avg_rating']:.1f}/5"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**Genres:** {rec['genres'].replace('|', ', ')}")
                                    st.write(f"**Average Rating:** {rec['avg_rating']:.1f}/5")
                                    st.write(f"**Total Ratings:** {rec['rating_count']:,}")
                                    
                                    if show_explanations:
                                        st.write(f"**Why Recommended:** {rec['explanation']}")
                                        st.write(f"**Details:** {rec['why_recommended']}")
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="recommendation-score">
                                    Match Score: {rec['recommendation_score']:.1f}/5
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if rec['tags']:
                                        st.write("**Tags:**")
                                        for tag in rec['tags'][:5]:
                                            st.write(f"• {tag}")
                        
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
    
    with tab2:
        st.header("🔍 Movie Search & Details")
        
        # Search functionality
        search_term = st.text_input("Search for a movie:", placeholder="Enter movie title...")
        
        if search_term:
            # Filter movies based on search term
            filtered_movies = engine.movies_df[
                engine.movies_df['title'].str.contains(search_term, case=False, na=False)
            ]
            
            if len(filtered_movies) > 0:
                st.write(f"Found {len(filtered_movies)} movies:")
                
                for _, movie in filtered_movies.head(10).iterrows():
                    with st.expander(f"{movie['title']}"):
                        movie_details = engine.get_movie_details(movie['movieId'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Movie ID:** {movie['movieId']}")
                            st.write(f"**Genres:** {movie['genres'].replace('|', ', ')}")
                            st.write(f"**Average Rating:** {movie_details.get('avg_rating', 'N/A'):.1f}/5")
                        
                        with col2:
                            st.write(f"**Total Ratings:** {movie_details.get('rating_count', 0):,}")
                            if movie_details.get('tags'):
                                st.write(f"**Tags:** {', '.join(movie_details['tags'][:5])}")
                        
                        # Get similar movies
                        if st.button(f"Find Similar Movies", key=f"similar_{movie['movieId']}"):
                            similar = engine.get_content_based_recommendations(movie['movieId'], 5)
                            st.write("**Similar Movies:**")
                            for sim_movie in similar:
                                st.write(f"• {sim_movie['title']} (Similarity: {sim_movie['similarity_score']:.3f})")
            else:
                st.info("No movies found matching your search.")
    
    with tab3:
        st.header("📊 Movie Database Analytics")
        
        # Genre distribution
        st.subheader("Genre Distribution")
        genres_list = []
        for genres in engine.movies_df['genres'].dropna():
            genres_list.extend(genres.split('|'))
        
        genre_counts = pd.Series(genres_list).value_counts()
        st.bar_chart(genre_counts.head(10))
        
        # Rating distribution
        st.subheader("Rating Distribution")
        rating_counts = engine.ratings_df['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)
        
        # Top rated movies
        st.subheader("Top Rated Movies (Min 100 ratings)")
        top_movies = engine.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        top_movies.columns = ['avg_rating', 'rating_count']
        top_movies = top_movies[top_movies['rating_count'] >= 100].sort_values('avg_rating', ascending=False)
        
        top_movies_with_titles = top_movies.merge(
            engine.movies_df[['movieId', 'title']], 
            left_index=True, 
            right_on='movieId'
        ).head(10)
        
        st.dataframe(top_movies_with_titles[['title', 'avg_rating', 'rating_count']])
    
    with tab4:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🎬 AI-Powered Movie Recommender
        
        This application combines traditional recommendation algorithms with modern Large Language Models (LLMs) 
        to provide personalized movie recommendations.
        
        **Features:**
        - 🤖 **LLM Integration**: Uses OpenAI, Groq, Hugging Face, or local Ollama
        - 🎯 **Multiple Algorithms**: Content-based, collaborative filtering, and popularity-based
        - 📊 **Analytics**: Explore your movie database with interactive charts
        - 🔍 **Search**: Find and explore movie details
        - 💡 **Explanations**: Understand why movies are recommended to you
        
        **Supported LLM Providers:**
        - **OpenAI**: GPT-3.5, GPT-4 models
        - **Groq**: Fast inference with Mixtral, Llama2
        - **Hugging Face**: Various open-source models
        - **Ollama**: Local deployment for privacy
        
        **How it works:**
        1. Enter your movie preferences in natural language
        2. The system generates base recommendations using ML algorithms
        3. LLM enhances recommendations with detailed explanations
        4. Results are personalized based on your specific taste
        
        **Data Requirements:**
        - `movies.csv`: Movie information with genres
        - `ratings.csv`: User ratings data
        - `tags.csv`: User-generated tags (optional)
        """)
        
        st.subheader("🔧 Setup Instructions")
        
        with st.expander("API Key Setup"):
            st.markdown("""
            **OpenAI:**
            1. Sign up at https://platform.openai.com/
            2. Generate API key in API section
            3. Add billing information for usage
            
            **Groq:**
            1. Sign up at https://console.groq.com/
            2. Generate API key (free tier available)
            3. Enjoy fast inference speeds
            
            **Hugging Face:**
            1. Sign up at https://huggingface.co/
            2. Generate access token in settings
            3. Use various open-source models
            
            **Ollama (Local):**
            1. Install Ollama from https://ollama.ai/
            2. Run `ollama pull llama2` to download model
            3. Start server with `ollama serve`
            """)
        
        st.subheader("📈 Performance Stats")
        if 'engine' in st.session_state:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Movies", f"{len(st.session_state.engine.movies_df):,}")
            with col2:
                st.metric("Ratings", f"{len(st.session_state.engine.ratings_df):,}")
            with col3:
                st.metric("Users", f"{st.session_state.engine.ratings_df['userId'].nunique():,}")
            with col4:
                avg_rating = st.session_state.engine.ratings_df['rating'].mean()
                st.metric("Avg Rating", f"{avg_rating:.2f}/5")

if __name__ == "__main__":
    main()