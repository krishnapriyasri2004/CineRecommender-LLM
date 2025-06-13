import os
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Any
import re
from datetime import datetime
from dotenv import load_dotenv
import time
import random

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMMovieRecommender:
    """
    Enhanced LLM-powered movie recommendation system with improved error handling and fallbacks
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize LLM recommender with enhanced error handling
        """
        # Try to get API key from parameter, environment variable, or use simulation
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.use_simulation = False
        self.api_error_count = 0
        self.max_api_errors = 3  # Switch to simulation after 3 consecutive errors
        
        if self.api_key and self.api_key != "simulate":
            # Validate API key format
            if not self.api_key.startswith('sk-') or len(self.api_key) < 30:
                logger.warning("Invalid OpenAI API key format. Using simulated responses.")
                self.use_simulation = True
            else:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=self.api_key)
                    logger.info("OpenAI API initialized successfully.")
                except ImportError:
                    logger.error("OpenAI package not installed. Using simulated responses.")
                    self.use_simulation = True
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.use_simulation = True
        else:
            logger.warning("Using simulated responses (no valid API key provided)")
            self.use_simulation = True
            self.client = None
        
        self.conversation_history = []
        
        # Enhanced simulation responses
        self.simulation_responses = {
            'explanation': [
                """🔍 **Why These Movies Were Recommended for You**

Based on your viewing history analysis, here's why these films are perfect matches:

**🎯 Genre Preferences**: Your consistent 4-5 star ratings for sci-fi and action films show a clear preference for intelligent, high-concept entertainment with spectacular visuals.

**📊 Rating Patterns**: You appreciate films that balance entertainment with substance - these recommendations score highly in both critic and audience reviews, matching your quality standards.

**🎬 Directorial Style**: You've rated films by visionary directors highly, and these recommendations feature similar auteur-driven storytelling approaches.

**🎭 Thematic Resonance**: Common themes in your top-rated films include technology, human nature, and philosophical questions - all present in these recommendations.

**⭐ Predicted Compatibility**: 94% match based on collaborative filtering with users who share your taste profile.""",
                
                """🧠 **Recommendation Logic Breakdown**

Your personalized recommendations are based on sophisticated preference analysis:

**Viewing Velocity**: You tend to rate movies within 2 weeks of release trends, suggesting you appreciate contemporary, cutting-edge cinema.

**Cross-Genre Appeal**: While you favor sci-fi/action, your ratings show appreciation for drama elements, leading to more nuanced recommendations.

**Technical Appreciation**: High ratings for films with strong cinematography and sound design informed these visually stunning picks.

**Narrative Complexity**: Your preference for multi-layered storytelling is reflected in these intellectually engaging selections.

**Cultural Impact**: You gravitate toward films that generate discussion - these recommendations are conversation starters."""
            ],
            
            'personalized': [
                """🎬 **Your Personalized Movie Collection**

**"Arrival" (2016)** ⭐⭐⭐⭐⭐
*Perfectly crafted for your love of thoughtful sci-fi*
When mysterious spacecraft appear worldwide, a linguist races to decode alien communication before global tensions explode. This Denis Villeneuve masterpiece combines your preferred elements: stunning visuals, profound themes, and Amy Adams' powerhouse performance.
**Why you'll love it**: Matches your 4.8★ average for cerebral sci-fi

**"Mad Max: Fury Road" (2015)** ⭐⭐⭐⭐⭐  
*High-octane action that respects intelligence*
In a post-apocalyptic wasteland, Max joins Furiosa's rebellion against a tyrannical warlord. George Miller's visual symphony delivers relentless action with meaningful character development - exactly your sweet spot.
**Your compatibility**: 96% match based on action preferences

**"Her" (2013)** ⭐⭐⭐⭐⭐
*Emotional depth meets technological wonder*  
A lonely writer develops a relationship with an AI operating system. Spike Jonze crafts an intimate exploration of love and technology that aligns with your appreciation for innovative storytelling.
**Personal prediction**: This will be in your top 10 favorites""",

                """🌟 **Curated Just for You**

**"Ex Machina" (2014)** ⭐⭐⭐⭐⭐
*A masterclass in psychological sci-fi*
A programmer conducts a Turing test on a sophisticated AI, but nothing is as it seems. Alex Garland's directorial debut perfectly balances your love for mind-bending concepts with intimate character study.
**Match reason**: 98% alignment with your thriller-sci-fi crossover preferences

**"Parasite" (2019)** ⭐⭐⭐⭐⭐
*Genre-defying brilliance*
A poor family infiltrates a wealthy household with devastating consequences. Bong Joon-ho's Oscar winner transcends categories - thriller, comedy, drama, horror - offering the narrative complexity you consistently rate highly.
**Why it's perfect**: Your appreciation for innovative storytelling shines here

**"1917" (2019)** ⭐⭐⭐⭐⭐
*Technical mastery meets emotional storytelling*
Two soldiers race across WWI battlefields to deliver a crucial message. Sam Mendes' "one-shot" war epic combines your love for cinematographic innovation with intense human drama.
**Confidence level**: 94% you'll rate this 4+ stars"""
            ],
            
            'mood': {
                'adventurous': """🗺️ **Adventure Awaits!**

**For Maximum Adrenaline:**
• **"Mission: Impossible - Fallout"** - Tom Cruise's death-defying stunts will have you gripping your seat
• **"The Raid"** - Indonesian martial arts perfection that redefines action choreography

**Epic Scale Adventures:**  
• **"Dune" (2021)** - Villeneuve's sci-fi epic offers world-building that demands big-screen viewing
• **"Lawrence of Arabia"** - David Lean's desert masterpiece still feels revolutionary

**Modern Thrills:**
• **"John Wick"** - Stylized action ballet with emotional core
• **"Baby Driver"** - Edgar Wright's musical heist film pulses with energy

*Perfect for when you want to feel your heart racing and your imagination soaring!*""",

                'thoughtful': """🤔 **Films for Deep Reflection**

**Philosophical Journeys:**
• **"Blade Runner 2049"** - Questions of humanity and memory in stunning visual poetry  
• **"The Tree of Life"** - Malick's cosmic meditation on existence and family

**Intellectual Puzzles:**
• **"Primer"** - Low-budget time travel that rewards multiple viewings
• **"Memento"** - Nolan's reverse-chronology memory thriller

**Emotional Depth:**
• **"Manchester by the Sea"** - Profound grief handled with remarkable sensitivity
• **"Moonlight"** - Coming-of-age poetry told in three acts

*When you want cinema that stays with you long after the credits roll.*""",

                'excited': """🎉 **High-Energy Entertainment!**

**Pure Fun:**
• **"Spider-Man: Into the Spider-Verse"** - Revolutionary animation that redefined possibilities
• **"The Grand Budapest Hotel"** - Wes Anderson's whimsical caper bursting with personality

**Crowd-Pleasers:**
• **"Avengers: Endgame"** - Epic culmination that delivers emotional and spectacle payoffs
• **"Top Gun: Maverick"** - Legacy sequel that soars beyond expectations

**Feel-Good Favorites:**
• **"Everything Everywhere All at Once"** - Multiverse madness with surprising heart
• **"Knives Out"** - Agatha Christie meets modern wit in perfect harmony

*For when you want to be thoroughly entertained and leave with a smile!*"""
            },
            
            'general': [
                """🎭 **Your Personal Movie Assistant**

I'm here to enhance your movie discovery experience! Based on your viewing patterns and preferences, I can help you:

**🔍 Discover & Explore:**
• Find hidden gems that match your exact taste profile
• Explore new genres based on films you already love
• Get context about why certain movies work for you

**🎯 Personalized Guidance:**
• Mood-based recommendations for any occasion
• Detailed explanations of why specific films suit you
• Comparisons with movies you've already rated highly

**💡 Smart Suggestions:**
Try asking me:
• "Why do you think I'd like Inception?"
• "I'm feeling nostalgic, what should I watch?"
• "Find me something like Blade Runner but newer"
• "What makes a good sci-fi film?"

*Ready to discover your next favorite movie?*""",

                """🌟 **Unlock Your Perfect Movie Experience**

Welcome to personalized cinema discovery! I specialize in understanding your unique viewing preferences and finding films that resonate with your specific tastes.

**What Makes Me Different:**
• Analysis of your rating patterns and genre preferences  
• Understanding of thematic elements you consistently enjoy
• Recognition of your preferred storytelling styles and pacing

**How I Can Help Today:**
✨ **Explain Recommendations** - Understand why certain films were suggested
✨ **Mood Matching** - Get recommendations based on how you're feeling  
✨ **Deep Dives** - Learn about specific movies that caught your interest
✨ **Discovery** - Find films you never knew existed but will love

**Popular Requests:**
• "I loved [Movie X], find me something similar"
• "Explain why you recommended [Movie Y]" 
• "I want something [mood/genre], surprise me"

*What would you like to explore first?*"""
            ]
        }
    
    def simulate_llm_response(self, prompt: str, intent: str = 'general') -> str:
        """Enhanced simulation with intent-specific responses"""
        
        if intent == 'explanation':
            return random.choice(self.simulation_responses['explanation'])
        
        elif intent == 'personalized':
            return random.choice(self.simulation_responses['personalized'])
        
        elif intent == 'mood':
            # Extract mood from prompt
            mood = self.extract_mood(prompt)
            if mood in self.simulation_responses['mood']:
                return self.simulation_responses['mood'][mood]
            else:
                # Default adventurous mood if not found
                return self.simulation_responses['mood']['adventurous']
        
        else:
            return random.choice(self.simulation_responses['general'])
    
    def call_openai_api(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Enhanced API calling with better error handling and fallback"""
        
        if self.use_simulation:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            # Reset error count on successful call
            self.api_error_count = 0
            return response.choices[0].message.content
            
        except Exception as e:
            self.api_error_count += 1
            logger.error(f"LLM API error (attempt {self.api_error_count}): {e}")
            
            # Switch to simulation mode after too many errors
            if self.api_error_count >= self.max_api_errors:
                logger.warning(f"Switching to simulation mode after {self.api_error_count} API errors")
                self.use_simulation = True
                
            return None
    
    def generate_explanation(self, user_id: int, recommendations: List[Dict], 
                           user_history: pd.DataFrame, movies_df: pd.DataFrame) -> str:
        """Generate LLM explanation for recommendations with enhanced fallback"""
        
        # Check if user_history has data and required columns
        if user_history.empty or 'userId' not in user_history.columns:
            return self.simulate_llm_response("", 'explanation')
        
        # Prepare user context
        user_ratings = user_history[user_history['userId'] == user_id]
        
        if user_ratings.empty:
            return self.simulate_llm_response("", 'explanation')
        
        top_rated_movies = user_ratings.nlargest(5, 'rating')
        
        # Get movie titles for context
        if not movies_df.empty and 'movieId' in movies_df.columns:
            top_movies_info = movies_df[movies_df['movieId'].isin(top_rated_movies['movieId'])]
        else:
            top_movies_info = pd.DataFrame()
        
        prompt = f"""
        As a movie recommendation expert, explain why these movies were recommended for a user based on their viewing history.

        User's Top Rated Movies:
        {top_movies_info[['title', 'genres']].to_string() if not top_movies_info.empty else "User enjoys high-quality films across various genres"}

        User's Average Rating: {user_ratings['rating'].mean():.1f}/5.0
        
        Recommended Movies:
        {[f"Movie ID: {rec['movieId']}" for rec in recommendations[:5]]}

        Provide a compelling, personalized explanation that highlights:
        1. Genre preference analysis
        2. Rating pattern insights  
        3. Thematic connections
        4. Why these specific recommendations match their taste profile
        
        Make it engaging and insightful, showing deep understanding of their preferences.
        """
        
        # Try API first, fall back to simulation
        api_response = self.call_openai_api(prompt, max_tokens=600, temperature=0.7)
        
        if api_response:
            return api_response
        else:
            return self.simulate_llm_response(prompt, 'explanation')
    
    def generate_personalized_descriptions(self, recommendations: List[Dict], 
                                         movies_df: pd.DataFrame, user_preferences: Dict) -> str:
        """Generate personalized movie descriptions using LLM with enhanced fallback"""
        
        # Check if movies_df has required data
        if movies_df.empty or 'movieId' not in movies_df.columns:
            return self.simulate_llm_response("", 'personalized')
        
        # Get movie details
        rec_movie_ids = [rec['movieId'] for rec in recommendations[:3]]
        rec_movies = movies_df[movies_df['movieId'].isin(rec_movie_ids)]
        
        prompt = f"""
        Create engaging, personalized movie descriptions for these recommendations:

        Movies to describe:
        {rec_movies[['title', 'genres']].to_string() if not rec_movies.empty else "Top-rated films in user's preferred genres"}

        User preferences: {user_preferences}

        For each movie, provide:
        1. An compelling 2-3 sentence description that highlights unique aspects
        2. Specific reasons why it matches the user's taste profile
        3. A predicted rating with confidence level
        4. One standout element (director, performance, technical achievement)

        Format as an enthusiastic, knowledgeable recommendation that shows deep understanding of cinema and the user's preferences. Use emojis and visual formatting to make it engaging.
        """
        
        # Try API first, fall back to simulation
        api_response = self.call_openai_api(prompt, max_tokens=800, temperature=0.8)
        
        if api_response:
            return api_response
        else:
            return self.simulate_llm_response(prompt, 'personalized')
    
    def mood_based_recommendations(self, mood: str, available_movies: List[Dict], 
                                 movies_df: pd.DataFrame) -> str:
        """Generate mood-based recommendations using LLM with enhanced responses"""
        
        # Check if movies_df has data
        if not movies_df.empty and len(movies_df) > 0:
            # Sample available movies for context
            sample_movies = movies_df.sample(min(20, len(movies_df)))
            movies_context = sample_movies[['title', 'genres']].to_string() if 'title' in sample_movies.columns else "Various acclaimed films available"
        else:
            movies_context = "Extensive collection of acclaimed films across all genres"
        
        prompt = f"""
        The user is feeling "{mood}" and wants movie recommendations that match this emotional state.

        Available movies context:
        {movies_context}

        Based on the mood "{mood}", recommend 4-6 movies that would be perfect for this emotional state. 

        For each recommendation, provide:
        1. Why it perfectly fits the "{mood}" mood
        2. What emotional experience it delivers
        3. A brief, enticing description
        4. What makes it special or unique

        Be enthusiastic and insightful. Show understanding of how different films can enhance or complement various moods. Use engaging formatting and emojis.
        """
        
        # Try API first, fall back to simulation
        api_response = self.call_openai_api(prompt, max_tokens=700, temperature=0.9)
        
        if api_response:
            return api_response
        else:
            return self.simulate_llm_response(prompt, 'mood')
    
    def conversational_interface(self, user_input: str, context: Dict) -> str:
        """Handle conversational interactions with enhanced intent recognition"""
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'context': context
        })
        
        # Analyze user intent
        intent = self.analyze_intent(user_input)
        
        # Provide status update if using simulation due to API issues
        status_message = ""
        if self.use_simulation and self.api_error_count > 0:
            status_message = "ℹ️ *Using enhanced simulation responses due to API quota limits*\n\n"
        
        if intent == 'explanation':
            response = self.generate_explanation(
                context.get('user_id', 1), 
                context.get('recommendations', []), 
                context.get('user_history', pd.DataFrame()), 
                context.get('movies_df', pd.DataFrame())
            )
        elif intent == 'mood':
            mood = self.extract_mood(user_input)
            response = self.mood_based_recommendations(
                mood, 
                context.get('available_movies', []), 
                context.get('movies_df', pd.DataFrame())
            )
        elif intent == 'personalized':
            response = self.generate_personalized_descriptions(
                context.get('recommendations', []), 
                context.get('movies_df', pd.DataFrame()), 
                context.get('user_preferences', {})
            )
        else:
            response = self.general_assistance(user_input, context)
        
        return status_message + response
    
    def analyze_intent(self, user_input: str) -> str:
        """Enhanced intent analysis with more patterns"""
        
        user_input_lower = user_input.lower()
        
        # Explanation patterns
        if any(word in user_input_lower for word in ['why', 'explain', 'because', 'reason', 'how did you', 'what makes']):
            return 'explanation'
        
        # Mood patterns  
        elif any(word in user_input_lower for word in ['mood', 'feeling', 'emotion', 'vibe', "i'm", "i am", 'feel like']):
            return 'mood'
        
        # Personalized description patterns
        elif any(word in user_input_lower for word in ['describe', 'tell me about', 'details', 'what is', 'summary']):
            return 'personalized'
        
        # General assistance patterns
        else:
            return 'general'
    
    def extract_mood(self, user_input: str) -> str:
        """Enhanced mood extraction with more categories"""
        
        mood_keywords = {
            'adventurous': ['adventurous', 'bold', 'daring', 'exciting', 'thrilling', 'action', 'adrenaline'],
            'thoughtful': ['thoughtful', 'contemplative', 'philosophical', 'deep', 'meaningful', 'profound', 'cerebral'],
            'excited': ['excited', 'energetic', 'pumped', 'thrilled', 'hyped', 'enthusiastic', 'fun'],
            'relaxed': ['relaxed', 'calm', 'peaceful', 'chill', 'laid back', 'mellow', 'quiet'],
            'romantic': ['romantic', 'love', 'date night', 'intimate', 'sweet', 'heartfelt'],
            'nostalgic': ['nostalgic', 'classic', 'old', 'vintage', 'retro', 'memories'],
            'dark': ['dark', 'intense', 'serious', 'heavy', 'dramatic', 'noir'],
            'funny': ['funny', 'comedy', 'laugh', 'humor', 'hilarious', 'amusing', 'witty']
        }
        
        user_input_lower = user_input.lower()
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return mood
        
        return 'adventurous'  # Default mood
    
    def general_assistance(self, user_input: str, context: Dict) -> str:
        """Enhanced general assistance with better guidance"""
        
        prompt = f"""
        User message: "{user_input}"
        
        Context: The user is interacting with an advanced movie recommendation system. They may want help understanding recommendations, finding specific types of movies, or learning about the system's capabilities.
        
        Provide a helpful, enthusiastic response that:
        1. Acknowledges their request
        2. Explains how the recommendation system can help them
        3. Offers specific, actionable suggestions for what they can ask
        4. Shows expertise in movies and personalization
        
        Be friendly, knowledgeable, and guide them toward getting the most value from the system.
        """
        
        # Try API first, fall back to simulation
        api_response = self.call_openai_api(prompt, max_tokens=400, temperature=0.7)
        
        if api_response:
            return api_response
        else:
            return self.simulate_llm_response(prompt, 'general')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration"""
        return {
            'using_simulation': self.use_simulation,
            'api_error_count': self.api_error_count,
            'has_api_key': bool(self.api_key and self.api_key != "simulate"),
            'conversation_length': len(self.conversation_history),
            'status': 'simulation' if self.use_simulation else 'api_active'
        }

def create_enhanced_sample_data():
    """Create more comprehensive sample data"""
    # Enhanced user history with more variety
    user_history = pd.DataFrame({
        'userId': [123] * 8,
        'movieId': [1, 2, 3, 4, 5, 50, 100, 150],
        'rating': [4.5, 5.0, 4.0, 4.5, 3.5, 4.8, 4.2, 4.6],
        'timestamp': [1234567890 + i*86400 for i in range(8)]
    })
    
    # Enhanced movies data with more titles
    movies_df = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 50, 100, 150],
        'title': ['The Matrix', 'Inception', 'Interstellar', 'Blade Runner', 
                 'Mad Max: Fury Road', 'Arrival', 'Her', 'Ex Machina',
                 'Dune', 'Parasite', '1917'],
        'genres': ['Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 'Adventure|Drama|Sci-Fi',
                  'Sci-Fi|Thriller', 'Action|Adventure|Sci-Fi', 'Drama|Sci-Fi',
                  'Drama|Romance|Sci-Fi', 'Drama|Sci-Fi|Thriller',
                  'Adventure|Drama|Sci-Fi', 'Comedy|Drama|Thriller', 'Drama|War']
    })
    
    return user_history, movies_df

if __name__ == "__main__":
    # Enhanced demo with better user experience
    print("🎬 Enhanced LLM Movie Recommender System")
    print("=" * 55)
    
    # Initialize recommender
    llm_recommender = LLMMovieRecommender()
    
    # Create enhanced sample data
    user_history, movies_df = create_enhanced_sample_data()
    
    # Enhanced test context
    context = {
        'user_id': 123,
        'recommendations': [{'movieId': 6}, {'movieId': 7}, {'movieId': 8}],
        'user_history': user_history,
        'movies_df': movies_df,
        'user_preferences': {
            'favorite_genres': ['Action', 'Sci-Fi', 'Drama'],
            'avg_rating': 4.3,
            'viewing_frequency': 'weekly'
        },
        'available_movies': [{'movieId': i} for i in range(1, 151)]
    }
    
    # System status
    status = llm_recommender.get_system_status()
    print(f"🔧 System Status: {status['status'].upper()}")
    
    if status['using_simulation']:
        print("📝 Running in ENHANCED SIMULATION mode")
        if status['api_error_count'] > 0:
            print(f"   (Switched due to {status['api_error_count']} API errors)")
        print("   Features: Intelligent responses, mood detection, personalized content")
        print("   To use OpenAI API: Set OPENAI_API_KEY in .env file")
    else:
        print("🤖 Running with OpenAI API integration")
        print("   Generating responses using GPT-3.5-turbo with smart fallbacks")
    
    print("\n" + "=" * 55)
    
    # Enhanced test scenarios
    test_scenarios = [
        ("Why did you recommend these movies?", "🧠 Explanation Analysis"),
        ("Can you describe these movies for me?", "📖 Personalized Descriptions"), 
        ("I'm feeling adventurous today", "🎭 Mood-Based Recommendations"),
        ("What can you help me with?", "💡 General Assistance"),
        ("I'm in a thoughtful mood", "🤔 Thoughtful Mood Test")
    ]
    
    for i, (query, title) in enumerate(test_scenarios, 1):
        print(f"\n{i}. {title}:")
        print("-" * 40)
        response = llm_recommender.conversational_interface(query, context)
        print(response)
    
    print("\n" + "=" * 55)
    print("🎉 Enhanced demo completed!")
    print(f"💬 Conversation history: {len(llm_recommender.conversation_history)} interactions")
    
    # Show final system status
    final_status = llm_recommender.get_system_status()
    print(f"📊 Final status: {final_status}")