"""
Answer generation agents (direct and final).
"""
import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def _filter_movie_titles(movie_list: List[str]) -> List[str]:
    """
    Filter out non-movie entries (themes, genres, keywords) from the movie list.
    
    Args:
        movie_list: List that may contain movie titles and/or keywords
        
    Returns:
        Filtered list containing only likely movie titles
    """
    if not movie_list:
        return []
    
    # Common keywords/genres/themes that are NOT movie titles
    # These should be filtered out
    non_movie_keywords = {
        # Genres
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'fantasy', 'horror', 'mystery', 'romance', 'sci-fi', 'thriller',
        'western', 'musical', 'war', 'biography', 'family', 'sport',
        # Themes/concepts
        'hero', 'villain', 'love', 'revenge', 'redemption', 'betrayal',
        'sacrifice', 'friendship', 'loyalty', 'courage', 'honor', 'justice',
        'freedom', 'destiny', 'fate', 'death', 'life', 'hope', 'fear',
        'dream', 'nightmare', 'reality', 'illusion', 'time', 'space',
        # Adjectives commonly used as themes
        'dark', 'light', 'epic', 'intense', 'emotional', 'powerful',
        'suspenseful', 'thrilling', 'scary', 'funny', 'sad', 'happy',
        # Common query words
        'movie', 'movies', 'film', 'films', 'cinema', 'picture', 'flick'
    }
    
    filtered = []
    for item in movie_list:
        if not item or not isinstance(item, str):
            continue
            
        # Normalize for comparison
        normalized = item.strip().lower()
        
        # Skip if empty
        if not normalized:
            continue
        
        # Skip single words that match common keywords/genres
        # (but allow multi-word entries as they're more likely real titles)
        words = normalized.split()
        if len(words) == 1 and normalized in non_movie_keywords:
            logger.warning(f"Filtered out keyword/theme from movies list: '{item}'")
            continue
        
        # Skip if it's just a genre/theme word without capitalization hints
        # Real movie titles usually have proper capitalization
        if len(words) == 1 and normalized == item:  # all lowercase single word
            logger.warning(f"Filtered out lowercase single word from movies list: '{item}'")
            continue
            
        # Keep this entry
        filtered.append(item)
    
    logger.info(f"Filtered movies: {len(movie_list)} -> {len(filtered)}")
    if len(movie_list) != len(filtered):
        logger.info(f"Removed: {set(movie_list) - set(filtered)}")
    
    return filtered


def generate_direct_answer(question: str, assessment: dict) -> dict:
    """
    Generate direct answer using fast LLM (high confidence path).
    
    Args:
        question: User's query
        assessment: Assessment data with suggested movies
        
    Returns:
        Dict with answer and movie list
    """
    try:
        system_prompt = """You are a movie recommendation expert. The user asked about movies and we have high confidence in our answer.

Provide a helpful response that:
1. Briefly acknowledges their query
2. Lists 5-10 recommended movies with short descriptions
3. Is concise and friendly

Format your response naturally, don't use numbered lists unless necessary."""

        human_prompt = """User Query: {question}

Suggested Movies: {suggested_movies}
Themes: {themes}

Provide your recommendation."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm = ChatOpenAI(
            model=settings.FAST_LLM,
            temperature=0.3,
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )
        
        chain = prompt | llm
        
        response = chain.invoke({
            "question": question,
            "suggested_movies": ", ".join(assessment.get("suggested_movies", [])),
            "themes": ", ".join(assessment.get("themes", []))
        })
        
        # Filter out any themes/keywords that snuck into suggested_movies
        raw_movies = assessment.get("suggested_movies", [])
        filtered_movies = _filter_movie_titles(raw_movies)
        
        return {
            "answer": response.content,
            "movies": filtered_movies
        }
        
    except Exception as e:
        logger.error(f"Direct answer generation error: {e}")
        return {
            "answer": "I apologize, but I encountered an error generating recommendations.",
            "movies": []
        }


def generate_final_answer(question: str, assessment: dict, research_data: dict) -> dict:
    """
    Generate final answer using smart LLM with vector research data.
    
    Args:
        question: User's query
        assessment: Initial assessment
        research_data: Results from semantic vector search
    
    Returns:
        Dict with answer and movie list
    """
    try:
        system_prompt = """You are an expert movie recommendation system. You have researched the user's query using a curated semantic vector database of movies.

Your task:
1. Synthesize the vector search findings
2. Recommend 5-10 movies that best match the query
3. Provide brief, engaging descriptions grounded in the provided metadata
4. Be confident and helpful

Focus on quality over quantity. Only recommend movies present in the research batch."""

        human_prompt = """User Query: {question}

== RESEARCH RESULTS ==

Vector Search Summary:
{vector_results}

Structured Movie Hits:
{movie_details}

====================

Based on this research, provide your movie recommendations."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm = ChatOpenAI(
            model=settings.SMART_LLM,
            temperature=0.4,
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )
        
        chain = prompt | llm
        
        # Format research data
        vector_results = research_data.get("vector_summary", "No vector results")
        movie_details = research_data.get("movie_details", [])
        
        formatted_details = []
        for movie in movie_details:
            title = movie.get("title", "Unknown")
            overview = movie.get("overview", "No overview available")
            genres = movie.get("genres", "")
            year = movie.get("release_date", "")[:4]
            score = movie.get("score", 0)
            formatted_details.append(
                f"- {title} ({year or 'N/A'}) | genres: {genres} | relevance: {score:.2f}\n  {overview[:200]}..."
            )
        structured_context = "\n".join(formatted_details) if formatted_details else "No structured movie hits available."
        
        response = chain.invoke({
            "question": question,
            "vector_results": vector_results,
            "movie_details": structured_context
        })
        
        # Extract movie names from research and filter
        raw_movies = research_data.get("movie_names", [])
        filtered_movies = _filter_movie_titles(raw_movies)
        
        return {
            "answer": response.content,
            "movies": filtered_movies
        }
        
    except Exception as e:
        logger.error(f"Final answer generation error: {e}")
        return {
            "answer": "I apologize, but I encountered an error generating recommendations.",
            "movies": []
        }
