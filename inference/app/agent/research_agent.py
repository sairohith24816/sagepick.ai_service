"""
Research agent for deep search using the vector tool.
"""
import logging
from typing import Dict, Any, List

from app.tools.vector_tool import vector_search_tool

logger = logging.getLogger(__name__)


async def deep_search_research(question: str, themes: List[str]) -> Dict[str, Any]:
    """
    Perform deep research using semantic vector search only.
    
    Args:
        question: User's query
        themes: Extracted themes from assessment
    
    Returns:
        Dict with research results from the vector database
    """
    try:
        logger.info("Starting deep search research (vector only)...")
        
        # Create a richer vector query by blending question with key themes
        theme_text = ", ".join(themes[:5]) if themes else ""
        if theme_text:
            vector_query = f"{question.strip()} | key themes: {theme_text}"
        else:
            vector_query = question
        
        vector_results = await vector_search_tool.search(
            query=vector_query,
            limit=10,
            score_threshold=0.6
        )
        
        # Handle errors
        if isinstance(vector_results, Exception):
            logger.error(f"Vector search error: {vector_results}")
            vector_results = {"success": False, "movies": []}
        
        # Extract movie names and metadata from vector results (PRIORITY SOURCE)
        movie_names = []
        movie_details = []  # Full movie info for final response
        
        if vector_results.get("success") and vector_results.get("movies"):
            for movie in vector_results["movies"][:10]:
                # Handle dict structure from vector search
                if isinstance(movie, dict):
                    metadata = movie.get("metadata", {})
                    title = metadata.get("title")
                    if title:
                        movie_names.append(title)
                        # Store full movie details
                        movie_details.append({
                            "movie_id": metadata.get("movie_id", ""),
                            "title": title,
                            "overview": metadata.get("overview", ""),
                            "genres": metadata.get("genres", ""),
                            "release_date": metadata.get("release_date", ""),
                            "vote_average": metadata.get("vote_average", 0),
                            "score": movie.get("score", 0)
                        })
        
        # Format summaries
        vector_summary = vector_search_tool.format_results_for_agent(vector_results)
        
        logger.info(f"Research complete: {len(movie_names)} movies from vector search")
        
        return {
            "vector_results": vector_results,
            "vector_summary": vector_summary,
            "movie_names": movie_names,
            "movie_details": movie_details  # Full movie info for response
        }
        
    except Exception as e:
        logger.error(f"Deep search research error: {e}")
        return {
            "vector_results": {},
            "vector_summary": "Research failed",
            "movie_names": []
        }
