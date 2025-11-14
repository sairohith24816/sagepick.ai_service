"""
Research agent for deep search using vector and tavily tools.
"""
import logging
import asyncio
from typing import Dict, Any, List

from app.tools.vector_tool import vector_search_tool
from app.tools.tavily_tool import tavily_tool

logger = logging.getLogger(__name__)


async def deep_search_research(question: str, themes: List[str]) -> Dict[str, Any]:
    """
    Perform deep research using vector search and tavily.
    Runs both tools in parallel for speed.
    
    Args:
        question: User's query
        themes: Extracted themes from assessment
        
    Returns:
        Dict with combined research results
    """
    try:
        logger.info("Starting deep search research...")
        
        # Create search queries
        vector_query = question
        tavily_query = f"movies about {', '.join(themes[:3])}" if themes else question
        
        # Run both searches in parallel
        vector_task = vector_search_tool.search(
            query=vector_query,
            limit=10,
            score_threshold=0.6
        )
        
        tavily_task = tavily_tool.search(query=tavily_query)
        
        # Wait for both
        vector_results, tavily_results = await asyncio.gather(
            vector_task,
            tavily_task,
            return_exceptions=True
        )
        
        # Handle errors
        if isinstance(vector_results, Exception):
            logger.error(f"Vector search error: {vector_results}")
            vector_results = {"success": False, "movies": []}
        
        if isinstance(tavily_results, Exception):
            logger.error(f"Tavily search error: {tavily_results}")
            # Create empty MovieSearchResult-like dict
            tavily_results = {"source": "tavily", "movies": [], "summary": ""}
        
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
        
        # Convert tavily_results to dict if it's a Pydantic model
        if hasattr(tavily_results, 'dict'):
            tavily_results_dict = tavily_results.dict()
        else:
            tavily_results_dict = tavily_results
        
        # Format summaries
        vector_summary = vector_search_tool.format_results_for_agent(vector_results)
        tavily_summary = tavily_tool.format_results_for_agent(tavily_results)
        
        tavily_count = len(tavily_results_dict.get('movies', []))
        logger.info(f"Research complete: {len(movie_names)} movies from vector, {tavily_count} from tavily")
        
        return {
            "vector_results": vector_results,
            "tavily_results": tavily_results_dict,
            "vector_summary": vector_summary,
            "tavily_summary": tavily_summary,
            "movie_names": movie_names,
            "movie_details": movie_details  # Full movie info for response
        }
        
    except Exception as e:
        logger.error(f"Deep search research error: {e}")
        return {
            "vector_results": {},
            "tavily_results": {},
            "vector_summary": "Research failed",
            "tavily_summary": "Research failed",
            "movie_names": []
        }
