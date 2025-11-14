import logging
from typing import List, Dict, Any, Optional

from app.services.vector_sync_service import vector_sync_service

logger = logging.getLogger(__name__)


class VectorSearchTool:
    """
    Tool for semantic search in the movie vector database.
    To be used by the agentic system for deep search route.
    """
    
    def __init__(self):
        self.name = "vector_search"
        self.description = (
            "Search for movies using natural language queries. "
            "Use this tool when the user describes a plot, scenario, theme, mood, "
            "or provides keywords about movies they want to find. "
            "This tool understands context and semantics, not just exact matches."
        )
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = 0.7
    ) -> Dict[str, Any]:
        """
        Execute semantic search for movies.
        
        Args:
            query: Natural language search query (e.g., "sci-fi movies about time travel")
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1), lower means more results
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            logger.info(f"Vector search: '{query}' (limit={limit}, threshold={score_threshold})")
            
            results = await vector_sync_service.search_movies(
                query=query,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "movies": results
            }
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "movies": []
            }
    
    def format_results_for_agent(self, results: Dict[str, Any]) -> str:
        """
        Format search results for agent consumption.
        
        Args:
            results: Results from search method
            
        Returns:
            Formatted string describing the results
        """
        if not results.get("success"):
            return f"Search failed: {results.get('error', 'Unknown error')}"
        
        movies = results.get("movies", [])
        
        if not movies:
            return "No movies found matching the query."
        
        formatted = [f"Found {len(movies)} movies:\n"]
        
        for i, movie in enumerate(movies, 1):
            metadata = movie.get("metadata", {})
            score = movie.get("score", 0)
            
            title = metadata.get("title", "Unknown")
            overview = metadata.get("overview", "No overview available")
            genres = metadata.get("genres", "")
            release_date = metadata.get("release_date", "")
            vote_avg = metadata.get("vote_average", 0)
            
            formatted.append(
                f"{i}. {title} ({release_date[:4] if release_date else 'N/A'})\n"
                f"   Relevance: {score:.2f}\n"
                f"   Genres: {genres}\n"
                f"   Rating: {vote_avg}/10\n"
                f"   Overview: {overview[:150]}...\n"
            )
        
        return "\n".join(formatted)
    
    def get_movie_ids(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract movie IDs from search results.
        
        Args:
            results: Results from search method
            
        Returns:
            List of movie IDs
        """
        if not results.get("success"):
            return []
        
        return [
            movie.get("movie_id")
            for movie in results.get("movies", [])
            if movie.get("movie_id")
        ]


# Global instance
vector_search_tool = VectorSearchTool()
