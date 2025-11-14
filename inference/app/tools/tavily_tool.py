"""
Tavily Search Tool for movie research.
"""
import logging
from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import settings
from app.agent.models import MovieSearchResult

logger = logging.getLogger(__name__)


class TavilyMovieTool:
    """
    Tool for searching movie information using Tavily.
    """
    
    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.max_results = settings.MAX_SEARCH_RESULTS
        self._tool = None
    
    @property
    def tool(self):
        """Lazy initialization of Tavily tool."""
        if self._tool is None and self.api_key:
            try:
                # Import os to set environment variable for Pydantic validation
                import os
                os.environ["TAVILY_API_KEY"] = self.api_key
                
                self._tool = TavilySearchResults(
                    max_results=self.max_results,
                    api_key=self.api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize Tavily tool: {e}")
                return None
        return self._tool
    
    async def search(self, query: str) -> MovieSearchResult:
        """
        Search for movie information using Tavily.
        
        Args:
            query: Search query about movies
            
        Returns:
            MovieSearchResult with findings
        """
        if not self.tool:
            logger.error("Tavily tool not initialized - API key missing")
            return MovieSearchResult(
                source="tavily",
                movies=[],
                summary="Tavily search unavailable - API key not configured"
            )
        
        try:
            logger.info(f"Tavily search: {query}")
            
            # Execute search
            results = await self.tool.ainvoke({"query": query})
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            # Create summary
            if formatted_results:
                summary = f"Found {len(formatted_results)} web results about: {query}"
            else:
                summary = "No results found"
            
            logger.info(f"Tavily found {len(formatted_results)} results")
            
            return MovieSearchResult(
                source="tavily",
                movies=formatted_results,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return MovieSearchResult(
                source="tavily",
                movies=[],
                summary=f"Search failed: {str(e)}"
            )
    
    def format_results_for_agent(self, results: MovieSearchResult) -> str:
        """
        Format Tavily results for agent consumption.
        
        Args:
            results: Search results from Tavily
            
        Returns:
            Formatted string for the agent
        """
        if not results.movies:
            return "No web results found."
        
        formatted = [f"Tavily Web Search Results ({len(results.movies)} found):\n"]
        
        for i, result in enumerate(results.movies, 1):
            title = result.get("title", "Unknown")
            content = result.get("content", "No content")
            url = result.get("url", "")
            
            formatted.append(
                f"{i}. {title}\n"
                f"   Content: {content[:200]}...\n"
                f"   Source: {url}\n"
            )
        
        return "\n".join(formatted)


# Global instance
tavily_tool = TavilyMovieTool()
