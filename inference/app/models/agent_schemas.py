"""
Schemas for agentic system endpoints.
"""
from typing import List
from pydantic import BaseModel, Field


class MovieSearchRequest(BaseModel):
    """Request for movie search."""
    query: str = Field(..., description="User's movie query", min_length=1)


class MovieSearchResponse(BaseModel):
    """Response from movie search."""
    answer: str = Field(..., description="Natural language answer")
    movies: List[str] = Field(default_factory=list, description="List of recommended movie titles")
    confidence: int = Field(..., description="Confidence score (0-100)")
    route_taken: str = Field(..., description="Which route was taken: direct_answer or research")
