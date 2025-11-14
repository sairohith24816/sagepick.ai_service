"""
Core models for the agentic system.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class InitialAssessment(BaseModel):
    """Assessment result from the initial assessor agent."""
    
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score (0-100) in answering without research"
    )
    query_type: Literal["direct_movie", "thematic", "complex_scenario", "vague"] = Field(
        ...,
        description="Type of query: direct_movie, thematic, complex_scenario, or vague"
    )
    suggested_movies: List[str] = Field(
        default_factory=list,
        description="Movies you already know (if confidence > 50%)"
    )
    known_info: str = Field(
        ...,
        description="What you already know about this query"
    )
    missing_info: str = Field(
        ...,
        description="What information needs research"
    )
    themes: List[str] = Field(
        default_factory=list,
        description="Key themes/keywords extracted from query"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of your confidence score"
    )


class RouteDecision(BaseModel):
    """Routing decision based on assessment."""
    
    route: Literal["direct_answer", "deep_search"] = Field(
        ...,
        description="Which route to take: direct_answer or deep_search"
    )
    reason: str = Field(
        ...,
        description="Reason for this routing decision"
    )


class MovieSearchResult(BaseModel):
    """Result from a movie search tool."""
    
    source: str = Field(..., description="Source of the result (vector, tmdb, tavily)")
    movies: List[dict] = Field(default_factory=list, description="List of movie results")
    summary: str = Field(default="", description="Summary of findings")


class AgentState(BaseModel):
    """State object passed through the agent graph."""
    
    # Input
    question: str = Field(..., description="User's original question")
    
    # Assessment stage
    assessment: Optional[InitialAssessment] = None
    route: Optional[str] = None
    
    # Research stage (for deep search)
    vector_results: Optional[MovieSearchResult] = None
    tmdb_results: Optional[MovieSearchResult] = None
    tavily_results: Optional[MovieSearchResult] = None
    
    # Final output
    answer: Optional[str] = None
    recommended_movies: List[dict] = Field(default_factory=list)
    confidence: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True
