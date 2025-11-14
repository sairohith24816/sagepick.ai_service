from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class UserRecommendationRequest(BaseModel):
    """
    Request model for user recommendations.
    """
    strategy: str = Field(default="best", pattern="^(best|mf|user_cf)$")
    top_k: int = Field(default=10, ge=1, le=50)


class MovieRecommendationRequest(BaseModel):
    """
    Request model for movie recommendations.
    """
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendationItem(BaseModel):
    """
    Single recommendation item.
    """
    movie_id: str
    score: float


class UserRecommendationResponse(BaseModel):
    """
    Response model for user recommendations.
    """
    user_id: str
    strategy: str
    recommendations: List[RecommendationItem]


class MovieRecommendationResponse(BaseModel):
    """
    Response model for movie recommendations.
    """
    movie_id: str
    recommendations: List[RecommendationItem]


class UpdateInferenceRequest(BaseModel):
    """
    Request model for updating inference models.
    """
    artifact_version: Optional[str] = None


class UpdateInferenceResponse(BaseModel):
    """
    Response model for update inference endpoint.
    """
    status: str
    message: str
    loaded_models: List[str]
    artifact_version: Optional[str] = None


class TrainTriggerResponse(BaseModel):
    """
    Response model for train trigger endpoint.
    """
    status: str
    message: str


class VectorSyncRequest(BaseModel):
    """
    Request model for vector database sync.
    """
    force_recreate: bool = Field(default=False, description="Force recreate the collection")


class VectorSyncResponse(BaseModel):
    """
    Response model for vector database sync.
    """
    status: str
    message: str
    movies_processed: int
    movies_indexed: int
    errors: List[str] = []


class VectorSearchRequest(BaseModel):
    """
    Request model for vector search.
    """
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    score_threshold: Optional[float] = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Minimum similarity score"
    )


class MovieResult(BaseModel):
    """
    Single movie result from vector search.
    """
    movie_id: str
    score: float
    metadata: Dict[str, Any]


class VectorSearchResponse(BaseModel):
    """
    Response model for vector search.
    """
    success: bool
    query: str
    count: int
    movies: List[MovieResult]


class VectorStatusResponse(BaseModel):
    """
    Response model for vector database status.
    """
    healthy: bool
    collection_exists: bool
    total_movies: int
    collection_info: Optional[Dict[str, Any]] = None
