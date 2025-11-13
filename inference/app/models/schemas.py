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
