import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Path, Query, Body

from app.config import settings
from app.models.schemas import (
    UserRecommendationResponse,
    MovieRecommendationResponse,
    UpdateInferenceRequest,
    UpdateInferenceResponse,
    TrainTriggerResponse,
    RecommendationItem,
)
from app.services.model_manager import model_manager
from app.services.recommender import recommend_for_user, recommend_for_movie, get_popular_items

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/recommendation/user/{user_id}", response_model=UserRecommendationResponse)
async def get_user_recommendations(
    user_id: str = Path(..., description="User identifier"),
    strategy: str = Query(default="best", pattern="^(best|mf|user_cf)$", description="Recommendation strategy"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
) -> UserRecommendationResponse:
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: User identifier
        strategy: Recommendation strategy (best, mf, or user_cf)
                 'best' automatically uses the model with better RMSE
        top_k: Number of recommendations to return
        
    Returns:
        User recommendations with scores
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Store original strategy for response
    original_strategy = strategy
    actual_strategy = strategy
    
    # Resolve 'best' to actual strategy
    if strategy == "best":
        actual_strategy = model_manager.get_best_strategy()
    
    try:
        recommendations = recommend_for_user(user_id, top_k, strategy)
        
        # Fallback to popular items if no recommendations
        if not recommendations:
            logger.warning(f"No recommendations for user {user_id}, using popular items")
            recommendations = get_popular_items(top_k)
        
        return UserRecommendationResponse(
            user_id=user_id,
            strategy=f"{original_strategy} ({actual_strategy})" if original_strategy == "best" else strategy,
            recommendations=[
                RecommendationItem(movie_id=r['movie_id'], score=r['score'])
                for r in recommendations
            ]
        )
        
    except Exception as e:
        logger.error(f"Error generating user recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendation/movie/{movie_id}", response_model=MovieRecommendationResponse)
async def get_movie_recommendations(
    movie_id: str = Path(..., description="Movie identifier"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
) -> MovieRecommendationResponse:
    """
    Get similar movies based on MF embeddings.
    
    Args:
        movie_id: Movie identifier
        top_k: Number of similar movies to return
        
    Returns:
        Similar movie recommendations
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        recommendations = recommend_for_movie(movie_id, top_k)
        
        # Fallback to popular items if no recommendations
        if not recommendations:
            logger.warning(f"No similar movies for {movie_id}, using popular items")
            recommendations = get_popular_items(top_k)
        
        return MovieRecommendationResponse(
            movie_id=movie_id,
            recommendations=[
                RecommendationItem(movie_id=r['movie_id'], score=r['score'])
                for r in recommendations
            ]
        )
        
    except Exception as e:
        logger.error(f"Error generating movie recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update_inference", response_model=UpdateInferenceResponse)
async def update_inference_models(
    request: UpdateInferenceRequest = Body(default=UpdateInferenceRequest())
) -> UpdateInferenceResponse:
    """
    Update inference models by downloading latest from W&B.
    
    Args:
        request: Optional request with specific artifact version
        
    Returns:
        Update status and loaded model info
    """
    try:
        artifact_version = request.artifact_version if request and request.artifact_version else None
        
        logger.info(f"Updating models (version: {artifact_version or 'best'})...")
        
        result = model_manager.load_models(artifact_version)
        
        return UpdateInferenceResponse(
            status="success",
            message="Models updated successfully",
            loaded_models=result['loaded_models'],
            artifact_version=result.get('artifact_version', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Failed to update models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainTriggerResponse)
async def trigger_training() -> TrainTriggerResponse:
    """
    Trigger training by calling the train service.
    
    Returns:
        Training trigger status
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.TRAIN_SERVICE_URL}/train"
            )
            response.raise_for_status()
            
        logger.info("Training triggered successfully")
        
        return TrainTriggerResponse(
            status="success",
            message="Training started on train service"
        )
        
    except httpx.HTTPError as e:
        logger.error(f"Failed to trigger training: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with train service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status with best model recommendation
    """
    models_loaded = model_manager.is_loaded()
    metadata = model_manager.get_metadata() if models_loaded else {}
    
    response = {
        "status": "ok",
        "models_loaded": models_loaded,
        "metadata": metadata
    }
    
    # Add best model recommendation
    if models_loaded:
        response["recommended_strategy"] = model_manager.get_best_strategy()
        response["best_rmse"] = metadata.get("best_rmse")
    
    return response
