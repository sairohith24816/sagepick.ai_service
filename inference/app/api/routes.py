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
    VectorSyncRequest,
    VectorSyncResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStatusResponse,
    MovieResult,
)
from app.models.agent_schemas import (
    MovieSearchRequest,
    MovieSearchResponse,
)
from app.services.model_manager import model_manager
from app.services.recommender import recommend_for_user, recommend_for_movie, get_popular_items
from app.services.vector_sync_service import vector_sync_service
from app.agent.graph import run_agent

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


# ==================== Vector Database Endpoints ====================

@router.post("/vector/sync", response_model=VectorSyncResponse)
async def sync_vector_database(
    request: VectorSyncRequest = Body(default=VectorSyncRequest())
) -> VectorSyncResponse:
    """
    Sync vector database from S3 movie data.
    Downloads movie CSV from S3, generates embeddings, and uploads to Qdrant.
    
    Args:
        request: Sync configuration (force_recreate flag)
        
    Returns:
        Sync status and statistics
    """
    try:
        logger.info(f"Starting vector DB sync (force_recreate={request.force_recreate})...")
        
        result = await vector_sync_service.sync_from_s3(
            force_recreate=request.force_recreate
        )
        
        return VectorSyncResponse(**result)
        
    except Exception as e:
        logger.error(f"Error syncing vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector/search", response_model=VectorSearchResponse)
async def search_vector_database(
    request: VectorSearchRequest
) -> VectorSearchResponse:
    """
    Search movies using natural language query.
    Uses semantic search to find relevant movies based on description.
    
    Args:
        request: Search parameters (query, limit, score_threshold)
        
    Returns:
        List of matching movies with relevance scores
    """
    try:
        results = await vector_sync_service.search_movies(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        return VectorSearchResponse(
            success=True,
            query=request.query,
            count=len(results),
            movies=[MovieResult(**r) for r in results]
        )
        
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector/status", response_model=VectorStatusResponse)
async def get_vector_database_status() -> VectorStatusResponse:
    """
    Get vector database status and statistics.
    
    Returns:
        Database health, collection info, and movie count
    """
    try:
        status = await vector_sync_service.get_status()
        
        return VectorStatusResponse(
            healthy=status.get("healthy", False),
            collection_exists=status.get("collection_exists", False),
            total_movies=status.get("total_movies", 0),
            collection_info=status.get("collection_info")
        )
        
    except Exception as e:
        logger.error(f"Error getting vector database status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Agentic Search Endpoint ====================

@router.post("/search", response_model=MovieSearchResponse)
async def search_movies(request: MovieSearchRequest) -> MovieSearchResponse:
    """
    Intelligent movie search using agentic system.
    
    Uses a multi-agent workflow:
    - Assessment: Analyzes query confidence (fast LLM)
    - High confidence (≥70%): Direct answer (fast LLM)
    - Low confidence (<70%): Deep research (vector + tavily) → final answer (smart LLM)
    
    Args:
        request: Search request with user query
        
    Returns:
        Natural language answer with recommended movies
    """
    try:
        logger.info(f"Agent search: {request.query}")
        
        # Run the agent graph
        result = await run_agent(request.query)
        
        return MovieSearchResponse(
            answer=result["answer"],
            movies=result["movies"],
            confidence=result["confidence"],
            route_taken=result["route_taken"]
        )
        
    except Exception as e:
        logger.error(f"Error in agent search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
