from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.api.deps import verify_token
from app.core.recommender import (
    fallback_recommendations,
    model_status,
    recommend_for_movie,
    recommend_for_user,
)
from app.core.scheduler import trigger_now
from app.core.settings import settings

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/users/{user_id}")
async def user_recommendations(
    user_id: str,
    strategy: str = Query(default="mf", pattern="^(mf|user_cf)$"),
    top_k: int | None = Query(default=None, ge=1, le=50),
    _: dict = Depends(verify_token),
):
    chosen_top_k = top_k or settings.TOP_K_DEFAULT
    recs = recommend_for_user(user_id, chosen_top_k, strategy=strategy)
    if not recs:
        recs = fallback_recommendations(chosen_top_k)
    return {"user_id": str(user_id), "strategy": strategy, "items": recs}


@router.get("/movies/{movie_id}")
async def movie_recommendations(
    movie_id: str,
    top_k: int | None = Query(default=None, ge=1, le=50),
    _: dict = Depends(verify_token),
):
    chosen_top_k = top_k or settings.TOP_K_DEFAULT
    recs = recommend_for_movie(movie_id, chosen_top_k)
    if not recs:
        recs = fallback_recommendations(chosen_top_k)
    return {"movie_id": str(movie_id), "items": recs}


@router.post("/train")
async def manual_retrain(background_tasks: BackgroundTasks, _: dict = Depends(verify_token)):
    background_tasks.add_task(trigger_now)
    return {"detail": "Retraining started"}


@router.get("/status")
async def recommendations_status(_: dict = Depends(verify_token)):
    return model_status()


@router.get("/health")
async def health_check():
    return {"status": "ok"}