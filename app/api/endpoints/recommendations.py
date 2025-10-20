from fastapi import Depends, HTTPException, APIRouter
from app.api.deps import verify_token

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Examples
@router.get("/")
async def get_recommendations(token_data: dict = Depends(verify_token)):
    user_id = token_data.get("sub")
    recommendations = [
        {"item_id": 1, "name": "Recommended Item 1"},
        {"item_id": 2, "name": "Recommended Item 2"},
    ]
    return {"user_id": user_id, "recommendations": recommendations}

@router.get("/health")
async def health_check():
    return {"status": "ok"}