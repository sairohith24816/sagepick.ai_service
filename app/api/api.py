from fastapi import APIRouter
from app.api.endpoints import recommendations

router = APIRouter(prefix="/api/v1")
router.include_router(recommendations.router)