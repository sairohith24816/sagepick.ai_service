import logging
from typing import Dict, Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import settings
from app.services.trainer import Trainer

router = APIRouter()
logger = logging.getLogger(__name__)


async def run_training_task():
    """
    Background task to run training and notify inference service.
    """
    try:
        # Run training
        trainer = Trainer()
        result = trainer.run_training()
        
        logger.info(f"Training completed: {result['artifact_version']}")
        
        # Notify inference service to update models
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.INFERENCE_SERVICE_URL}/update_inference",
                    json={"artifact_version": result['artifact_version']}
                )
                response.raise_for_status()
                logger.info("Inference service updated successfully")
        except Exception as e:
            logger.error(f"Failed to notify inference service: {e}")
            # Don't raise - training was successful even if notification failed
        
    except Exception as e:
        logger.error(f"Training task failed: {e}")
        raise


@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Trigger model training.
    
    Process:
    1. Fetch ratings.csv from MinIO
    2. Train MF and User-CF models
    3. Upload models to W&B
    4. Notify inference service to update
    
    Returns:
        Acknowledgment message
    """
    try:
        # Add training task to background
        background_tasks.add_task(run_training_task)
        
        return {
            "status": "training_started",
            "message": "Training pipeline started in background"
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Status message
    """
    return {"status": "ok"}
