from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration settings for the inference service.
    Loads from environment variables or .env file.
    """
    
    # W&B Configuration
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: str = "sagepick-train"
    WANDB_ENTITY: Optional[str] = None
    
    # Train Service Configuration
    TRAIN_SERVICE_URL: str = "http://localhost:8000"
    
    # Model Cache Configuration
    MODEL_CACHE_DIR: str = "./models"
    
    # Recommendation Configuration
    DEFAULT_TOP_K: int = 10
    DEFAULT_STRATEGY: str = "best"  # Auto-selects the model with better RMSE
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
