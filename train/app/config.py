from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration settings for the training service.
    Loads from environment variables or .env file.
    """
    
    # S3 Configuration (works with AWS S3, MinIO, DigitalOcean Spaces, etc.)
    S3_ENDPOINT_URL: Optional[str] = None  # e.g., "http://localhost:9000" for MinIO, None for AWS S3
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_BUCKET: str = "datasets"
    S3_RATINGS_KEY: str = "ratings.csv"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = True
    
    # W&B Configuration
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: str = "sagepick-train"
    WANDB_ENTITY: Optional[str] = None
    
    # Training Configuration
    MF_N_FACTORS: int = 40
    MF_N_ITER: int = 7
    UCF_K_NEIGHBORS: int = 50
    UCF_SIMILARITY: str = "pearson"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Inference Service Configuration
    INFERENCE_SERVICE_URL: str = "http://localhost:8001"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
