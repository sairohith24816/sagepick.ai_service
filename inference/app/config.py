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
    
    # Qdrant Configuration
    QDRANT_URL: str = "https://qdrant.sagepick.in"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "movies"
    QDRANT_VERIFY_SSL: bool = True
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    
    # S3/MinIO Configuration (matching train service)
    S3_ENDPOINT_URL: Optional[str] = None  # e.g., "https://storage.sagepick.in"
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_BUCKET_NAME: str = "sagepick-data"
    S3_MOVIE_DATA_KEY: str = "movie_items.csv"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = True
    
    # Vector DB Configuration
    VECTOR_DIMENSION: int = 768  # Gemini text-embedding-004 dimension
    VECTOR_BATCH_SIZE: int = 100
    
    # LangChain/OpenRouter Configuration
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    FAST_LLM: str = "meta-llama/llama-3.1-8b-instruct:free"  # Fast model for assessment
    SMART_LLM: str = "meta-llama/llama-3.1-70b-instruct"  # Smart model for final answers
    
    # Agent Configuration
    CONFIDENCE_THRESHOLD: int = 70  # Score threshold for direct answer vs deep search
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
