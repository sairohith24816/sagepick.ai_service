from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = Field(default="changeme")
    SECRET_ISS: str = Field(default="sagepick")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    DATA_BACKEND: str = "local"
    LOCAL_RATINGS_PATH: str = "data/ml-100k/ratings.csv"
    LOCAL_MOVIES_PATH: str = "data/ml-100k/movies.csv"
    MODEL_DIR: str = "models"

    S3_BUCKET: Optional[str] = None
    S3_RATINGS_KEY: Optional[str] = None
    S3_MOVIES_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None

    WANDB_PROJECT: Optional[str] = None
    WANDB_ENTITY: Optional[str] = None
    WANDB_TAGS: List[str] = Field(default_factory=list)
    WANDB_GROUP: Optional[str] = None

    RETRAIN_DAY: str = "sunday"
    RETRAIN_HOUR_UTC: int = 2
    TOP_K_DEFAULT: int = 10

    AUTO_TRAIN_ON_STARTUP: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
