import logging
from io import BytesIO
from typing import Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """
    Client for fetching datasets from S3-compatible storage.
    Works with AWS S3, MinIO, DigitalOcean Spaces, and other S3-compatible services.
    """
    
    def __init__(self):
        """
        Initialize S3 client with credentials from settings.
        """
        # Build client configuration
        config_kwargs = {
            'aws_access_key_id': settings.S3_ACCESS_KEY,
            'aws_secret_access_key': settings.S3_SECRET_KEY,
            'region_name': settings.S3_REGION,
        }
        
        # Add endpoint URL for S3-compatible services (MinIO, etc.)
        if settings.S3_ENDPOINT_URL:
            config_kwargs['endpoint_url'] = settings.S3_ENDPOINT_URL
        
        self.client = boto3.client('s3', **config_kwargs)
        self.bucket = settings.S3_BUCKET
        logger.info(f"S3 client initialized for bucket: {self.bucket}")
    
    def fetch_ratings(self) -> pd.DataFrame:
        """
        Fetch ratings.csv from S3 and return as DataFrame.
        
        Returns:
            DataFrame with columns: user_id, movie_id, rating
            
        Raises:
            ClientError: If file doesn't exist or connection fails
        """
        try:
            logger.info(f"Fetching {settings.S3_RATINGS_KEY} from S3...")
            
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=settings.S3_RATINGS_KEY
            )
            
            data = response['Body'].read()
            
            df = pd.read_csv(BytesIO(data))
            
            # Normalize column names
            df = df.rename(columns={
                'userId': 'user_id',
                'movieId': 'movie_id'
            })
            
            # Ensure required columns exist
            required_cols = ['user_id', 'movie_id', 'rating']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            # Convert to strings for consistency
            df['user_id'] = df['user_id'].astype(str)
            df['movie_id'] = df['movie_id'].astype(str)
            df['rating'] = df['rating'].astype(float)
            
            logger.info(f"Successfully fetched {len(df)} ratings")
            return df[required_cols]
            
        except ClientError as e:
            logger.error(f"S3 error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching ratings: {e}")
            raise
    
    def check_connection(self) -> bool:
        """
        Check if S3 connection is working and bucket exists.
        
        Returns:
            True if connection is successful
        """
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 connection successful")
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"S3 connection check failed: {e}")
            return False
