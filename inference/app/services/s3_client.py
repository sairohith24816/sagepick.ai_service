"""
S3/MinIO client for downloading movie data.
"""
import logging
from io import BytesIO
from typing import Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """
    Client for fetching movie data from S3-compatible storage.
    Works with AWS S3, MinIO, DigitalOcean Spaces, and other S3-compatible services.
    """
    
    def __init__(self):
        """
        Initialize S3 client with credentials from settings.
        """
        # Build client configuration
        config_kwargs = {
            'aws_access_key_id': settings.S3_ACCESS_KEY_ID,
            'aws_secret_access_key': settings.S3_SECRET_ACCESS_KEY,
        }
        
        # Add endpoint URL for S3-compatible services (MinIO, etc.)
        if settings.S3_ENDPOINT_URL:
            config_kwargs['endpoint_url'] = settings.S3_ENDPOINT_URL
        
        self.client = boto3.client('s3', **config_kwargs)
        self.bucket = settings.S3_BUCKET_NAME
        logger.info(f"S3 client initialized for bucket: {self.bucket}")
    
    def download_movie_data(self) -> Optional[pd.DataFrame]:
        """
        Download movie data CSV from S3/MinIO.
        
        Returns:
            DataFrame with movie data or None if failed
        """
        try:
            logger.info(f"Fetching {settings.S3_MOVIE_DATA_KEY} from S3...")
            
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=settings.S3_MOVIE_DATA_KEY
            )
            
            data = response['Body'].read()
            df = pd.read_csv(BytesIO(data))
            
            logger.info(f"Successfully downloaded {len(df)} movies from S3")
            return df
            
        except ClientError as e:
            logger.error(f"S3 error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading movie data from S3: {e}")
            return None
    
    def check_file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def check_connection(self) -> bool:
        """
        Check if S3 connection is working and bucket exists.
        
        Returns:
            True if connection is successful
        """
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info("S3 connection successful")
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"S3 connection check failed: {e}")
            return False


# Global instance
s3_client = S3Client()
