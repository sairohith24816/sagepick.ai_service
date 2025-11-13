import boto3
from pathlib import Path
import sys
from typing import Union
from app.config import settings

# S3 Configuration
S3_CONFIG = {
    "bucket": "sagepick-datasets",
    "key": "datasets/ratings.csv",
    "endpoint_url": "https://storage.sagepick.in",
    "access_key": settings.S3_ACCESS_KEY,
    "secret_key": settings.S3_SECRET_KEY,
    "region_name": "ap-south-1",
}

def upload_ratings_to_s3(file_path: Union[str, Path]):
    """
    Upload ratings.csv file to S3 storage.
    
    Args:
        file_path: Path to the ratings.csv file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Uploading {file_path} to S3...")
    print(f"Bucket: {S3_CONFIG['bucket']}")
    print(f"Key: {S3_CONFIG['key']}")
    print(f"Endpoint: {S3_CONFIG['endpoint_url']}")
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_CONFIG['endpoint_url'],
            aws_access_key_id=S3_CONFIG['access_key'],
            aws_secret_access_key=S3_CONFIG['secret_key'],
            region_name=S3_CONFIG['region_name']
        )
        
        # Upload file
        s3_client.upload_file(
            str(file_path),
            S3_CONFIG['bucket'],
            S3_CONFIG['key']
        )
        
        print(f"✓ Successfully uploaded {file_path.name} to S3!")
        print(f"S3 URI: s3://{S3_CONFIG['bucket']}/{S3_CONFIG['key']}")
        
        # Try to verify upload by checking object metadata (optional)
        try:
            response = s3_client.head_object(
                Bucket=S3_CONFIG['bucket'],
                Key=S3_CONFIG['key']
            )
            file_size = response['ContentLength']
            print(f"Verified - File size: {file_size:,} bytes")
        except Exception as verify_error:
            # Verification failed but upload was successful
            print(f"Note: Could not verify upload (permissions issue), but upload completed successfully")
        
    except Exception as e:
        print(f"✗ Error uploading file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Default path to ratings.csv in the data folder
    ratings_file = Path(__file__).parent / "data/ratings.csv"
    
    # Allow custom path as command line argument
    if len(sys.argv) > 1:
        ratings_file = Path(sys.argv[1])
    
    upload_ratings_to_s3(ratings_file)
