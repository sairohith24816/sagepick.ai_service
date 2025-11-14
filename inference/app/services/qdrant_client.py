"""
Qdrant Vector Database Client using REST API.
This avoids SSL certificate issues with the Python client library.
"""
import logging
from typing import List, Dict, Any, Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantClient:
    """
    Qdrant client using REST API for vector operations.
    """
    
    def __init__(self):
        self.base_url = settings.QDRANT_URL
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.api_key = settings.QDRANT_API_KEY
        self.verify_ssl = settings.QDRANT_VERIFY_SSL
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["api-key"] = self.api_key
    
    async def health_check(self) -> bool:
        """
        Check if Qdrant is accessible.
        """
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/",
                    headers=self.headers
                )
                response.raise_for_status()
                logger.info("Qdrant health check passed")
                return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        """
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/collections/{self.collection_name}",
                    headers=self.headers
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    async def create_collection(self, vector_size: int = 768) -> bool:
        """
        Create a new collection with the specified vector size.
        """
        try:
            payload = {
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine"
                }
            }
            
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=30.0) as client:
                response = await client.put(
                    f"{self.base_url}/collections/{self.collection_name}",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                logger.info(f"Collection '{self.collection_name}' created successfully")
                return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    async def delete_collection(self) -> bool:
        """
        Delete the collection.
        """
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/collections/{self.collection_name}",
                    headers=self.headers
                )
                response.raise_for_status()
                logger.info(f"Collection '{self.collection_name}' deleted successfully")
                return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    async def upsert_points(self, points: List[Dict[str, Any]]) -> bool:
        """
        Upsert points to the collection.
        
        Args:
            points: List of point dictionaries with id, vector, and payload
        """
        try:
            payload = {
                "points": points
            }
            
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=300.0) as client:
                response = await client.put(
                    f"{self.base_url}/collections/{self.collection_name}/points",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                logger.info(f"Upserted {len(points)} points to collection")
                return True
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with id, score, and payload
        """
        try:
            payload = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True
            }
            
            if score_threshold is not None:
                payload["score_threshold"] = score_threshold
            
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/collections/{self.collection_name}/points/search",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("result", [])
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    async def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """
        Get collection information including count of points.
        """
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl, timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/collections/{self.collection_name}",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    async def count_points(self) -> int:
        """
        Get the count of points in the collection.
        """
        info = await self.get_collection_info()
        if info and "result" in info:
            return info["result"].get("points_count", 0)
        return 0


# Global instance
qdrant_client = QdrantClient()
