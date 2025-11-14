import logging
from typing import List, Dict, Any, Optional

from app.config import settings
from app.services.s3_client import s3_client
from app.services.embeddings_service import embeddings_service
from app.services.qdrant_client import qdrant_client

logger = logging.getLogger(__name__)


class VectorDBSyncService:
    """
    Service for syncing movie data to vector database.
    """
    
    def __init__(self):
        self.batch_size = settings.VECTOR_BATCH_SIZE
    
    async def sync_from_s3(self, force_recreate: bool = False) -> Dict[str, Any]:
        """
        Sync movie data from S3 to Qdrant vector database.
        
        Args:
            force_recreate: If True, delete and recreate the collection
            
        Returns:
            Dictionary with sync status and statistics
        """
        result = {
            "status": "failed",
            "message": "",
            "movies_processed": 0,
            "movies_indexed": 0,
            "errors": []
        }
        
        try:
            # Step 1: Check Qdrant connection
            logger.info("Checking Qdrant connection...")
            if not await qdrant_client.health_check():
                result["message"] = "Failed to connect to Qdrant"
                return result
            
            # Step 2: Handle collection
            collection_exists = await qdrant_client.collection_exists()
            
            if force_recreate and collection_exists:
                logger.info("Deleting existing collection...")
                await qdrant_client.delete_collection()
                collection_exists = False
            
            if not collection_exists:
                logger.info("Creating new collection...")
                if not await qdrant_client.create_collection(settings.VECTOR_DIMENSION):
                    result["message"] = "Failed to create collection"
                    return result
            else:
                logger.info("Using existing collection")
            
            # Step 3: Download movie data from S3
            logger.info("Downloading movie data from S3...")
            df = s3_client.download_movie_data()
            
            if df is None or df.empty:
                result["message"] = "Failed to download movie data from S3"
                return result
            
            result["movies_processed"] = len(df)
            logger.info(f"Downloaded {len(df)} movies")
            
            # Step 4: Prepare texts for embedding
            logger.info("Preparing movie texts for embedding...")
            texts = []
            movie_metadata = []
            
            for idx, row in df.iterrows():
                text = embeddings_service.prepare_movie_text(row.to_dict())
                texts.append(text)
                
                # Store metadata
                metadata = {
                    "movie_id": str(row.get("movie_id", idx)),
                    "tmdb_id": str(row.get("tmdb_id", "")),
                    "title": str(row.get("title", "")),
                    "original_title": str(row.get("original_title", "")),
                    "overview": str(row.get("overview", "")),
                    "release_date": str(row.get("release_date", "")),
                    "genres": str(row.get("genres", "")),
                    "keywords": str(row.get("keywords", "")),
                    "runtime_minutes": float(row.get("runtime_minutes", 0)) if row.get("runtime_minutes") else 0,
                    "vote_average": float(row.get("vote_average", 0)) if row.get("vote_average") else 0,
                    "popularity": float(row.get("popularity", 0)) if row.get("popularity") else 0,
                }
                movie_metadata.append(metadata)
            
            # Step 5: Generate embeddings
            logger.info("Generating embeddings with Gemini...")
            embeddings = embeddings_service.generate_embeddings_batch(
                texts, 
                batch_size=self.batch_size
            )
            
            # Step 6: Prepare points for Qdrant
            logger.info("Preparing points for Qdrant...")
            points = []
            point_id = 0  # Sequential ID counter
            
            for i, (embedding, metadata) in enumerate(zip(embeddings, movie_metadata)):
                if embedding is None:
                    logger.warning(f"Skipping movie {metadata['movie_id']} due to embedding failure")
                    result["errors"].append(f"Failed to embed movie {metadata['movie_id']}")
                    continue
                
                # Use sequential integer as ID (Qdrant requirement)
                # movie_id is stored in payload for retrieval
                point = {
                    "id": point_id,
                    "vector": embedding,
                    "payload": metadata
                }
                points.append(point)
                point_id += 1
            
            # Step 7: Upload to Qdrant in batches
            logger.info(f"Uploading {len(points)} points to Qdrant...")
            
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                if await qdrant_client.upsert_points(batch):
                    result["movies_indexed"] += len(batch)
                    logger.info(f"Uploaded batch {i//self.batch_size + 1}/{(len(points)-1)//self.batch_size + 1}")
                else:
                    logger.error(f"Failed to upload batch {i//self.batch_size + 1}")
            
            # Step 8: Verify
            count = await qdrant_client.count_points()
            logger.info(f"Total points in collection: {count}")
            
            result["status"] = "success"
            result["message"] = f"Successfully indexed {result['movies_indexed']} movies"
            
        except Exception as e:
            logger.error(f"Error syncing vector database: {e}")
            result["message"] = f"Error: {str(e)}"
        
        return result
    
    async def search_movies(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for movies using natural language query.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of movie results with scores
        """
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = embeddings_service.generate_embedding(query)
            
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in Qdrant
            logger.info("Searching in Qdrant...")
            results = await qdrant_client.search(
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "movie_id": result.get("id"),
                    "score": result.get("score"),
                    "metadata": result.get("payload", {})
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching movies: {e}")
            return []
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get vector database status.
        
        Returns:
            Status information
        """
        try:
            health = await qdrant_client.health_check()
            count = 0
            collection_info = None
            
            if health:
                exists = await qdrant_client.collection_exists()
                if exists:
                    count = await qdrant_client.count_points()
                    collection_info = await qdrant_client.get_collection_info()
            
            return {
                "healthy": health,
                "collection_exists": await qdrant_client.collection_exists() if health else False,
                "total_movies": count,
                "collection_info": collection_info
            }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }


# Global instance
vector_sync_service = VectorDBSyncService()
