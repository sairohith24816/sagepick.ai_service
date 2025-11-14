"""
Embeddings service using Google Gemini API.
"""
import logging
from typing import List, Optional
import google.generativeai as genai

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingsService:
    """
    Service for generating embeddings using Google Gemini API.
    """
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = "models/text-embedding-004"
        
        # Configure the API
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully")
        else:
            logger.warning("Gemini API key not configured")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.api_key:
            logger.error("Gemini API key not configured")
            return None
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result.get('embedding')
            
            if not embedding:
                logger.warning("Empty embedding returned")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        if not self.api_key:
            logger.error("Gemini API key not configured")
            return [None] * len(texts)
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Batch API call
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                batch_embeddings = result.get('embedding', [])
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Add None for failed batch
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def prepare_movie_text(self, movie_data: dict) -> str:
        """
        Prepare text for embedding from movie data.
        Combines title, overview, genres, and keywords.
        
        Args:
            movie_data: Dictionary containing movie information
            
        Returns:
            Formatted text for embedding
        """
        parts = []
        
        # Title
        if title := movie_data.get("title"):
            parts.append(f"Title: {title}")
        
        # Overview
        if overview := movie_data.get("overview"):
            parts.append(f"Overview: {overview}")
        
        # Genres
        if genres := movie_data.get("genres"):
            parts.append(f"Genres: {genres}")
        
        # Keywords
        if keywords := movie_data.get("keywords"):
            parts.append(f"Keywords: {keywords}")
        
        # Release date
        if release_date := movie_data.get("release_date"):
            parts.append(f"Release Date: {release_date}")
        
        # Runtime
        if runtime := movie_data.get("runtime_minutes"):
            parts.append(f"Runtime: {runtime} minutes")
        
        return " | ".join(parts)


# Global instance
embeddings_service = GeminiEmbeddingsService()
