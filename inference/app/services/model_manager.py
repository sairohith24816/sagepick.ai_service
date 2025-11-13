import logging
from threading import Lock
from typing import Dict, Any, Optional

from app.services.wandb_client import WandBClient

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Thread-safe manager for loading and caching recommendation models.
    """
    
    def __init__(self):
        """
        Initialize model manager with empty cache.
        """
        self._models = {}
        self._metadata = {}
        self._lock = Lock()
        self.wandb_client = WandBClient()
    
    def load_models(self, artifact_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load models from W&B and update cache.
        Thread-safe operation.
        
        Args:
            artifact_version: Specific version to load, or None for latest
            
        Returns:
            Dictionary with status and loaded model info
        """
        with self._lock:
            try:
                logger.info(f"Loading models (version: {artifact_version or 'latest'})...")
                
                if artifact_version:
                    mf_model, ucf_model, metadata = self.wandb_client.download_specific_version(
                        artifact_version
                    )
                else:
                    mf_model, ucf_model, metadata = self.wandb_client.download_latest_models()
                
                # Update cache
                self._models = {
                    'mf': mf_model,
                    'user_cf': ucf_model,
                }
                self._metadata = metadata
                
                logger.info(f"Models loaded successfully: v{metadata.get('artifact_version')}")
                
                return {
                    'status': 'success',
                    'loaded_models': ['mf', 'user_cf'],
                    'artifact_version': metadata.get('artifact_version'),
                    'metadata': metadata,
                }
                
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise
    
    def get_model(self, strategy: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached model by strategy.
        Thread-safe operation.
        
        Args:
            strategy: Model strategy ('mf' or 'user_cf')
            
        Returns:
            Model dictionary or None if not loaded
        """
        with self._lock:
            return self._models.get(strategy)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get cached model metadata.
        Thread-safe operation.
        
        Returns:
            Metadata dictionary
        """
        with self._lock:
            return self._metadata.copy()
    
    def is_loaded(self) -> bool:
        """
        Check if models are loaded.
        
        Returns:
            True if models are loaded
        """
        with self._lock:
            return len(self._models) > 0
    
    def get_best_strategy(self) -> str:
        """
        Get the recommended strategy based on training metrics.
        
        Returns:
            Best strategy name ('mf' or 'user_cf')
        """
        with self._lock:
            best_model = self._metadata.get('best_model', 'mf')
            return best_model


# Global model manager instance
model_manager = ModelManager()
