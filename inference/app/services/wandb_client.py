import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import joblib
import wandb

from app.config import settings

logger = logging.getLogger(__name__)


class WandBClient:
    """
    Client for downloading model artifacts from Weights & Biases.
    """
    
    def __init__(self):
        """
        Initialize W&B client.
        """
        self.project = settings.WANDB_PROJECT
        self.entity = settings.WANDB_ENTITY
        
        if settings.WANDB_API_KEY:
            os.environ['WANDB_API_KEY'] = settings.WANDB_API_KEY
        
        # Initialize W&B API
        self.api = wandb.Api()
    
    def download_latest_models(self, cache_dir: Optional[str] = None) -> Tuple[dict, dict, dict]:
        """
        Download the best model artifacts from W&B using 'best' alias.
        
        Args:
            cache_dir: Directory to cache downloaded models
            
        Returns:
            Tuple of (mf_model, ucf_model, metadata)
        """
        try:
            # Get the best artifact using alias
            artifact_name = f"{self.project}/sagepick-models:best"
            if self.entity:
                artifact_name = f"{self.entity}/{artifact_name}"
            
            logger.info(f"Downloading artifact: {artifact_name}")
            artifact = self.api.artifact(artifact_name, type="model")
            
            # Download to cache directory
            if cache_dir is None:
                cache_dir = settings.MODEL_CACHE_DIR
            
            download_dir = Path(cache_dir) / f"artifact_{artifact.version}"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            artifact_dir = artifact.download(root=str(download_dir))
            artifact_path = Path(artifact_dir)
            
            # Load models
            mf_model = joblib.load(artifact_path / "mf_model.joblib")
            ucf_model = joblib.load(artifact_path / "user_cf_model.joblib")
            
            # Load metadata
            import json
            with open(artifact_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            metadata['artifact_version'] = artifact.version
            
            logger.info(f"Successfully loaded models from artifact v{artifact.version}")
            logger.info(f"Best model: {metadata.get('best_model', 'unknown')} (RMSE: {metadata.get('best_rmse', 'N/A')})")
            return mf_model, ucf_model, metadata
            
        except Exception as e:
            logger.error(f"Failed to download models from W&B: {e}")
            raise
    
    def download_specific_version(
        self,
        version: str,
        cache_dir: Optional[str] = None
    ) -> Tuple[dict, dict, dict]:
        """
        Download a specific version of model artifacts from W&B.
        
        Args:
            version: Artifact version (e.g., "v0", "v1")
            cache_dir: Directory to cache downloaded models
            
        Returns:
            Tuple of (mf_model, ucf_model, metadata)
        """
        try:
            # Get the specific artifact version
            artifact_name = f"{self.project}/sagepick-models:{version}"
            if self.entity:
                artifact_name = f"{self.entity}/{artifact_name}"
            
            logger.info(f"Downloading artifact: {artifact_name}")
            artifact = self.api.artifact(artifact_name, type="model")
            
            # Download to cache directory
            if cache_dir is None:
                cache_dir = settings.MODEL_CACHE_DIR
            
            download_dir = Path(cache_dir) / f"artifact_{version}"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            artifact_dir = artifact.download(root=str(download_dir))
            artifact_path = Path(artifact_dir)
            
            # Load models
            mf_model = joblib.load(artifact_path / "mf_model.joblib")
            ucf_model = joblib.load(artifact_path / "user_cf_model.joblib")
            
            # Load metadata
            import json
            with open(artifact_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            metadata['artifact_version'] = artifact.version
            
            logger.info(f"Successfully loaded models from artifact {version}")
            return mf_model, ucf_model, metadata
            
        except Exception as e:
            logger.error(f"Failed to download models from W&B: {e}")
            raise
