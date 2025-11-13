import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import wandb

from app.config import settings

logger = logging.getLogger(__name__)


class WandBClient:
    """
    Client for uploading model artifacts and logging metrics to Weights & Biases.
    """
    
    def __init__(self):
        """
        Initialize W&B client.
        """
        self.project = settings.WANDB_PROJECT
        self.entity = settings.WANDB_ENTITY
        self.run = None
        
        if settings.WANDB_API_KEY:
            os.environ['WANDB_API_KEY'] = settings.WANDB_API_KEY
    
    def start_run(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new W&B run.
        
        Args:
            config: Configuration dictionary to log
            
        Returns:
            Run ID
        """
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=config or {},
                reinit=True
            )
            logger.info(f"W&B run started: {self.run.id}")
            return self.run.id
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to current W&B run.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        if self.run is None:
            logger.warning("No active W&B run, skipping metric logging")
            return
        
        try:
            wandb.log(metrics)
            logger.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def upload_models(
        self,
        mf_model: Dict[str, Any],
        ucf_model: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Upload trained models as W&B artifacts.
        Uses consistent naming and marks best model with alias.
        
        Args:
            mf_model: Matrix Factorization model dictionary
            ucf_model: User Collaborative Filtering model dictionary
            metadata: Additional metadata to include
            
        Returns:
            Artifact version
        """
        if self.run is None:
            raise RuntimeError("No active W&B run")
        
        try:
            # Create temporary directory for artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save models
                mf_path = temp_path / "mf_model.joblib"
                ucf_path = temp_path / "user_cf_model.joblib"
                metadata_path = temp_path / "metadata.json"
                
                joblib.dump(mf_model, mf_path)
                joblib.dump(ucf_model, ucf_path)
                
                # Add timestamp to metadata
                metadata['upload_timestamp'] = datetime.utcnow().isoformat()
                
                # Determine best model based on RMSE
                mf_rmse = metadata.get('mf_rmse', float('inf'))
                ucf_rmse = metadata.get('user_cf_rmse', float('inf'))
                best_model = 'mf' if mf_rmse <= ucf_rmse else 'user_cf'
                metadata['best_model'] = best_model
                metadata['best_rmse'] = min(mf_rmse, ucf_rmse)
                
                # Save metadata as JSON
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create artifact with consistent name (no timestamp in name)
                artifact = wandb.Artifact(
                    name="sagepick-models",
                    type="model",
                    description="Trained recommendation models (MF + User-CF)",
                    metadata=metadata
                )
                
                # Add files to artifact
                artifact.add_file(str(mf_path))
                artifact.add_file(str(ucf_path))
                artifact.add_file(str(metadata_path))
                
                # Log artifact with aliases
                aliases = ["latest", "best"]
                self.run.log_artifact(artifact, aliases=aliases)
                
                # Wait for upload to complete
                artifact.wait()
                
                version = artifact.version
                logger.info(f"Models uploaded as artifact: sagepick-models:{version}")
                logger.info(f"Best model: {best_model} (RMSE: {metadata['best_rmse']:.4f})")
                logger.info(f"Aliases: {aliases}")
                return version
                
        except Exception as e:
            logger.error(f"Failed to upload models: {e}")
            raise
    
    def finish_run(self) -> None:
        """
        Finish the current W&B run.
        """
        if self.run is not None:
            try:
                self.run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
            finally:
                self.run = None
