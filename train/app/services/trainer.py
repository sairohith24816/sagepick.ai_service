import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import settings
from app.models.matrix_factorization import mf_fit, mf_evaluate
from app.models.user_collaborative import ucf_fit, ucf_evaluate
from app.services.s3_client import S3Client
from app.services.wandb_client import WandBClient

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates the complete training pipeline:
    - Fetch data from S3
    - Train models
    - Evaluate models
    - Upload to W&B
    """
    
    def __init__(self):
        self.s3_client = S3Client()
        self.wandb_client = WandBClient()
    
    def run_training(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary with training results and metadata
        """
        logger.info("Starting training pipeline...")
        
        try:
            # Step 1: Fetch data from S3
            logger.info("Step 1: Fetching data from S3")
            ratings_df = self.s3_client.fetch_ratings()
            
            # Step 2: Prepare data
            logger.info("Step 2: Preparing train/test split")
            train_df, test_df = train_test_split(
                ratings_df,
                test_size=settings.TEST_SIZE,
                random_state=settings.RANDOM_STATE
            )
            
            logger.info(f"Training set: {len(train_df)} ratings")
            logger.info(f"Test set: {len(test_df)} ratings")
            
            # Step 3: Train Matrix Factorization model
            logger.info("Step 3: Training Matrix Factorization model")
            mf_model = mf_fit(
                train_df,
                n_factors=settings.MF_N_FACTORS,
                n_iter=settings.MF_N_ITER
            )
            mf_rmse = mf_evaluate(mf_model, test_df)
            logger.info(f"MF RMSE: {mf_rmse:.4f}")
            
            # Step 4: Train User Collaborative Filtering model
            logger.info("Step 4: Training User Collaborative Filtering model")
            ucf_model = ucf_fit(
                train_df,
                k=settings.UCF_K_NEIGHBORS,
                sim_name=settings.UCF_SIMILARITY
            )
            ucf_rmse = ucf_evaluate(ucf_model, test_df)
            logger.info(f"User-CF RMSE: {ucf_rmse:.4f}")
            
            # Step 5: Prepare metadata
            n_users = len(ratings_df['user_id'].unique())
            n_items = len(ratings_df['movie_id'].unique())
            
            metadata = {
                'train_date': datetime.utcnow().isoformat(),
                'dataset_size': len(ratings_df),
                'train_size': len(train_df),
                'test_size': len(test_df),
                'n_users': n_users,
                'n_items': n_items,
                'mf_rmse': float(mf_rmse),
                'user_cf_rmse': float(ucf_rmse),
                'mf_n_factors': settings.MF_N_FACTORS,
                'mf_n_iter': settings.MF_N_ITER,
                'ucf_k_neighbors': settings.UCF_K_NEIGHBORS,
                'ucf_similarity': settings.UCF_SIMILARITY,
            }
            
            # Step 6: Upload to W&B
            logger.info("Step 5: Uploading to W&B")
            
            # Start W&B run
            config = {
                'mf_n_factors': settings.MF_N_FACTORS,
                'mf_n_iter': settings.MF_N_ITER,
                'ucf_k_neighbors': settings.UCF_K_NEIGHBORS,
                'ucf_similarity': settings.UCF_SIMILARITY,
                'test_size': settings.TEST_SIZE,
            }
            run_id = self.wandb_client.start_run(config=config)
            
            # Log metrics
            metrics = {
                'mf_rmse': mf_rmse,
                'user_cf_rmse': ucf_rmse,
                'dataset_size': len(ratings_df),
                'n_users': n_users,
                'n_items': n_items,
            }
            self.wandb_client.log_metrics(metrics)
            
            # Upload models
            artifact_version = self.wandb_client.upload_models(
                mf_model=mf_model,
                ucf_model=ucf_model,
                metadata=metadata
            )
            
            # Finish run
            self.wandb_client.finish_run()
            
            logger.info("Training pipeline completed successfully")
            
            return {
                'status': 'success',
                'run_id': run_id,
                'artifact_version': artifact_version,
                'metrics': metadata,
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            self.wandb_client.finish_run()
            raise
