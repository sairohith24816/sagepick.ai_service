# SagePick Train Service

Training service for SagePick recommendation models. Handles model training, evaluation, and artifact management.

## Features

- Fetches training data from S3-compatible storage (AWS S3, MinIO, DigitalOcean Spaces, etc.)
- Trains two collaborative filtering models:
  - Matrix Factorization (SVD-based)
  - User-Based Collaborative Filtering (KNN-based)
- Evaluates models on test set
- Uploads trained models to Weights & Biases
- Notifies inference service when new models are ready

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Configure environment variables in `.env`:
   - S3 credentials and endpoint (supports AWS S3, MinIO, etc.)
   - W&B API key and project
   - Inference service URL

## Running the Service

Start the service:
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Or with reload for development:
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### POST /train
Trigger model training pipeline.

**Response:**
```json
{
  "status": "training_started",
  "message": "Training pipeline started in background"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Training Pipeline

1. **Fetch Data**: Downloads `ratings.csv` from S3-compatible storage
2. **Prepare Data**: Creates train/test split
3. **Train MF Model**: Trains matrix factorization model
4. **Train User-CF Model**: Trains user collaborative filtering model
5. **Evaluate**: Calculates RMSE on test set
6. **Upload**: Saves models to W&B as artifacts
7. **Notify**: Calls inference service to update models

## Configuration

Key environment variables:

- `S3_ENDPOINT_URL`: S3 endpoint (empty for AWS, http://localhost:9000 for MinIO)
- `S3_ACCESS_KEY`: S3 access key
- `S3_SECRET_KEY`: S3 secret key
- `S3_BUCKET`: Bucket name containing datasets
- `S3_RATINGS_KEY`: Path to ratings.csv file
- `S3_REGION`: AWS region (default: us-east-1)
- `WANDB_PROJECT`: W&B project name
- `INFERENCE_SERVICE_URL`: URL of inference service
- `MF_N_FACTORS`: Number of latent factors for MF (default: 40)
- `UCF_K_NEIGHBORS`: Number of neighbors for User-CF (default: 50)

## Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black src/
uv run ruff check src/
```
