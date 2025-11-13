# SagePick Inference Service

Inference service for SagePick recommendation models. Serves personalized recommendations using trained models.

## Features

- Loads models from Weights & Biases artifacts
- Provides personalized user recommendations
- Finds similar movies based on embeddings
- Hot-swaps models without downtime
- Thread-safe model caching
- Triggers training on train service

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
   - W&B API key and project
   - Train service URL

## Running the Service

Start the service:
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001
```

Or with reload for development:
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

### GET /recommendation/user/{user_id}
Get personalized recommendations for a user.

**Query Parameters:**
- `strategy`: Recommendation strategy (`best`, `mf`, or `user_cf`, default: `best`)
  - `best`: Automatically uses the model with better RMSE from training
  - `mf`: Matrix Factorization
  - `user_cf`: User Collaborative Filtering
- `top_k`: Number of recommendations (1-50, default: 10)

**Response:**
```json
{
  "user_id": "123",
  "strategy": "best (mf)",
  "recommendations": [
    {"movie_id": "456", "score": 4.8},
    {"movie_id": "789", "score": 4.6}
  ]
}
```

### GET /recommendation/movie/{movie_id}
Get similar movies based on embeddings.

**Query Parameters:**
- `top_k`: Number of similar movies (1-50, default: 10)

**Response:**
```json
{
  "movie_id": "456",
  "recommendations": [
    {"movie_id": "789", "score": 0.95},
    {"movie_id": "101", "score": 0.89}
  ]
}
```

### POST /update_inference
Update models by downloading latest from W&B.

**Request Body (optional):**
```json
{
  "artifact_version": "v2"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Models updated successfully",
  "loaded_models": ["mf", "user_cf"],
  "artifact_version": "v2"
}
```

### POST /train
Trigger training on train service.

**Response:**
```json
{
  "status": "success",
  "message": "Training started on train service"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "metadata": {
    "artifact_version": "v1",
    "train_date": "2025-11-14T10:30:00",
    "n_users": 610,
    "n_items": 9724
  }
}
```

## Configuration

Key environment variables:

- `WANDB_PROJECT`: W&B project name
- `TRAIN_SERVICE_URL`: URL of train service
- `MODEL_CACHE_DIR`: Directory for caching downloaded models
- `DEFAULT_TOP_K`: Default number of recommendations (default: 10)
- `DEFAULT_STRATEGY`: Default recommendation strategy (default: mf)

## Model Strategies

### Matrix Factorization (mf)
- Fast and scalable
- Works well for large datasets
- Uses SVD-based factorization

### User Collaborative Filtering (user_cf)
- KNN-based approach
- Better for cold-start scenarios
- Uses Pearson correlation similarity

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
