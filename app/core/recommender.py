import logging
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb
from boto3 import client as boto3_client
from joblib import dump, load
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from app.core.settings import settings


logger = logging.getLogger(__name__)

MODEL_CACHE = {}
MOVIES_DF = None
POPULAR_MOVIES = None
LATEST_TRAIN_INFO = {}
_MODEL_LOCK = Lock()


def _ensure_model_dir() -> Path:
    path = Path(settings.MODEL_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _s3_client():
    if not settings.S3_BUCKET:
        raise ValueError("S3 bucket not configured")
    kwargs = {"endpoint_url": settings.S3_ENDPOINT_URL} if settings.S3_ENDPOINT_URL else {}
    return boto3_client("s3", **kwargs)


def _load_csv(local_path: str, s3_key: Optional[str]) -> pd.DataFrame:
    backend = settings.DATA_BACKEND.lower()
    if backend == "s3":
        if not s3_key:
            raise ValueError("S3 key not configured for dataset")
        with BytesIO() as buffer:
            response = _s3_client().get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
            buffer.write(response["Body"].read())
            buffer.seek(0)
            return pd.read_csv(buffer)
    return pd.read_csv(local_path)


def _prepare_frames():
    ratings = _load_csv(settings.LOCAL_RATINGS_PATH, settings.S3_RATINGS_KEY)
    movies = _load_csv(settings.LOCAL_MOVIES_PATH, settings.S3_MOVIES_KEY)
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    movies = movies.rename(columns={"movieId": "movie_id"})
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["movie_id"] = ratings["movie_id"].astype(str)
    movies["movie_id"] = movies["movie_id"].astype(str)
    return {"ratings": ratings, "movies": movies}


def _build_mappings(ratings):
    users = ratings["user_id"].unique().tolist()
    items = ratings["movie_id"].unique().tolist()
    user_to_index = {user: idx for idx, user in enumerate(users)}
    item_to_index = {item: idx for idx, item in enumerate(items)}
    index_to_item = [""] * len(items)
    for item, idx in item_to_index.items():
        index_to_item[idx] = item
    return {
        "user_to_index": user_to_index,
        "item_to_index": item_to_index,
        "index_to_item": index_to_item,
    }


def _attach_indices(df, mappings):
    return df.assign(
        user_idx=df["user_id"].map(mappings["user_to_index"]),
        item_idx=df["movie_id"].map(mappings["item_to_index"]),
    ).dropna(subset=["user_idx", "item_idx"]).astype({"user_idx": int, "item_idx": int})


def _build_sparse_matrix(df, n_users, n_items):
    data = df["rating"].astype(np.float32).to_numpy()
    row_idx = df["user_idx"].to_numpy()
    col_idx = df["item_idx"].to_numpy()
    return csr_matrix((data, (row_idx, col_idx)), shape=(n_users, n_items))


def _train_user_cf_model(matrix, num_users):
    if num_users == 0:
        raise ValueError("No users available for training")
    n_neighbors = min(max(5, int(np.sqrt(num_users))), num_users)
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(matrix)
    return model, n_neighbors


def _train_nmf_model(matrix):
    min_dim = max(2, min(matrix.shape))
    proposed = max(10, min_dim // 2)
    components = min(50, proposed, min_dim)
    nmf = NMF(
        n_components=components,
        init="nndsvda",
        random_state=42,
        max_iter=200,
    )
    factors_users = nmf.fit_transform(matrix.astype(np.float64))
    factors_items = nmf.components_.T
    return factors_users, factors_items


def _predict_user_cf_rating(payload, user_idx: int, item_idx: int):
    model = payload["model"]
    matrix = payload["matrix"]
    user_means = payload["user_means"]
    distances, indices = model.kneighbors(matrix[user_idx], n_neighbors=payload["n_neighbors"])
    similarities = 1 - distances[0]
    baseline = user_means.get(user_idx, 0.0)
    numerator = 0.0
    denominator = 0.0
    for sim, neighbor_idx in zip(similarities, indices[0]):
        if neighbor_idx == user_idx or sim <= 0:
            continue
        neighbor_rating = matrix[neighbor_idx, item_idx]
        if neighbor_rating == 0:
            continue
        neighbor_mean = user_means.get(neighbor_idx, 0.0)
        numerator += sim * (neighbor_rating - neighbor_mean)
        denominator += abs(sim)
    return float(baseline + (numerator / denominator if denominator else 0.0))


def _predict_nmf_rating(payload, user_idx: int, item_idx: int):
    user_factors: np.ndarray = payload["user_factors"]
    item_factors: np.ndarray = payload["item_factors"]
    return float(np.dot(user_factors[user_idx], item_factors[item_idx]))


def _rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2))) if actual.size else float("nan")


def _mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted))) if actual.size else float("nan")


def _log_models_to_wandb(run, files):
    if not run:
        return
    for model_name, path in files.items():
        artifact = wandb.Artifact(model_name.replace("_", "-"), type="model")
        artifact.add_file(str(path))
        run.log_artifact(artifact)


def _save_models(models):
    model_dir = _ensure_model_dir()
    saved_paths = {}
    for key, model in models.items():
        path = model_dir / f"{key}.joblib"
        dump(model, path)
        saved_paths[key] = path
    return saved_paths


def _load_models_from_disk():
    model_dir = _ensure_model_dir()
    loaded = {}
    for key in ("user_cf", "mf"):
        path = model_dir / f"{key}.joblib"
        if path.exists():
            loaded[key] = load(path)
    return loaded


def _compute_popular_movies(ratings, movies):
    aggregated = (
        ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        .rename(columns={"mean": "avg_rating", "count": "rating_count"})
        .reset_index()
    )
    if aggregated.empty:
        return pd.DataFrame()
    threshold = max(10, int(aggregated["rating_count"].quantile(0.7)))
    popular = (
        aggregated[aggregated["rating_count"] >= threshold]
        .sort_values(["avg_rating", "rating_count"], ascending=[False, False])
        .head(50)
    )
    return popular.merge(movies, on="movie_id", how="left")


def train_and_refresh_models(manual_trigger: bool = False):
    logger.info("Starting recommendation model training", extra={"manual": manual_trigger})
    frames = _prepare_frames()
    ratings = frames["ratings"]
    movies = frames["movies"]

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    mappings = _build_mappings(ratings)
    train_indexed = _attach_indices(train_df, mappings)
    test_indexed = _attach_indices(test_df, mappings)

    num_users = len(mappings["user_to_index"])
    num_items = len(mappings["item_to_index"])
    user_item_matrix = _build_sparse_matrix(train_indexed, num_users, num_items)
    user_means = train_indexed.groupby("user_idx")["rating"].mean().to_dict()

    user_cf_model, n_neighbors = _train_user_cf_model(user_item_matrix, num_users)
    user_factors, item_factors = _train_nmf_model(user_item_matrix)

    eval_rows = test_indexed[["user_idx", "item_idx", "rating"]].to_numpy()
    actual = []
    user_cf_preds = []
    mf_preds = []
    payload_user_cf = {
        "model": user_cf_model,
        "matrix": user_item_matrix,
        "user_means": user_means,
        "n_neighbors": n_neighbors,
    }
    payload_mf = {
        "user_factors": user_factors,
        "item_factors": item_factors,
    }
    for user_idx, item_idx, rating in eval_rows:
        actual.append(float(rating))
        user_cf_preds.append(_predict_user_cf_rating(payload_user_cf, int(user_idx), int(item_idx)))
    mf_preds.append(_predict_nmf_rating(payload_mf, int(user_idx), int(item_idx)))

    actual_arr = np.array(actual, dtype=float)
    user_cf_arr = np.array(user_cf_preds, dtype=float)
    mf_arr = np.array(mf_preds, dtype=float)

    metrics = {
        "user_cf_rmse": _rmse(actual_arr, user_cf_arr),
        "user_cf_mae": _mae(actual_arr, user_cf_arr),
    "mf_rmse": _rmse(actual_arr, mf_arr),
    "mf_mae": _mae(actual_arr, mf_arr),
    }

    config = {
        "backend": settings.DATA_BACKEND,
        "ratings_rows": int(len(ratings)),
        "movies_rows": int(len(movies)),
        "manual_trigger": manual_trigger,
    }

    run = None
    if settings.WANDB_PROJECT:
        try:
            run = wandb.init(
                project=settings.WANDB_PROJECT,
                entity=settings.WANDB_ENTITY,
                group=settings.WANDB_GROUP,
                tags=settings.WANDB_TAGS,
                job_type="weekly-retraining",
                config=config,
            )
            run.log(metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb tracking disabled: %s", exc)
            run = None

    user_cf_payload = {
        "model": user_cf_model,
        "matrix": user_item_matrix,
        "user_to_index": mappings["user_to_index"],
        "item_to_index": mappings["item_to_index"],
        "index_to_item": mappings["index_to_item"],
        "user_means": user_means,
        "n_neighbors": n_neighbors,
    }
    mf_payload = {
        "user_item_matrix": user_item_matrix,
        "user_to_index": mappings["user_to_index"],
        "item_to_index": mappings["item_to_index"],
        "index_to_item": mappings["index_to_item"],
        "user_factors": user_factors,
        "item_factors": item_factors,
        "item_norms": np.linalg.norm(item_factors, axis=1) + 1e-8,
    }

    saved_paths = _save_models({"user_cf": user_cf_payload, "mf": mf_payload})
    _log_models_to_wandb(run, saved_paths)

    global MOVIES_DF, POPULAR_MOVIES, LATEST_TRAIN_INFO
    with _MODEL_LOCK:
        MODEL_CACHE["user_cf"] = user_cf_payload
        MODEL_CACHE["mf"] = mf_payload
        MOVIES_DF = movies
        POPULAR_MOVIES = _compute_popular_movies(ratings, movies)
        LATEST_TRAIN_INFO = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "config": config,
        }

    if run:
        run.log({"trained_at": datetime.now(timezone.utc).isoformat()})
        run.finish()

    logger.info("Completed recommendation model training", extra={"metrics": metrics})
    return LATEST_TRAIN_INFO


def initialize_recommender() -> None:
    loaded = _load_models_from_disk()
    if loaded.get("user_cf") and loaded.get("mf"):
        logger.info("Loading recommendation models from disk cache")
        frames = _prepare_frames()
        with _MODEL_LOCK:
            MODEL_CACHE.update(loaded)
            global MOVIES_DF, POPULAR_MOVIES
            MOVIES_DF = frames["movies"]
            POPULAR_MOVIES = _compute_popular_movies(frames["ratings"], MOVIES_DF)
            LATEST_TRAIN_INFO.update({"trained_at": "loaded-from-disk"})
        return
    if settings.AUTO_TRAIN_ON_STARTUP:
        logger.info("No cached models detected; triggering initial training")
        train_and_refresh_models()


def _resolve_movie_rows(movie_ids):
    if MOVIES_DF is None:
        return []
    subset = MOVIES_DF[MOVIES_DF["movie_id"].isin(movie_ids)]
    rows: List[Dict[str, Any]] = []
    for record in subset.itertuples(index=False):
        rows.append({
            "movie_id": str(record.movie_id),
            "title": getattr(record, "title", None),
            "genres": getattr(record, "genres", None),
        })
    return rows


def _format_recommendations(pairs):
    movie_ids = [movie_id for movie_id, _ in pairs]
    metadata = {row["movie_id"]: row for row in _resolve_movie_rows(movie_ids)}
    results = []
    for movie_id, score in pairs:
        record = metadata.get(movie_id, {"movie_id": movie_id, "title": None, "genres": None})
        results.append({
            "movie_id": record["movie_id"],
            "title": record.get("title"),
            "genres": record.get("genres"),
            "score": score,
        })
    return results


def _recommend_user_cf(user_id: str, top_k: int, payload):
    user_to_index = payload["user_to_index"]
    matrix = payload["matrix"]
    if user_id not in user_to_index:
        return []
    user_idx = user_to_index[user_id]
    rated_items = set(matrix[user_idx].indices.tolist())
    if matrix.shape[0] == 0:
        return []
    model = payload["model"]
    distances, indices = model.kneighbors(matrix[user_idx], n_neighbors=payload["n_neighbors"])
    similarities = 1 - distances[0]
    user_means: Dict[int, float] = payload["user_means"]
    baseline = user_means.get(user_idx, 0.0)
    scores: Dict[int, float] = defaultdict(float)
    weights: Dict[int, float] = defaultdict(float)
    for sim, neighbor_idx in zip(similarities, indices[0]):
        if neighbor_idx == user_idx or sim <= 0:
            continue
        neighbor_row = matrix[neighbor_idx]
        neighbor_mean = user_means.get(neighbor_idx, 0.0)
        for item_idx, rating in zip(neighbor_row.indices, neighbor_row.data):
            if item_idx in rated_items:
                continue
            scores[item_idx] += sim * (rating - neighbor_mean)
            weights[item_idx] += abs(sim)
    predictions: List[Tuple[str, float]] = []
    items = payload["index_to_item"]
    for item_idx, numerator in scores.items():
        denom = weights[item_idx]
        estimate = baseline + (numerator / denom if denom else 0.0)
        predictions.append((items[item_idx], float(estimate)))
    predictions.sort(key=lambda row: row[1], reverse=True)
    return _format_recommendations(predictions[:top_k])


def _recommend_mf(user_id: str, top_k: int, payload):
    user_to_index = payload["user_to_index"]
    if user_id not in user_to_index:
        return []
    user_idx = user_to_index[user_id]
    user_item_matrix = payload["user_item_matrix"]
    rated_indices = set(user_item_matrix[user_idx].indices.tolist())
    user_factors: np.ndarray = payload["user_factors"]
    item_factors: np.ndarray = payload["item_factors"]
    scores = user_factors[user_idx] @ item_factors.T
    scores = np.asarray(scores).flatten()
    for idx in rated_indices:
        scores[idx] = float("-inf")
    if top_k >= len(scores):
        candidate_indices = np.argsort(scores)[::-1]
    else:
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    items = payload["index_to_item"]
    pairs: List[Tuple[str, float]] = []
    for idx in candidate_indices:
        score = scores[idx]
        if not np.isfinite(score):
            continue
        pairs.append((items[idx], float(score)))
        if len(pairs) == top_k:
            break
    return _format_recommendations(pairs)


def recommend_for_user(user_id: str, top_k: Optional[int] = None, strategy: str = "mf"):
    top_k = top_k or settings.TOP_K_DEFAULT
    payload = MODEL_CACHE.get(strategy)
    if not payload:
        return []
    user_id_str = str(user_id)
    if strategy == "user_cf":
        recs = _recommend_user_cf(user_id_str, top_k, payload)
    else:
        recs = _recommend_mf(user_id_str, top_k, payload)
    return recs or fallback_recommendations(top_k)


def recommend_for_movie(movie_id: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K_DEFAULT
    payload = MODEL_CACHE.get("mf")
    if not payload:
        return []
    item_to_index = payload["item_to_index"]
    items = payload["index_to_item"]
    movie_id_str = str(movie_id)
    if movie_id_str not in item_to_index:
        return fallback_recommendations(top_k)
    item_idx = item_to_index[movie_id_str]
    item_factors: np.ndarray = payload["item_factors"]
    norms: np.ndarray = payload["item_norms"]
    target_vec = item_factors[item_idx]
    target_norm = norms[item_idx]
    similarities = (item_factors @ target_vec) / (norms * target_norm)
    similarities[item_idx] = float("-inf")
    if top_k >= len(similarities):
        candidate_idx = np.argsort(similarities)[::-1]
    else:
        candidate_idx = np.argpartition(similarities, -top_k)[-top_k:]
        candidate_idx = candidate_idx[np.argsort(similarities[candidate_idx])[::-1]]
    pairs: List[Tuple[str, float]] = []
    for idx in candidate_idx:
        score = similarities[idx]
        if not np.isfinite(score):
            continue
        pairs.append((items[idx], float(score)))
        if len(pairs) == top_k:
            break
    return _format_recommendations(pairs) or fallback_recommendations(top_k)


def fallback_recommendations(top_k: int):
    if POPULAR_MOVIES is None or POPULAR_MOVIES.empty:
        return []
    top_rows = POPULAR_MOVIES.head(top_k)
    output: List[Dict[str, Any]] = []
    for record in top_rows.itertuples(index=False):
        output.append({
            "movie_id": str(record.movie_id),
            "title": getattr(record, "title", None),
            "genres": getattr(record, "genres", None),
            "score": round(float(getattr(record, "avg_rating", 0.0)), 3),
        })
    return output


def model_status():
    return {
        "trained_models": [name for name, model in MODEL_CACHE.items() if model is not None],
        "latest_training": LATEST_TRAIN_INFO,
    }
