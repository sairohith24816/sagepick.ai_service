import logging
from typing import List, Tuple, Dict, Any
import numpy as np

from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)


def recommend_for_user(user_id: str, top_k: int = 10, strategy: str = "best") -> List[Dict[str, Any]]:
    """
    Generate recommendations for a user.
    
    Args:
        user_id: User identifier
        top_k: Number of recommendations to return
        strategy: Recommendation strategy ('best', 'mf', or 'user_cf')
                 'best' automatically uses the model with better RMSE
        
    Returns:
        List of recommendation dictionaries with movie_id and score
    """
    # Resolve 'best' strategy to actual model
    if strategy == "best":
        strategy = model_manager.get_best_strategy()
        logger.info(f"Using best strategy: {strategy}")
    
    model = model_manager.get_model(strategy)
    
    if model is None:
        logger.warning(f"Model {strategy} not loaded")
        return []
    
    try:
        if strategy == "mf":
            recommendations = _recommend_mf(model, user_id, top_k)
        elif strategy == "user_cf":
            recommendations = _recommend_user_cf(model, user_id, top_k)
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return []
        
        return [
            {"movie_id": movie_id, "score": score}
            for movie_id, score in recommendations
        ]
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []


def recommend_for_movie(movie_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Find similar movies using MF item embeddings.
    
    Args:
        movie_id: Movie identifier
        top_k: Number of similar movies to return
        
    Returns:
        List of similar movie dictionaries with movie_id and score
    """
    model = model_manager.get_model("mf")
    
    if model is None:
        logger.warning("MF model not loaded")
        return []
    
    try:
        recommendations = _similar_movies_mf(model, movie_id, top_k)
        
        return [
            {"movie_id": mid, "score": score}
            for mid, score in recommendations
        ]
        
    except Exception as e:
        logger.error(f"Error finding similar movies: {e}")
        return []


def _recommend_mf(model: Dict[str, Any], user_id: str, top_k: int) -> List[Tuple[str, float]]:
    """
    Generate MF recommendations for a user.
    """
    user_map = model['user_map']
    
    if user_id not in user_map:
        logger.warning(f"User {user_id} not in training data")
        return []
    
    u = user_map[user_id]
    rated = set(model['user_items'].get(user_id, set()))
    
    # Calculate scores for all items
    scores = []
    inv_item_map = {idx: item for item, idx in model['item_map'].items()}
    for i in range(len(model['item_factors'])):
        item_id = inv_item_map[i]
        if item_id not in rated:
            score = model['global_mean'] + float(
                np.dot(model['user_factors'][u], model['item_factors'][i])
            )
            scores.append((item_id, float(np.clip(score, 0.5, 5.0))))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def _recommend_user_cf(model: Dict[str, Any], user_id: str, top_k: int) -> List[Tuple[str, float]]:
    """
    Generate User-CF recommendations for a user.
    """
    user_map = model['user_map']
    
    if user_id not in user_map:
        logger.warning(f"User {user_id} not in training data")
        return []
    
    u = user_map[user_id]
    rated = set(model['user_items'].get(user_id, set()))
    R = model['R']
    R_c = model['R_centered']
    
    # Find neighbors
    distances, indices = model['nbrs'].kneighbors(
        R_c[u],
        n_neighbors=min(model['k'], R.shape[0])
    )
    
    # Calculate scores for all unrated items
    scores = []
    inv_item_map = {idx: item for item, idx in model['item_map'].items()}
    mu = model['user_means'][u]
    
    for i in range(R.shape[1]):
        item_id = inv_item_map[i]
        if item_id in rated:
            continue
        
        # Predict rating using neighbors
        numerator = 0.0
        denominator = 0.0
        for neighbor_idx, dist in zip(indices[0], distances[0]):
            if neighbor_idx == u:
                continue
            sim = 1 - dist
            if sim <= 0:
                continue
            neighbor_rating = R[neighbor_idx, i]
            if neighbor_rating == 0:
                continue
            neighbor_mean = model['user_means'][neighbor_idx]
            numerator += sim * (neighbor_rating - neighbor_mean)
            denominator += abs(sim)
        
        if denominator > 0:
            pred = mu + (numerator / denominator)
        else:
            pred = mu if mu > 0 else model['global_mean']
        
        scores.append((item_id, float(np.clip(pred, 0.5, 5.0))))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def _similar_movies_mf(model: Dict[str, Any], movie_id: str, top_k: int) -> List[Tuple[str, float]]:
    """
    Find similar movies using MF item embeddings.
    """
    item_map = model['item_map']
    
    if movie_id not in item_map:
        logger.warning(f"Movie {movie_id} not in training data")
        return []
    
    i = item_map[movie_id]
    item_vector = model['item_factors'][i]
    
    # Calculate cosine similarity with all other items
    similarities = []
    inv_item_map = {idx: item for item, idx in item_map.items()}
    
    for j in range(len(model['item_factors'])):
        if j == i:
            continue
        
        other_vector = model['item_factors'][j]
        
        # Cosine similarity
        dot_product = float(np.dot(item_vector, other_vector))
        norm_i = float(np.linalg.norm(item_vector))
        norm_j = float(np.linalg.norm(other_vector))
        
        if norm_i > 0 and norm_j > 0:
            similarity = dot_product / (norm_i * norm_j)
            other_movie_id = inv_item_map[j]
            similarities.append((other_movie_id, float(similarity)))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def get_popular_items(top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Get popular items as fallback recommendations.
    
    Args:
        top_k: Number of items to return
        
    Returns:
        List of popular item dictionaries
    """
    model = model_manager.get_model("mf")
    
    if model is None:
        return []
    
    try:
        # Count ratings per item
        item_counts = {}
        for user_id, items in model['user_items'].items():
            for item_id in items:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1
        
        # Sort by count
        popular = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"movie_id": movie_id, "score": float(count)}
            for movie_id, count in popular[:top_k]
        ]
        
    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        return []
