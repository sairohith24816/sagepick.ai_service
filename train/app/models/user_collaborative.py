import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, List, Tuple


def ucf_fit(ratings_df: pd.DataFrame, k: int = 50, sim_name: str = 'pearson') -> Dict[str, Any]:
    """
    Train a user-based collaborative filtering model using KNN.
    
    Args:
        ratings_df: DataFrame with columns user_id, movie_id, rating
        k: Number of nearest neighbors
        sim_name: Similarity metric (pearson uses cosine on centered data)
        
    Returns:
        Dictionary containing trained model components
    """
    df = ratings_df[['user_id', 'movie_id', 'rating']].copy()
    
    users = df['user_id'].unique()
    items = df['movie_id'].unique()
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(items)}
    
    rows = df['user_id'].map(user_map).values
    cols = df['movie_id'].map(item_map).values
    data = df['rating'].values
    n_users = len(users)
    n_items = len(items)
    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=float)
    
    # Compute user means
    user_means = np.zeros(n_users, dtype=float)
    for u in range(n_users):
        start = R.indptr[u]
        end = R.indptr[u + 1]
        if end > start:
            user_means[u] = float(R.data[start:end].mean())
        else:
            user_means[u] = 0.0
    
    # Center ratings by user mean
    R_centered = R.copy().tocsr()
    for u in range(n_users):
        start = R_centered.indptr[u]
        end = R_centered.indptr[u + 1]
        if end > start:
            R_centered.data[start:end] -= user_means[u]
    
    # Train KNN model
    metric = 'cosine'
    n_neighbors = min(k, max(1, n_users - 1))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='auto')
    nbrs.fit(R_centered)
    
    # Track rated items per user
    user_items = df.groupby('user_id')['movie_id'].apply(lambda s: set(s.tolist())).to_dict()
    global_mean = float(df['rating'].mean()) if len(df) else 3.0
    
    return {
        'user_map': user_map,
        'item_map': item_map,
        'R': R,
        'R_centered': R_centered,
        'user_means': user_means,
        'nbrs': nbrs,
        'k': k,
        'user_items': user_items,
        'all_items': set(items),
        'global_mean': global_mean,
    }


def ucf_predict(model: Dict[str, Any], user_id: str, item_id: str) -> float:
    """
    Predict rating for a user-item pair using weighted average of neighbors.
    
    Args:
        model: Trained UCF model
        user_id: User identifier
        item_id: Item identifier
        
    Returns:
        Predicted rating
    """
    um = model['user_map']
    im = model['item_map']
    
    if user_id not in um:
        return float(model['global_mean'])
    if item_id not in im:
        mu = model['user_means'][um[user_id]]
        base = mu if mu > 0 else model['global_mean']
        return float(np.clip(base, 0.5, 5.0))
    
    u = um[user_id]
    i = im[item_id]
    R = model['R']
    R_c = model['R_centered']
    mu = model['user_means'][u]
    
    # Find neighbors
    distances, indices = model['nbrs'].kneighbors(R_c[u], n_neighbors=min(model['k'], R.shape[0]))
    neighbors = indices[0]
    dists = distances[0]
    
    # Calculate weighted prediction
    numerator = 0.0
    denominator = 0.0
    for neighbor_idx, dist in zip(neighbors, dists):
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
    
    return float(np.clip(pred, 0.5, 5.0))


def ucf_evaluate(model: Dict[str, Any], ratings_df: pd.DataFrame) -> float:
    """
    Evaluate model using RMSE.
    
    Args:
        model: Trained UCF model
        ratings_df: Test ratings DataFrame
        
    Returns:
        RMSE score
    """
    preds = []
    acts = []
    for row in ratings_df[['user_id', 'movie_id', 'rating']].itertuples(index=False):
        uid = str(row.user_id)
        iid = str(row.movie_id)
        r = float(row.rating)
        preds.append(ucf_predict(model, uid, iid))
        acts.append(r)
    return float(np.sqrt(mean_squared_error(acts, preds))) if preds else 0.0


def ucf_recommend_top_n(model: Dict[str, Any], user_id: str, n: int = 10) -> List[Tuple[str, float]]:
    """
    Generate top-N recommendations for a user.
    
    Args:
        model: Trained UCF model
        user_id: User identifier
        n: Number of recommendations
        
    Returns:
        List of (item_id, score) tuples
    """
    um = model['user_map']
    if user_id not in um:
        return []
    
    u = um[user_id]
    rated = set(model['user_items'].get(user_id, set()))
    
    # Calculate scores for all unrated items
    scores = []
    inv_item_map = {idx: item for item, idx in model['item_map'].items()}
    for i in range(model['R'].shape[1]):
        item_id = inv_item_map[i]
        if item_id not in rated:
            score = ucf_predict(model, user_id, item_id)
            scores.append((item_id, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]
