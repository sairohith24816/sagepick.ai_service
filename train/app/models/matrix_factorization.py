import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List, Tuple, Set


def mf_fit(ratings_df: pd.DataFrame, n_factors: int = 40, n_iter: int = 7) -> Dict[str, Any]:
    """
    Train a matrix factorization model using TruncatedSVD.
    
    Args:
        ratings_df: DataFrame with columns user_id, movie_id, rating
        n_factors: Number of latent factors
        n_iter: Number of SVD iterations
        
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
    
    global_mean = float(df['rating'].mean()) if len(df) else 3.0
    
    # Center ratings by global mean
    R_centered = R.copy()
    R_centered.data = R_centered.data - global_mean
    
    # Apply SVD
    n_components = min(n_factors, min(n_users, n_items) - 1) if n_users > 1 and n_items > 1 else 1
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)
    U = svd.fit_transform(R_centered)
    Vt = svd.components_
    S = np.diag(svd.singular_values_)
    
    # Compute latent factors
    user_factors = U @ np.sqrt(S)
    item_factors = (np.sqrt(S) @ Vt).T
    
    # Track rated items per user for filtering during recommendation
    user_items = df.groupby('user_id')['movie_id'].apply(lambda s: set(s.tolist())).to_dict()
    
    return {
        'user_map': user_map,
        'item_map': item_map,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'global_mean': global_mean,
        'user_items': user_items,
        'all_items': set(items),
    }


def mf_predict(model: Dict[str, Any], user_id: str, item_id: str) -> float:
    """
    Predict rating for a user-item pair.
    
    Args:
        model: Trained MF model
        user_id: User identifier
        item_id: Item identifier
        
    Returns:
        Predicted rating
    """
    um = model['user_map']
    im = model['item_map']
    if user_id not in um or item_id not in im:
        return float(model['global_mean'])
    u = um[user_id]
    i = im[item_id]
    pred = model['global_mean'] + float(np.dot(model['user_factors'][u], model['item_factors'][i]))
    return float(np.clip(pred, 0.5, 5.0))


def mf_evaluate(model: Dict[str, Any], ratings_df: pd.DataFrame) -> float:
    """
    Evaluate model using RMSE.
    
    Args:
        model: Trained MF model
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
        preds.append(mf_predict(model, uid, iid))
        acts.append(r)
    return float(np.sqrt(mean_squared_error(acts, preds))) if preds else 0.0


def mf_recommend_top_n(model: Dict[str, Any], user_id: str, n: int = 10) -> List[Tuple[str, float]]:
    """
    Generate top-N recommendations for a user.
    
    Args:
        model: Trained MF model
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
    
    # Calculate scores for all items
    scores = []
    inv_item_map = {idx: item for item, idx in model['item_map'].items()}
    for i in range(len(model['item_factors'])):
        item_id = inv_item_map[i]
        if item_id not in rated:
            score = model['global_mean'] + float(np.dot(model['user_factors'][u], model['item_factors'][i]))
            scores.append((item_id, float(np.clip(score, 0.5, 5.0))))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]
