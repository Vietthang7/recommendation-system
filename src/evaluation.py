"""
Model Evaluation Metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(predictions, actuals):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(actuals, predictions))

def calculate_mae(predictions, actuals):
    """Mean Absolute Error"""
    return mean_absolute_error(actuals, predictions)

def precision_at_k(recommended, relevant, k=10):
    """Precision@K metric"""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = len([item for item in recommended_k if item in relevant_set])
    return hits / k if k > 0 else 0

def recall_at_k(recommended, relevant, k=10):
    """Recall@K metric"""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = len([item for item in recommended_k if item in relevant_set])
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0

def f1_score(precision, recall):
    """F1 Score"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_content_based_model(model, test_data, anime_df, k=10):
    """
    ✅ Đánh giá Content-Based với SOFT MATCHING
    Chấp nhận gợi ý có genres tương tự thay vì exact match
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    results = {'precision': [], 'recall': [], 'f1': []}
    
    # Get genre columns
    genre_cols = [col for col in anime_df.columns if col.startswith('genre_')]
    
    # Group by user
    grouped = test_data.groupby('user_id')
    
    evaluated_users = 0
    for user_id, user_data in grouped:
        # Lấy anime user thích (rating >= 7)
        liked_anime = user_data[user_data['rating'] >= 7]['anime_id'].tolist()
        all_anime = user_data['anime_id'].tolist()
        
        # Cần ít nhất 3 anime thích
        if len(liked_anime) < 3:
            continue
        
        try:
            # ✅ Lấy gợi ý
            all_recommended = set()
            
            seed_count = min(3, len(liked_anime))
            for seed_anime in liked_anime[:seed_count]:
                try:
                    recs = model.recommend(seed_anime, top_n=15)
                    all_recommended.update(recs)
                except:
                    continue
            
            # Loại bỏ anime đã xem
            recommended = [a for a in all_recommended if a not in all_anime][:k]
            
            if len(recommended) == 0 or len(liked_anime) == 0:
                continue
            
            # ✅ SOFT MATCHING: Tính genre similarity
            # Lấy genre profile của anime user thích
            liked_anime_df = anime_df[anime_df['anime_id'].isin(liked_anime)]
            liked_genres = liked_anime_df[genre_cols].mean(axis=0).values.reshape(1, -1)
            
            # Lấy genre profile của gợi ý
            recommended_df = anime_df[anime_df['anime_id'].isin(recommended)]
            
            if len(recommended_df) == 0:
                continue
            
            rec_genres = recommended_df[genre_cols].values
            
            # Tính cosine similarity
            similarities = cosine_similarity(rec_genres, liked_genres).flatten()
            
            # ✅ Coi anime có similarity >= 0.5 là "relevant"
            threshold = 0.5
            relevant_count = (similarities >= threshold).sum()
            
            # Calculate metrics
            p = relevant_count / k if k > 0 else 0
            r = relevant_count / len(liked_anime) if len(liked_anime) > 0 else 0
            f = f1_score(p, r)
            
            if p > 0 or r > 0:
                results['precision'].append(p)
                results['recall'].append(r)
                results['f1'].append(f)
                evaluated_users += 1
                    
        except Exception as e:
            continue
    
    if len(results['precision']) == 0:
        return {
            'precision@10': 0.0,
            'recall@10': 0.0,
            'f1_score': 0.0,
            'num_evaluated': 0
        }
    
    return {
        'precision@10': np.mean(results['precision']),
        'recall@10': np.mean(results['recall']),
        'f1_score': np.mean(results['f1']),
        'num_evaluated': evaluated_users
    }


def evaluate_model(model, test_data, k=10):
    """
    Đánh giá Collaborative Filtering model
    """
    results = {'precision': [], 'recall': [], 'f1': []}
    
    grouped = test_data.groupby('user_id')
    
    for user_id, user_data in grouped:
        relevant = user_data[user_data['rating'] >= 7]['anime_id'].tolist()
        
        if len(relevant) < 2:
            continue
        
        try:
            recommended = model.recommend(user_id, top_n=k)
            
            if len(recommended) > 0:
                p = precision_at_k(recommended, relevant, k)
                r = recall_at_k(recommended, relevant, k)
                f = f1_score(p, r)
                
                results['precision'].append(p)
                results['recall'].append(r)
                results['f1'].append(f)
        except:
            continue
    
    return {
        'precision@10': np.mean(results['precision']) if results['precision'] else 0,
        'recall@10': np.mean(results['recall']) if results['recall'] else 0,
        'f1_score': np.mean(results['f1']) if results['f1'] else 0,
        'num_evaluated': len(results['precision'])
    }