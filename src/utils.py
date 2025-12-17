"""
Utility functions
"""
import pandas as pd
import numpy as np

def load_data(anime_path, rating_path):
    """Load anime and rating datasets"""
    anime_df = pd.read_csv(anime_path)
    rating_df = pd.read_csv(rating_path)
    return anime_df, rating_df

def create_user_item_matrix(rating_df, max_users=5000, max_items=2000):
    """Create user-item matrix from ratings"""
    # Filter valid ratings
    rating_df = rating_df[rating_df['rating'] != -1]
    
    # Sample if too large
    if len(rating_df) > 100000:
        rating_df = rating_df.sample(100000, random_state=42)
    
    # Create pivot table
    user_item_matrix = rating_df.pivot_table(
        index='user_id',
        columns='anime_id',
        values='rating',
        fill_value=0
    )
    
    # Limit size
    user_item_matrix = user_item_matrix.iloc[:max_users, :max_items]
    
    return user_item_matrix

def get_anime_info(anime_id, anime_df):
    """Get anime information"""
    anime = anime_df[anime_df['anime_id'] == anime_id]
    if len(anime) == 0:
        return None
    return anime.iloc[0].to_dict()