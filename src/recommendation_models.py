"""
Recommendation Models Implementation
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
class UserBasedCF:
    """User-Based Collaborative Filtering"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity = None
    
    def fit(self):
        """Calculate user similarity matrix"""
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        return self
    
    def recommend(self, user_id, top_n=10):
        """Get top N recommendations for user"""
        if user_id not in self.user_similarity_df.index:
            return []
        
        # Get similar users
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Weighted ratings
        similar_users_ratings = self.user_item_matrix.loc[similar_users.index]
        weighted_ratings = similar_users_ratings.T.dot(similar_users)
        
        # Remove already watched
        user_watched = self.user_item_matrix.loc[user_id]
        recommendations = weighted_ratings[user_watched == 0].sort_values(ascending=False)[:top_n]
        
        return recommendations.index.tolist()


class ItemBasedCF:
    """Item-Based Collaborative Filtering"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.item_similarity = None
    
    def fit(self):
        """Calculate item similarity matrix"""
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        return self
    
    def recommend(self, user_id, top_n=10):
        """Get top N recommendations for user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        watched_anime = user_ratings[user_ratings > 0].index
        
        if len(watched_anime) == 0:
            return []
        
        scores = {}
        for anime in self.user_item_matrix.columns:
            if user_ratings[anime] == 0:
                similarities = self.item_similarity_df.loc[anime, watched_anime]
                ratings = user_ratings[watched_anime]
                if similarities.sum() > 0:
                    scores[anime] = (similarities * ratings).sum() / similarities.sum()
        
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [anime_id for anime_id, _ in recommendations]


class ContentBasedFiltering:
    """Content-Based Filtering using genres"""
    
    def __init__(self, anime_df):
        self.anime_df = anime_df
        self.genre_cols = [col for col in anime_df.columns if col.startswith('genre_')]
        self.similarity_matrix = None
    
    def fit(self):
        """Calculate content similarity"""
        genre_matrix = self.anime_df[['anime_id'] + self.genre_cols].set_index('anime_id')
        self.similarity_matrix = cosine_similarity(genre_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=genre_matrix.index,
            columns=genre_matrix.index
        )
        return self
    
    def recommend(self, anime_id, top_n=10):
        """Get similar anime"""
        if anime_id not in self.similarity_df.index:
            return []
        
        similar = self.similarity_df[anime_id].sort_values(ascending=False)[1:top_n+1]
        return similar.index.tolist()


class HybridRecommender:
    """Hybrid model combining CF methods"""
    
    def __init__(self, user_based_model, item_based_model):
        self.user_based = user_based_model
        self.item_based = item_based_model
    
    def recommend(self, user_id, top_n=10):
        """Combine recommendations from both models"""
        user_recs = self.user_based.recommend(user_id, top_n=20)
        item_recs = self.item_based.recommend(user_id, top_n=20)
        
        scores = {}
        for i, anime_id in enumerate(user_recs):
            scores[anime_id] = scores.get(anime_id, 0) + (20 - i)
        
        for i, anime_id in enumerate(item_recs):
            scores[anime_id] = scores.get(anime_id, 0) + (20 - i)
        
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [anime_id for anime_id, _ in final]
    

class EmbeddingBasedRecommender:
    """Advanced recommendation using sentence embeddings"""
    
    def __init__(self, anime_df):
        self.anime_df = anime_df
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
    
    def fit(self):
        """Generate embeddings from genres + synopsis"""
        texts = self.anime_df.apply(
            lambda x: f"{x['Genres']} {x.get('Synopsis', '')}", axis=1
        ).tolist()
        self.embeddings = self.model.encode(texts)
        return self
    
    def recommend(self, anime_id, top_n=10):
        """Get recommendations using cosine similarity on embeddings"""
        idx = self.anime_df[self.anime_df['anime_id'] == anime_id].index[0]
        similarities = cosine_similarity([self.embeddings[idx]], self.embeddings)[0]
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        return self.anime_df.iloc[similar_indices]['anime_id'].tolist()    