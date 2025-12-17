"""
Data Preprocessing Module
Author: Nguyen Viet Thang - B22DCCN815
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class AnimeDataPreprocessor:
    """Preprocessing cho anime dataset"""
    
    def __init__(self, anime_path, rating_path):
        self.anime_df = pd.read_csv(anime_path)
        self.rating_df = pd.read_csv(rating_path)
        
    def handle_missing_values(self):
        """1. Xử lý missing values"""
        print("=== Xử lý Missing Values ===")
        print(f"Missing values trước:\n{self.anime_df.isnull().sum()}")
        
        # ✅ SỬA: Convert Score sang numeric trước khi xử lý
        self.anime_df['Score'] = pd.to_numeric(self.anime_df['Score'], errors='coerce')
        
        # Fill missing Score với median
        score_median = self.anime_df['Score'].median()
        self.anime_df['Score'].fillna(score_median, inplace=True)
        
        # Fill missing Genres
        self.anime_df['Genres'].fillna('Unknown', inplace=True)
        
        # Fill missing Type
        if self.anime_df['Type'].isnull().sum() > 0:
            self.anime_df['Type'].fillna(self.anime_df['Type'].mode()[0], inplace=True)
        
        # ✅ Convert Episodes sang numeric
        self.anime_df['Episodes'] = pd.to_numeric(self.anime_df['Episodes'], errors='coerce')
        self.anime_df['Episodes'].fillna(1, inplace=True)
        
        # Drop rows với missing Name
        self.anime_df.dropna(subset=['Name'], inplace=True)
        
        print(f"Missing values sau:\n{self.anime_df.isnull().sum()}")
        return self
    
    def remove_duplicates(self):
        """2. Loại bỏ duplicates"""
        print("\n=== Loại bỏ Duplicates ===")
        before = len(self.anime_df)
        
        # Kiểm tra có cột anime_id không
        if 'anime_id' in self.anime_df.columns:
            self.anime_df.drop_duplicates(subset=['anime_id'], inplace=True)
        elif 'MAL_ID' in self.anime_df.columns:
            # Đổi tên MAL_ID thành anime_id
            self.anime_df.rename(columns={'MAL_ID': 'anime_id'}, inplace=True)
            self.anime_df.drop_duplicates(subset=['anime_id'], inplace=True)
        
        # Sort by Score và remove duplicate names
        self.anime_df.sort_values('Score', ascending=False, inplace=True)
        self.anime_df.drop_duplicates(subset=['Name'], keep='first', inplace=True)
        
        after = len(self.anime_df)
        print(f"Đã xóa {before - after} duplicates ({before} -> {after})")
        return self
    
    def handle_outliers(self):
        """3. Xử lý outliers"""
        print("\n=== Xử lý Outliers ===")
        before = len(self.anime_df)
        
        # Remove invalid scores
        self.anime_df = self.anime_df[
            (self.anime_df['Score'] >= 0) & 
            (self.anime_df['Score'] <= 10)
        ]
        
        # ✅ Convert Members sang numeric nếu cần
        self.anime_df['Members'] = pd.to_numeric(self.anime_df['Members'], errors='coerce')
        self.anime_df['Members'].fillna(0, inplace=True)
        
        # Remove anime với Members âm
        self.anime_df = self.anime_df[self.anime_df['Members'] >= 0]
        
        after = len(self.anime_df)
        print(f"Đã xóa {before - after} outliers")
        return self
    
    def normalize_data(self):
        """4. Chuẩn hóa dữ liệu"""
        print("\n=== Chuẩn hóa dữ liệu ===")
        
        scaler = StandardScaler()
        
        # Normalize Score và Members
        self.anime_df['Score_normalized'] = scaler.fit_transform(
            self.anime_df[['Score']]
        )
        self.anime_df['Members_normalized'] = scaler.fit_transform(
            self.anime_df[['Members']]
        )
        
        print("Đã chuẩn hóa: Score, Members")
        return self
    
    def vectorize_genres(self):
        """5. Vector hóa genres"""
        print("\n=== Vector hóa Genres ===")
        
        all_genres = set()
        
        for genres in self.anime_df['Genres'].dropna():
            all_genres.update([g.strip() for g in str(genres).split(',')])
        
        print(f"Tìm thấy {len(all_genres)} genres unique")
        
        for genre in all_genres:
            col_name = f'genre_{genre.lower().replace(" ", "_").replace("-", "_")}'
            self.anime_df[col_name] = self.anime_df['Genres'].apply(
                lambda x: 1 if genre in str(x) else 0
            )
        
        print(f"Đã tạo {len(all_genres)} genre columns")
        return self
    
    def preprocess_ratings(self):
        """Xử lý ratings"""
        print("\n=== Xử lý Ratings ===")
        
        # Kiểm tra tên cột
        print(f"Columns trong rating_df: {self.rating_df.columns.tolist()}")
        
        before = len(self.rating_df)
        
        # Xử lý dựa trên cột thực tế
        if 'rating' in self.rating_df.columns:
            self.rating_df = self.rating_df[self.rating_df['rating'] != -1]
        else:
            print("⚠️ Không tìm thấy cột 'rating', bỏ qua bước này")
        
        after = len(self.rating_df)
        print(f"Đã xóa {before - after} invalid ratings")
        return self
    
    def save_processed_data(self, output_dir='data/processed'):
        """Lưu dữ liệu"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        anime_output = os.path.join(output_dir, 'anime_processed.csv')
        self.anime_df.to_csv(anime_output, index=False)
        print(f"\n✅ Saved: {anime_output}")
        
        rating_output = os.path.join(output_dir, 'rating_processed.csv')
        self.rating_df.to_csv(rating_output, index=False)
        print(f"✅ Saved: {rating_output}")
        
        print(f"\n📊 Tổng kết:")
        print(f"  - Anime: {len(self.anime_df)}")
        print(f"  - Ratings: {len(self.rating_df)}")
        print(f"  - Features: {self.anime_df.shape[1]}")
        
        return self


def main():
    print("🚀 Bắt đầu tiền xử lý dữ liệu...\n")
    
    preprocessor = AnimeDataPreprocessor(
        anime_path='data/raw/anime.csv',
        rating_path='data/raw/rating_complete.csv'
    )
    
    preprocessor \
        .handle_missing_values() \
        .remove_duplicates() \
        .handle_outliers() \
        .normalize_data() \
        .vectorize_genres() \
        .preprocess_ratings() \
        .save_processed_data()
    
    print("\n✨ Hoàn thành!")


if __name__ == "__main__":
    main()