import pandas as pd

# Load test data
test_df = pd.read_csv('data/processed/test_data.csv')

print(f"Total ratings: {len(test_df)}")
print(f"Unique users: {test_df['user_id'].nunique()}")
print(f"High ratings (>=7): {len(test_df[test_df['rating'] >= 7])}")
print("\nSample:")
print(test_df.head())

# Thêm thông tin chi tiết
print("\n" + "="*60)
print("📊 DETAILED ANALYSIS:")
print("="*60)
print(f"Rating distribution:")
print(test_df['rating'].value_counts().sort_index())

print(f"\nAnime coverage:")
print(f"Unique anime: {test_df['anime_id'].nunique()}")