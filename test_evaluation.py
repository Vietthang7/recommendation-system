"""Test evaluation metrics"""
import pandas as pd
import sys
sys.path.insert(0, 'src')

from evaluation import evaluate_content_based_model
from recommendation_models import ContentBasedFiltering

# Load data
print("📂 Loading data...")
anime_df = pd.read_csv('data/processed/anime_processed.csv')
test_df = pd.read_csv('data/processed/test_data.csv')

print(f"Anime: {len(anime_df)}")
print(f"Test ratings: {len(test_df)}")

# ✅ DEBUG 1: Kiểm tra anime_id match
print("\n" + "="*60)
print("🔍 DEBUG 1: Checking anime_id compatibility")
print("="*60)
print(f"Sample anime_id in anime_df: {anime_df['anime_id'].iloc[:5].tolist()}")
print(f"Sample anime_id in test_df: {test_df['anime_id'].iloc[:5].tolist()}")

# Check overlap
anime_ids_in_model = set(anime_df['anime_id'].unique())
anime_ids_in_test = set(test_df['anime_id'].unique())
overlap = anime_ids_in_model.intersection(anime_ids_in_test)
print(f"\n✅ Anime in model: {len(anime_ids_in_model)}")
print(f"✅ Anime in test: {len(anime_ids_in_test)}")
print(f"✅ Overlap: {len(overlap)} ({len(overlap)/len(anime_ids_in_test)*100:.1f}%)")

if len(overlap) < len(anime_ids_in_test) * 0.5:
    print("⚠️ WARNING: Less than 50% overlap! Model may not work properly.")

# ✅ DEBUG 2: Kiểm tra users có đủ ratings không
print("\n" + "="*60)
print("🔍 DEBUG 2: Checking user ratings")
print("="*60)
user_counts = test_df.groupby('user_id').size()
users_with_enough = user_counts[user_counts >= 3]
print(f"Total users: {len(user_counts)}")
print(f"Users with ≥3 ratings: {len(users_with_enough)}")
print(f"Users with ≥3 high ratings (>=7):")

high_rating_users = 0
for user_id in users_with_enough.index:
    user_data = test_df[test_df['user_id'] == user_id]
    if len(user_data[user_data['rating'] >= 7]) >= 3:
        high_rating_users += 1

print(f"  → {high_rating_users} users qualify for evaluation")

# Build model
print("\n" + "="*60)
print("🔨 Building Content-Based model...")
print("="*60)
model = ContentBasedFiltering(anime_df)
model.fit()

# ✅ DEBUG 3: Test model với 1 anime cụ thể
print("\n" + "="*60)
print("🔍 DEBUG 3: Testing model.recommend()")
print("="*60)

# Lấy anime từ test set
test_anime_ids = test_df['anime_id'].unique()[:5]
for test_id in test_anime_ids:
    try:
        recs = model.recommend(test_id, top_n=5)
        print(f"✅ Anime {test_id} → Recommendations: {recs[:3]}")
        break  # Chỉ test 1 cái thành công
    except Exception as e:
        print(f"❌ Anime {test_id} → Error: {e}")

# ✅ DEBUG 4: Test evaluation với 1 user
print("\n" + "="*60)
print("🔍 DEBUG 4: Testing evaluation with 1 user")
print("="*60)

test_user = test_df[test_df.groupby('user_id')['rating'].transform('count') >= 3]['user_id'].iloc[0]
user_data = test_df[test_df['user_id'] == test_user]
liked_anime = user_data[user_data['rating'] >= 7]['anime_id'].tolist()

print(f"Test user: {test_user}")
print(f"Total ratings: {len(user_data)}")
print(f"Liked anime (>=7): {liked_anime[:5]}")

if len(liked_anime) >= 3:
    print("\n🔄 Getting recommendations for this user:")
    all_recommended = set()
    
    for seed_anime in liked_anime[:3]:
        try:
            recs = model.recommend(seed_anime, top_n=10)
            print(f"  Seed {seed_anime} → {len(recs)} recommendations")
            all_recommended.update(recs)
        except Exception as e:
            print(f"  Seed {seed_anime} → Error: {e}")
    
    print(f"\n📊 Total unique recommendations: {len(all_recommended)}")
    
    # Filter out watched
    watched = user_data['anime_id'].tolist()
    final_recs = [a for a in all_recommended if a not in watched]
    print(f"📊 After removing watched: {len(final_recs)}")
    
    # Check overlap with liked
    overlap_count = len(set(final_recs).intersection(set(liked_anime)))
    print(f"📊 Overlap with liked anime: {overlap_count} / {len(liked_anime)}")
    
    if overlap_count > 0:
        print("✅ Model CAN find relevant anime!")
    else:
        print("❌ Model CANNOT find relevant anime - this is the problem!")
else:
    print(f"⚠️ User only has {len(liked_anime)} liked anime (need ≥3)")

# Evaluate
print("\n" + "="*60)
print("📊 Running full evaluation...")
print("="*60)
metrics = evaluate_content_based_model(model, test_df, anime_df, k=10)

print("\n" + "="*60)
print("📊 FINAL RESULTS:")
print("="*60)
print(f"Precision@10: {metrics['precision@10']:.4f}")
print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Users Evaluated: {metrics['num_evaluated']}")
print("="*60)