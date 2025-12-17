"""
Load data directly from Google Drive (optimized)
"""
import pandas as pd
import os

# ✅ CHỈ GIỮ 3 FILES CẦN THIẾT
FILE_IDS = {
    'anime_processed.csv': '1A5d97l86OfmD30iLdhpe7KeoTakFfnGJ',
    'train_data.csv': '1Ueyn7udL3C_BcOQwSGia-WuhGAtPclXG', 
    'test_data.csv': '14RxWVGo6j0HsvtPzRhbgbLc3TXvhpj9m'
    # ❌ BỎ: 'rating_processed.csv' (không cần thiết)
}


def load_from_gdrive(filename, use_cache=True):
    """
    Load single file from Google Drive
    ✅ Đọc trực tiếp + cache local
    """
    cache_dir = 'data/processed'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f'{cache_dir}/{filename}'
    
    # Check cache
    if use_cache and os.path.exists(cache_path):
        print(f"📂 Loading {filename} from cache...")
        return pd.read_csv(cache_path)
    
    # Load from Google Drive
    if filename not in FILE_IDS:
        print(f"❌ {filename} not found in FILE_IDS")
        return None
    
    file_id = FILE_IDS[filename]
    print(f"☁️ Loading {filename} from Google Drive...")
    
    try:
        url = f'https://drive.google.com/uc?id={file_id}&export=download'
        df = pd.read_csv(url)
        
        # Save cache
        df.to_csv(cache_path, index=False)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"💾 Cached {filename} ({size_mb:.2f} MB)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None


def get_anime_data(use_cache=True):
    """Load anime_processed.csv"""
    return load_from_gdrive('anime_processed.csv', use_cache)


def get_train_data(use_cache=True):
    """Load train_data.csv"""
    return load_from_gdrive('train_data.csv', use_cache)


def get_test_data(use_cache=True):
    """Load test_data.csv"""
    return load_from_gdrive('test_data.csv', use_cache)


def load_all_data(use_cache=True):
    """
    Load tất cả 3 files cần thiết
    """
    print("="*60)
    print("🎬 LOADING DATA FROM GOOGLE DRIVE")
    print("="*60)
    
    data = {}
    
    for filename in FILE_IDS.keys():
        df = load_from_gdrive(filename, use_cache)
        if df is not None:
            data[filename] = df
        else:
            print(f"\n❌ Failed to load {filename}!")
            return None
    
    print("\n" + "="*60)
    print("✅ ALL DATA LOADED!")
    print("="*60)
    print("\n📊 Summary:")
    for name, df in data.items():
        size_mb = len(df) * len(df.columns) * 8 / (1024 * 1024)
        print(f"  - {name}: {len(df):,} rows × {len(df.columns)} cols (~{size_mb:.1f} MB)")
    
    return data


def check_data_exists():
    """Check if required files exist locally"""
    required_files = [
        'data/processed/anime_processed.csv',
        'data/processed/train_data.csv',
        'data/processed/test_data.csv'
    ]
    return all(os.path.exists(f) for f in required_files)


def main():
    """Main function"""
    print("="*60)
    print("🎬 SETUP DATA FROM GOOGLE DRIVE")
    print("="*60)
    
    # Check cache
    if check_data_exists():
        print("\n✅ Data already cached locally!")
        print("Loading from cache...\n")
    
    # Load data
    data = load_all_data()
    
    if data:
        print("\n🚀 Ready to use!")
        print("\n📝 Usage in Python:")
        print("   from setup_data_gdrive import get_anime_data, get_train_data")
        print("   anime_df = get_anime_data()")
        print("\n📝 Run Streamlit app:")
        print("   streamlit run app/streamlit_app.py")
        return True
    else:
        print("\n❌ Setup failed!")
        print("\n💡 Troubleshooting:")
        print("1. Check File IDs in setup_data_gdrive.py")
        print("2. Verify files are shared publicly on Google Drive")
        print("3. Check internet connection")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)