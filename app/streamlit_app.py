"""
Anime Recommendation System - Streamlit App
Author: Nguyen Viet Thang - B22DCCN815
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# ✅ THÊM: Import setup từ Google Drive
sys.path.insert(0, '.')
from setup_data_gdrive import get_anime_data, get_train_data, get_test_data

# Set page config
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .anime-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ✅ SỬA: Load data từ Google Drive
@st.cache_data(show_spinner=False)
def load_data():
    """Load anime dataset from Google Drive"""
    with st.spinner("☁️ Loading data from Google Drive..."):
        try:
            anime_df = get_anime_data(use_cache=True)
            
            if anime_df is None:
                st.error("❌ Không thể tải dữ liệu!")
                st.info("""
                ### 💡 Troubleshooting:
                
                1. Kiểm tra File IDs trong `setup_data_gdrive.py`
                2. Đảm bảo files đã share publicly trên Google Drive
                3. Chạy thử: `python setup_data_gdrive.py`
                """)
                st.stop()
            
            return anime_df
            
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            st.stop()

# ✅ THÊM: Load train data
@st.cache_data(show_spinner=False)
def load_train_data():
    """Load training data from Google Drive"""
    with st.spinner("☁️ Loading training data..."):
        try:
            return get_train_data(use_cache=True)
        except Exception as e:
            st.warning(f"⚠️ Không thể tải train data: {e}")
            return None

# Build Content-Based model
@st.cache_resource
def build_content_model(_anime_df):
    """Build content-based similarity matrix"""
    genre_cols = [col for col in _anime_df.columns if col.startswith('genre_')]
    
    if len(genre_cols) == 0:
        st.error("❌ Không tìm thấy genre features!")
        return None, []
    
    genre_matrix = _anime_df[['anime_id'] + genre_cols].set_index('anime_id')
    similarity = cosine_similarity(genre_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=genre_matrix.index,
        columns=genre_matrix.index
    )
    
    return similarity_df, genre_cols

# Build CF models
@st.cache_resource
def build_cf_models(_anime_df):
    """Build Collaborative Filtering models"""
    sys.path.insert(0, 'src')
    
    from utils import create_user_item_matrix
    from recommendation_models import UserBasedCF, ItemBasedCF
    
    try:
        # ✅ SỬA: Dùng load_train_data() thay vì đọc file
        train_df = load_train_data()
        
        if train_df is None:
            st.warning("⚠️ CF models không khả dụng (thiếu train data)")
            return None, None, None
        
        user_item_matrix = create_user_item_matrix(train_df)
        
        # User-Based CF
        ub_model = UserBasedCF(user_item_matrix)
        ub_model.fit()
        
        # Item-Based CF
        ib_model = ItemBasedCF(user_item_matrix)
        ib_model.fit()
        
        return ub_model, ib_model, user_item_matrix
        
    except Exception as e:
        st.warning(f"⚠️ CF models error: {e}")
        return None, None, None

# ✅ LOAD DATA
with st.spinner("🚀 Đang khởi động ứng dụng..."):
    anime_df = load_data()
    similarity_df, genre_cols = build_content_model(anime_df)
    ub_model, ib_model, user_item_matrix = build_cf_models(anime_df)

st.success("✅ Dữ liệu đã sẵn sàng!")


# Initialize session state
if 'my_library' not in st.session_state:
    st.session_state.my_library = []

if 'model_type' not in st.session_state:
    st.session_state.model_type = "Content-Based"

# Header
st.markdown('<p class="main-header">🎬 ANIME RECOMMENDATION SYSTEM</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray;">By Nguyen Viet Thang - B22DCCN815</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    
    st.markdown("## 🎯 Menu")
    
    # Navigation
    page = st.radio(
        "Chọn trang:",
        ["🏠 Trang chủ", "📚 Quản lý Tư phim", "🔍 Tìm kiếm", "📊 Biểu đồ phân tích"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### 🤖 Chọn mô hình gợi ý:")
    
    model_type = st.radio(
        "",
        ["Content-Based", "User-Based CF", "Item-Based CF"],
        index=["Content-Based", "User-Based CF", "Item-Based CF"].index(st.session_state.model_type),
        label_visibility="collapsed",
        help="""
        - **Content-Based**: Dựa trên thể loại anime
        - **User-Based CF**: Dựa trên người dùng tương tự
        - **Item-Based CF**: Dựa trên anime tương tự
        """
    )
    
    if model_type != st.session_state.model_type:
        st.session_state.model_type = model_type
        st.rerun()
    
    st.markdown("---")
    
    # Mood Filter
    st.markdown("### Tâm trạng:")
    
    mood_filter = st.selectbox(
        "",
        ["Bình thường", "Vui vẻ", "Hồi hộp", "Lãng mạn", "Buồn"],
        label_visibility="collapsed"
    )
    
    mood_genres = {
        "Bình thường": [],
        "Vui vẻ": ["Comedy", "Slice of Life"],
        "Hồi hộp": ["Action", "Thriller", "Mystery"],
        "Lãng mạn": ["Romance", "Drama"],
        "Buồn": ["Drama", "Psychological"]
    }

# Helper functions
def get_personalized_recommendations(library_ids, model_type="Content-Based", top_n=12):
    """Get personalized recommendations based on model type"""
    
    if model_type == "Content-Based":
        if len(library_ids) == 0:
            return anime_df.nlargest(top_n, 'Score')['anime_id'].tolist()
        
        if similarity_df is None:
            return anime_df.nlargest(top_n, 'Score')['anime_id'].tolist()
        
        scores = {}
        for anime_id in anime_df['anime_id']:
            if anime_id not in library_ids:
                similarities = []
                for lib_id in library_ids:
                    if lib_id in similarity_df.index and anime_id in similarity_df.columns:
                        similarities.append(similarity_df.loc[anime_id, lib_id])
                
                if similarities:
                    scores[anime_id] = np.mean(similarities)
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [anime_id for anime_id, _ in sorted_recs]
    
    elif model_type == "User-Based CF":
        if ub_model is None or len(library_ids) == 0:
            return anime_df.nlargest(top_n, 'Score')['anime_id'].tolist()
        
        # Simplified: return popular anime
        return anime_df.nlargest(top_n, 'Members')['anime_id'].tolist()
    
    elif model_type == "Item-Based CF":
        if ib_model is None or len(library_ids) == 0 or similarity_df is None:
            return anime_df.nlargest(top_n, 'Score')['anime_id'].tolist()
        
        # Recommend based on library items
        all_recs = set()
        for anime_id in library_ids[:5]:
            if anime_id in similarity_df.index:
                similar = similarity_df[anime_id].nlargest(top_n + 1).index.tolist()
                all_recs.update([a for a in similar if a not in library_ids])
        
        return list(all_recs)[:top_n]
    
    return []

def display_anime_card(anime_row, show_add_button=True):
    """Display anime card"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"**⭐ {anime_row['Score']:.2f}**")
            st.caption(f"👥 {int(anime_row['Members']):,}")
        
        with col2:
            st.markdown(f"### {anime_row['Name']}")
            st.caption(f"🎭 {anime_row['Genres']}")
            st.caption(f"📺 {anime_row['Type']} • Episodes: {anime_row['Episodes']}")
            
            if show_add_button:
                if st.button(f"➕ Thêm vào Tư phim", key=f"add_{anime_row['anime_id']}"):
                    if anime_row['anime_id'] not in st.session_state.my_library:
                        st.session_state.my_library.append(anime_row['anime_id'])
                        st.success(f"✅ Đã thêm '{anime_row['Name']}' vào Tư phim!")
                        st.rerun()
                    else:
                        st.warning("Anime đã có trong Tư phim!")

# Pages
if page == "🏠 Trang chủ":
    st.markdown("## 🎯 Dựa trên Tư phim của tôi")
    
    if len(st.session_state.my_library) > 0:
        st.info(f"📚 Bạn có {len(st.session_state.my_library)} anime trong Tư phim")
        
        recommendations = get_personalized_recommendations(
            st.session_state.my_library,
            model_type=st.session_state.model_type,
            top_n=12
        )
        
        # Apply mood filter
        if mood_filter != "Bình thường":
            mood_genre_list = mood_genres[mood_filter]
            recommendations = [
                anime_id for anime_id in recommendations
                if any(genre in str(anime_df[anime_df['anime_id'] == anime_id]['Genres'].iloc[0])
                       for genre in mood_genre_list)
            ][:12]
        
        if len(recommendations) > 0:
            st.markdown(f"### 🎬 Gợi ý cho bạn ({st.session_state.model_type})")
            
            for i in range(0, len(recommendations), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        anime_id = recommendations[i + j]
                        anime_info = anime_df[anime_df['anime_id'] == anime_id]
                        
                        if not anime_info.empty:
                            with col:
                                display_anime_card(anime_info.iloc[0])
        else:
            st.warning("Không tìm thấy gợi ý phù hợp với tâm trạng này!")
    
    else:
        st.warning("🎯 Thư viện trống! Hãy thêm anime vào Tư phim để nhận gợi ý.")
        
        st.markdown("### 🔥 Top Anime phổ biến")
        top_anime = anime_df.nlargest(12, 'Score')
        
        for i in range(0, len(top_anime), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(top_anime):
                    with col:
                        display_anime_card(top_anime.iloc[i + j])

elif page == "📚 Quản lý Tư phim":
    st.markdown("## 📚 Tư phim của tôi")
    
    if len(st.session_state.my_library) > 0:
        library_anime = anime_df[anime_df['anime_id'].isin(st.session_state.my_library)]
        
        st.markdown(f"**Tổng số: {len(library_anime)} anime**")
        
        for idx, anime in library_anime.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {anime['Name']}")
                    st.caption(f"🎭 {anime['Genres']} • ⭐ {anime['Score']:.2f}")
                
                with col2:
                    if st.button("❌ Xóa", key=f"remove_{anime['anime_id']}"):
                        st.session_state.my_library.remove(anime['anime_id'])
                        st.success("Đã xóa!")
                        st.rerun()
                
                st.markdown("---")
        
        if st.button("🗑️ Xóa toàn bộ"):
            st.session_state.my_library = []
            st.success("Đã xóa toàn bộ Tư phim!")
            st.rerun()
    
    else:
        st.info("Tư phim trống. Hãy thêm anime từ trang Tìm kiếm!")

elif page == "🔍 Tìm kiếm":
    st.markdown("## 🔍 Tìm kiếm Anime")
    
    search_query = st.text_input("Nhập tên anime:", placeholder="Tìm kiếm...")
    
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Điểm tối thiểu:", 0.0, 10.0, 7.0, 0.5)
    with col2:
        anime_type = st.multiselect("Loại:", anime_df['Type'].unique().tolist())
    
    # Filter
    filtered_df = anime_df.copy()
    
    if search_query:
        filtered_df = filtered_df[
            filtered_df['Name'].str.contains(search_query, case=False, na=False)
        ]
    
    filtered_df = filtered_df[filtered_df['Score'] >= min_score]
    
    if anime_type:
        filtered_df = filtered_df[filtered_df['Type'].isin(anime_type)]
    
    st.markdown(f"**Tìm thấy {len(filtered_df)} anime**")
    
    for i in range(0, min(len(filtered_df), 30), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered_df):
                with col:
                    display_anime_card(filtered_df.iloc[i + j])

elif page == "📊 Biểu đồ phân tích":
    st.markdown("## 📊 Thống kê & Phân tích")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Tổng Anime", f"{len(anime_df):,}")
    with col2:
        st.metric("⭐ Điểm TB", f"{anime_df['Score'].mean():.2f}")
    with col3:
        st.metric("🎭 Thể loại", len(genre_cols))
    with col4:
        st.metric("📚 Tư phim", len(st.session_state.my_library))
    
    st.markdown("---")
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Phân bố", "🏆 Top Anime", "🎯 Model Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(anime_df, x='Score', nbins=20, 
                             title='Phân bố Điểm')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            type_counts = anime_df['Type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title='Phân bố Loại')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            top_rated = anime_df.nlargest(10, 'Score')[['Name', 'Score']]
            fig = px.bar(top_rated, x='Score', y='Name', orientation='h',
                        title='Top 10 Anime Rating cao nhất')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_popular = anime_df.nlargest(10, 'Members')[['Name', 'Members']]
            fig = px.bar(top_popular, x='Members', y='Name', orientation='h',
                        title='Top 10 Anime phổ biến')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown(f"### 🎯 Đánh giá mô hình: {st.session_state.model_type}")
        
        try:
            import json
            
            metrics_dir = 'data/processed/metrics'
            os.makedirs(metrics_dir, exist_ok=True)
            
            model_key = st.session_state.model_type.replace(' ', '_').replace('-', '_').lower()
            metrics_file = f'{metrics_dir}/{model_key}_metrics.json'
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Precision@10", f"{metrics['precision@10']:.3f}")
                with col2:
                    st.metric("📈 Recall@10", f"{metrics['recall@10']:.3f}")
                with col3:
                    st.metric("🎯 F1-Score", f"{metrics['f1_score']:.3f}")
                with col4:
                    st.metric("👥 Users", f"{metrics['num_evaluated']:,}")
                
                st.success(f"✅ Metrics cho {st.session_state.model_type} (đã đánh giá {metrics['num_evaluated']} users)")
                
                # Comparison
                with st.expander("📊 So sánh các models"):
                    all_metrics = {}
                    model_types = ['Content-Based', 'User-Based CF', 'Item-Based CF']
                    
                    for mt in model_types:
                        mkey = mt.replace(' ', '_').replace('-', '_').lower()
                        mfile = f'{metrics_dir}/{mkey}_metrics.json'
                        
                        if os.path.exists(mfile):
                            with open(mfile, 'r') as f:
                                all_metrics[mt] = json.load(f)
                    
                    if len(all_metrics) > 1:
                        comparison_df = pd.DataFrame({
                            'Model': list(all_metrics.keys()),
                            'Precision@10': [m['precision@10'] for m in all_metrics.values()],
                            'Recall@10': [m['recall@10'] for m in all_metrics.values()],
                            'F1-Score': [m['f1_score'] for m in all_metrics.values()]
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        fig = go.Figure(data=[
                            go.Bar(name='Precision@10', x=comparison_df['Model'], y=comparison_df['Precision@10']),
                            go.Bar(name='Recall@10', x=comparison_df['Model'], y=comparison_df['Recall@10']),
                            go.Bar(name='F1-Score', x=comparison_df['Model'], y=comparison_df['F1-Score'])
                        ])
                        
                        fig.update_layout(barmode='group', title="So sánh Performance")
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning(f"⚠️ Chưa có metrics cho {st.session_state.model_type}")
                st.info("💡 Chạy lệnh: `python test_evaluation.py` để tạo metrics")
        
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: gray;">Made with ❤️ by Nguyen Viet Thang • B22DCCN815</p>',
    unsafe_allow_html=True
)