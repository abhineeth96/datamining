import streamlit as st
import pandas as pd
import pickle

# Load data
df = pd.read_csv('preprocessed_movies.csv')
movies_df = pd.read_csv('movies.csv')

# Rename for consistency (only if needed)
df.rename(columns={'userId': 'userid', 'movieId': 'movieid'}, inplace=True)
movies_df.rename(columns={'movieId': 'movieid'}, inplace=True)

# Load trained SVD model
with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Recommendation function
def get_top_n_recommendations(model, user_id, df, movies_df, n=5):
    rated_movies = df[df['userid'] == user_id]['movieid'].tolist()
    all_movies = df['movieid'].unique()
    unrated_movies = [movie_id for movie_id in all_movies if movie_id not in rated_movies]

    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]

    top_movie_ids = [pred.iid for pred in top_n]
    top_movies = movies_df[movies_df['movieid'].isin(top_movie_ids)].copy()
    top_movies['predicted_rating'] = [pred.est for pred in top_n]

    return top_movies[['movieid', 'title', 'genres', 'predicted_rating']]

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬🔎 Movie-lens Movie Recommender System")
st.markdown("Get personalized movie recommendations based on your past ratings.")

# User ID input
user_ids = df['userid'].unique().tolist()
user_id = st.selectbox("Select User ID", sorted(user_ids))

# Show how many movies user rated
user_ratings_count = len(df[df['userid'] == user_id])
st.markdown(f"✅ User **{user_id}** has rated **{user_ratings_count}** movies.")

# Slider for number of recommendations
n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)

# Recommendation button
if st.button("Get Recommendations"):
    recommendations = get_top_n_recommendations(model, user_id, df, movies_df, n)

    st.subheader("🎯 Top Recommended Movies:")
    st.dataframe(
        recommendations.reset_index(drop=True),
        use_container_width=True
    )

# Optional style
st.markdown(
    """
    <style>
        .stDataFrame { font-size: 16px; }
        .stSlider > div > div { color: #333; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)
