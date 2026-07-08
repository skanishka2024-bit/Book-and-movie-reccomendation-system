import streamlit as st
import pickle
import numpy as np

# ============================================================
# Load Pickle Files
# ============================================================

@st.cache_resource
def load_pickles():
    with open("movies_df.pkl", "rb") as f:
        movies_df = pickle.load(f)
    with open("books_df.pkl", "rb") as f:
        books_df = pickle.load(f)
    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)
    with open("decision_tree_model.pkl", "rb") as f:
        dt_model = pickle.load(f)
    return movies_df, books_df, mlb, dt_model

movies_df, books_df, mlb, dt_model = load_pickles()

# ============================================================
# Recreate Genre Vectors
# ============================================================
# BUG FIX (perf): also cache this, so genre vectors aren't
# recomputed via mlb.transform on every rerun.
@st.cache_data
def build_vectors(_movies_df, _books_df):
    movie_vectors = mlb.transform(_movies_df["genre_list"])
    book_vectors = mlb.transform(_books_df["genre_list"])
    return movie_vectors, book_vectors

movie_vectors, book_vectors = build_vectors(movies_df, books_df)

# ============================================================
# Streamlit Page
# ============================================================
st.set_page_config(
    page_title="Book & Movie Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚🎬 Book & Movie Recommendation System")
st.write("Recommend books based on movies and movies based on books using Machine Learning.")

option = st.radio(
    "Recommendation Type",
    ["Movie ➜ Book", "Book ➜ Movie"]
)

# ============================================================
# Movie -> Book
# ============================================================
if option == "Movie ➜ Book":
    movie = st.selectbox(
        "Select a Movie",
        movies_df["title"].tolist()
    )
    if st.button("Recommend Books"):
        with st.spinner("Finding matching books..."):
            movie_index = movies_df[movies_df["title"] == movie].index[0]
            movie_vector = movie_vectors[movie_index]

            # BUG FIX (perf): the original called dt_model.predict([feature])
            # inside a Python for-loop, once per book (could be thousands of
            # calls). Each call has fixed overhead in scikit-learn. Building
            # the full feature matrix and calling predict/predict_proba ONCE
            # on the whole batch is dramatically faster.
            repeated_movie = np.repeat(
                movie_vector.reshape(1, -1), len(book_vectors), axis=0
            )
            features = np.concatenate([repeated_movie, book_vectors], axis=1)

            predictions = dt_model.predict(features)
            probabilities = dt_model.predict_proba(features)[:, 1]

            match_mask = predictions == 1
            matched_titles = books_df.loc[match_mask, "title"].tolist()
            matched_scores = probabilities[match_mask]

            recommendations = sorted(
                zip(matched_titles, matched_scores),
                key=lambda x: x[1],
                reverse=True
            )

        st.subheader("📚 Recommended Books")
        if len(recommendations) == 0:
            st.warning("No recommendations found.")
        else:
            for title, score in recommendations[:10]:
                st.success(f"{title}   |   Compatibility Score : {score:.2f}")

# ============================================================
# Book -> Movie
# ============================================================
else:
    book = st.selectbox(
        "Select a Book",
        books_df["title"].tolist()
    )
    if st.button("Recommend Movies"):
        with st.spinner("Finding matching movies..."):
            book_index = books_df[books_df["title"] == book].index[0]
            book_vector = book_vectors[book_index]

            # Same batch-prediction fix as above, mirrored for the
            # book -> movie direction.
            repeated_book = np.repeat(
                book_vector.reshape(1, -1), len(movie_vectors), axis=0
            )
            features = np.concatenate([movie_vectors, repeated_book], axis=1)

            predictions = dt_model.predict(features)
            probabilities = dt_model.predict_proba(features)[:, 1]

            match_mask = predictions == 1
            matched_titles = movies_df.loc[match_mask, "title"].tolist()
            matched_scores = probabilities[match_mask]

            recommendations = sorted(
                zip(matched_titles, matched_scores),
                key=lambda x: x[1],
                reverse=True
            )

        st.subheader("🎬 Recommended Movies")
        if len(recommendations) == 0:
            st.warning("No recommendations found.")
        else:
            for title, score in recommendations[:10]:
                st.success(f"{title}   |   Compatibility Score : {score:.2f}")
