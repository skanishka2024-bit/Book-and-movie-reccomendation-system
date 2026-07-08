import streamlit as st
import pickle
import numpy as np

# ===============================
# Load Files
# ===============================
@st.cache_resource
def load_pickles():
    with open("movies_df.pkl", "rb") as f:
        movies_df = pickle.load(f)

    with open("books_df.pkl", "rb") as f:
        books_df = pickle.load(f)

    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    return movies_df, books_df, mlb

movies_df, books_df, mlb = load_pickles()

movie_vectors = mlb.transform(movies_df["genre_list"])
book_vectors = mlb.transform(books_df["genre_list"])

st.title("📚🎬 Book & Movie Recommendation System")

option = st.radio(
    "Recommendation Type",
    ["Movie ➜ Book", "Book ➜ Movie"]
)

# ==================================================
# Movie -> Book
# ==================================================
if option == "Movie ➜ Book":

    movie = st.selectbox(
        "Select Movie",
        movies_df["title"]
    )

    if st.button("Recommend Books"):

        movie_index = movies_df[movies_df["title"] == movie].index[0]
        movie_vector = movie_vectors[movie_index]

        overlap = np.sum(book_vectors * movie_vector, axis=1)

        top = np.argsort(overlap)[::-1]

        st.subheader("📚 Recommended Books")

        count = 0

        for idx in top:

            if overlap[idx] > 0:

                st.success(
                    f"{books_df.iloc[idx]['title']} (Genre Match = {overlap[idx]})"
                )

                count += 1

            if count == 10:
                break

        if count == 0:
            st.warning("No recommendations found.")

# ==================================================
# Book -> Movie
# ==================================================
else:

    book = st.selectbox(
        "Select Book",
        books_df["title"]
    )

    if st.button("Recommend Movies"):

        book_index = books_df[books_df["title"] == book].index[0]
        book_vector = book_vectors[book_index]

        overlap = np.sum(movie_vectors * book_vector, axis=1)

        top = np.argsort(overlap)[::-1]

        st.subheader("🎬 Recommended Movies")

        count = 0

        for idx in top:

            if overlap[idx] > 0:

                st.success(
                    f"{movies_df.iloc[idx]['title']} (Genre Match = {overlap[idx]})"
                )

                count += 1

            if count == 10:
                break

        if count == 0:
            st.warning("No recommendations found.")
