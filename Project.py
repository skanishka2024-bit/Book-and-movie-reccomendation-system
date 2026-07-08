import streamlit as st
import pickle
import numpy as np

# ============================================================
# Load Pickle Files
# ============================================================

with open("movies_df.pkl", "rb") as f:
    movies_df = pickle.load(f)

with open("books_df.pkl", "rb") as f:
    books_df = pickle.load(f)

with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

# ============================================================
# Recreate Genre Vectors
# ============================================================

movie_vectors = mlb.transform(movies_df["genre_list"])
book_vectors = mlb.transform(books_df["genre_list"])

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

        movie_index = movies_df[movies_df["title"] == movie].index[0]
        movie_vector = movie_vectors[movie_index]

        recommendations = []

        for i in range(len(book_vectors)):

            feature = np.concatenate(
                [movie_vector, book_vectors[i]]
            )

            prediction = dt_model.predict([feature])[0]

            if prediction == 1:

                probability = dt_model.predict_proba([feature])[0][1]

                recommendations.append(
                    (
                        books_df.iloc[i]["title"],
                        probability
                    )
                )

        recommendations.sort(
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("📚 Recommended Books")

        if len(recommendations) == 0:
            st.warning("No recommendations found.")

        else:

            for title, score in recommendations[:10]:

                st.success(
                    f"{title}   |   Compatibility Score : {score:.2f}"
                )

# ============================================================
# Book -> Movie
# ============================================================

else:

    book = st.selectbox(
        "Select a Book",
        books_df["title"].tolist()
    )

    if st.button("Recommend Movies"):

        book_index = books_df[books_df["title"] == book].index[0]
        book_vector = book_vectors[book_index]

        recommendations = []

        for i in range(len(movie_vectors)):

            feature = np.concatenate(
                [movie_vectors[i], book_vector]
            )

            prediction = dt_model.predict([feature])[0]

            if prediction == 1:

                probability = dt_model.predict_proba([feature])[0][1]

                recommendations.append(
                    (
                        movies_df.iloc[i]["title"],
                        probability
                    )
                )

        recommendations.sort(
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("🎬 Recommended Movies")

        if len(recommendations) == 0:
            st.warning("No recommendations found.")

        else:

            for title, score in recommendations[:10]:

                st.success(
                    f"{title}   |   Compatibility Score : {score:.2f}"
                )