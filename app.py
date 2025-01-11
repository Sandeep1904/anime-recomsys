# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
st.title("Anime Recommendation System")
st.write("### Proof of Concept using Streamlit")
st.sidebar.header("Project Features")

# File uploader for dataset
uploaded_file = st.sidebar.file_uploader("Upload your anime dataset", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display dataset overview
    st.write("### Dataset Overview")
    st.write(df.head(5))

    # Data exploration using pandas
    st.write("### Data Exploration")
    st.write(f"Dataset Shape: {df.shape}")
    st.write(f"Columns: {', '.join(df.columns)}")
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Filter for relevant columns
    required_columns = ["Name", "Genres", "Rating", "Members"]
    if not all(col in df.columns for col in required_columns):
        st.write("Error: Dataset must contain the columns: 'name', 'genre', 'rating', 'members'.")
    else:
        df = df[required_columns]
        df.dropna(inplace=True)

        # Visualizations
        st.write("### Data Visualizations")
        st.write("#### Top 10 Most Popular Anime by Members")
        popular_anime = df.sort_values(by="Members", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=popular_anime, x="Members", y="Name", palette="viridis", ax=ax)
        ax.set_title("Top 10 Anime by Popularity (Members)")
        ax.set_xlabel("Number of Members")
        ax.set_ylabel("Anime Name")
        st.pyplot(fig)

        st.write("#### Distribution of Ratings")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Rating"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_title("Distribution of Anime Ratings")
        ax.set_xlabel("Rating")
        st.pyplot(fig)

        # Recommendation System: Content-Based Filtering
        st.write("### Recommendation System")
        st.write("#### Content-Based Recommendation Using Genres")

        # Vectorize the genre column using TF-IDF
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["Genres"])

        # Compute cosine similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Helper function to recommend anime
        def recommend_anime(title, cosine_sim=cosine_sim, df=df):
            # Get the index of the anime that matches the title
            idx = df[df["Name"] == title].index[0]

            # Get the pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the anime based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the indices of the 10 most similar anime
            sim_indices = [i[0] for i in sim_scores[1:11]]

            # Return the top 10 most similar anime
            return df.iloc[sim_indices]

        # Input for anime recommendation
        anime_title = st.selectbox("Select an Anime for Recommendations", df["Name"].unique())

        if st.button("Get Recommendations"):
            recommendations = recommend_anime(anime_title)
            st.write("#### Recommended Anime:")
            st.write(recommendations[["Name", "Genres", "Rating", "Members"]])