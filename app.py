# Import required libraries
import streamlit as st
import pandas as pd
import openai
import faiss
import json
import numpy as np


# ---- OpenAI API Setup ----
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

st.title("🎌 Your AI Anime Guide")
st.write("### Discover Your Next Favorite Anime!")



@st.cache_data
def load_csv(file):
    return pd.read_csv(file)  # Adjust the filename if necessary

@st.cache_data
def load_parquet(file):
    return pd.read_parquet(file)

inputdf = load_csv("anime.csv")
df = load_parquet("df.parquet")

url = "https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020"
st.write("## You can checkout the original dataset on [Kaggle](%s)" % url)
st.write("""# A lot preprocessing can be viewed in the Jupyter Notebook on my Github,
         But the final dataframe with the embeddings from OpenAI look like the following.""")
st.write(df.head(5))  # Show sample anime entries

# ---- FAISS Index Construction ----
@st.cache_data
def build_faiss_index(df):
    df["embedding"] = df["embedding"].apply(json.loads)
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
    dimension = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss_index(df)

# ---- User Input Section ----
user_input = st.text_area("Describe an Anime You Like (or Enter an Anime Name)", "A sci-fi adventure with deep storytelling.")

if st.button("Find Similar Anime"):
    with st.spinner("Generating embeddings..."):
        # ---- Generate Embedding for User Input ----
        response = client.embeddings.create(
            input=user_input,
            model="text-embedding-3-small"
        )
        input_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)  # Convert to correct shape

        # ---- Perform FAISS Search ----
        k = 5  # Number of recommendations
        distances, indices = index.search(input_embedding, k)

        # ---- Display Recommendations ----
        st.write("### 🌟 Based on Your Input, You Might Like:")
        results = df.iloc[indices[0]][["MAL_ID", "content"]]
        st.write(results)

        # ---- Show Search Details ----
        st.write("### 📊 Nearest Neighbors' Distances")
        st.write(distances[0])







