# Import required libraries
import streamlit as st
import pandas as pd
import zipfile
import io
# import openai
import faiss
import json
import numpy as np
import logging 
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- App Introduction ----
st.title("🎌 Your AI Anime Guide")
st.write("### Discover Your Next Favorite Anime Effortlessly!")
st.write(
    """
    Welcome to your personalized anime recommendation tool! Simply describe your favorite type of anime,
    and we'll find the perfect match for you. Whether you're into action, romance, or sci-fi, 
    we've got you covered with the power of **AI embeddings** and similarity search.
    """
)

# ---- Dataset Information ----
st.write("## 🔍 Dataset Information")
url = "https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020"
st.write(f"You can check out the original dataset on [Kaggle]({url}).")
st.write(
    """
    Our recommendations are powered by embeddings generated from preprocessed data. 
    The dataset includes a wide variety of anime with detailed descriptions, genres, and other metadata.
    """
)

# ---- Load Data ----
@st.cache_data(ttl=120)
def load_from_zip(file_path):
    with open(file_path, "rb") as f:
        zip_bytes = f.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zipf:
        with zipf.open("df.csv") as f:
            df = pd.read_csv(f)
            return df

df = load_from_zip("df.zip")

# Show a preview of the dataset
st.write("### 📂 Sample Dataset")
st.write(df.head())

# ---- FAISS Index Construction ----
@st.cache_data(ttl=120)
def build_faiss_index(df):
    df["embedding"] = df["embedding"].apply(json.loads)
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss_index(df)

# ---- OpenAI vs Sentence Transformers ----
st.write("## ⚔️ Embedding Model Comparison: OpenAI vs Sentence Transformers")
st.write(
    """
    Initially, we utilized **OpenAI embeddings**, which provided outstanding recommendation quality. 
    However, they were computationally expensive, and the heavy API calls made it difficult to scale for production on Streamlit. 
    As a result, we transitioned to the **Sentence Transformers model** (`all-MiniLM-L6-v2`), 
    which offers a lighter and more efficient alternative with decent performance.
    """
)

# Comparison Summary
st.write("### 🔑 Key Differences")
st.write(
    """
    | Feature                          | OpenAI Embeddings                          | Sentence Transformers                   |
    |----------------------------------|--------------------------------------------|-----------------------------------------|
    | Performance (Recommendation)     | 🌟🌟🌟🌟🌟                                   | 🌟🌟🌟🌟                                |
    | Latency                          | ⚡⚡⚡                                        | ⚡⚡⚡⚡⚡                                |
    | Production Feasibility           | 🚫 Not suitable for lightweight apps       | ✅ Ideal for lightweight apps           |
    | Cost                             | 💸💸💸                                       | 💸                                    |
    """
)

# ---- User Input Section ----
st.write("## ✏️ Describe Your Favorite Anime")
user_input = st.text_area("Describe an Anime You Like (or Enter an Anime Name)", "A sci-fi adventure with deep storytelling.")

# Load Sentence Transformer Model
st.write("### 🔄 Loading the Embedding Model...")
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
st.success("Model Loaded Successfully!")

# ---- Recommendation Section ----
if st.button("Find Similar Anime"):
    logging.debug("Button clicked!")  # Log when the button is clicked
    try:
        with st.spinner("Generating Recommendations..."):
            logging.debug("Generating embeddings for user input.")
            
            # Generate embeddings for user input
            embedding = model.encode([user_input])
            logging.debug("Embedding generated successfully.")

            # FAISS similarity search
            logging.debug("Performing FAISS search for similar anime.")
            k = 5
            distances, indices = index.search(embedding, k)
            logging.debug("FAISS search completed successfully.")

            # Display Results
            st.write("### 🌟 Based on Your Input, You Might Like:")
            results = df.iloc[indices[0]][["MAL_ID", "content"]]
            st.write(results)

            st.write("### 📊 Nearest Neighbors' Distances")
            st.write(distances[0])

    except Exception as e:
        logging.exception(f"An error occurred: {e}")  # Log the full traceback
        st.error(f"An error occurred: {e}")  # Display error in Streamlit

# ---- Conclusion ----
st.write("## 🏁 Conclusion")
st.write(
    """
    Thank you for using **Your AI Anime Guide**! 
    We hope you discovered some new anime to enjoy. 
    Whether you're a fan of OpenAI or Sentence Transformers, this tool demonstrates the power of embeddings in creating personalized recommendations.
    """
)