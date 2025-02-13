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
import boto3

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 configs
@st.cache_resource  # Cache the boto3 client to avoid recreating it on every interaction
def get_s3_client():
    try:
        # 1. Try environment variables first (recommended for production)
        s3 = boto3.client('s3')  # Boto3 will automatically look for env vars

    except Exception as env_err:
        try:
            # 2. If env vars are not set, try Streamlit secrets (for development)
            s3 = boto3.client('s3',
                            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                            region_name=st.secrets["AWS_DEFAULT_REGION"])

        except Exception as secret_err:
            st.error("AWS credentials not found. Please set environment variables or Streamlit secrets.")
            st.stop()  # Stop execution if credentials are not found
            return None # Return None to indicate failure

    return s3

# Get the S3 client (it will be cached)
s3 = get_s3_client()



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
@st.cache_data()
def load_from_s3():
    if s3: # Only proceed if the client was successfully created
    # Now you can use the s3 client:
        bucket_name = 'vscprojects'
        key = 'df.zip'

        try:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            zip_bytes = obj['Body'].read()
            zip_buffer = io.BytesIO(zip_bytes)  # Use BytesIO for binary data

            # Extract the DataFrame from the zip file
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                # Assuming your CSV file inside the zip is named 'df.csv' (adjust if different)
                csv_filename = 'df.csv'  # Or whatever your CSV filename is
                with zip_ref.open(csv_filename) as csv_file:  # Open the CSV file within the zip
                    df = pd.read_csv(csv_file)

            print("DataFrame loaded from S3 (df.zip):")
            return df

        except Exception as e:
            st.error(f"Error interacting with S3: {e}")
    
    else:
        print("s3 client was not loaded properly!")

df = load_from_s3()

@st.cache_data()
def load_data(file):
    return pd.read_csv(file)

dfr = load_data('anime.csv')

# Show a preview of the dataset
st.write("### 📂 Sample Dataset")
st.write(df.head())

# ---- FAISS Index Construction ----
@st.cache_data()
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
    | Performance (Recommendations)     | 🌟🌟🌟🌟🌟                                   | 🌟🌟🌟                                |
    | Latency                          | ⚡⚡⚡                                        | ⚡⚡⚡                                |
    | Production Feasibility           | 🚫 Not suitable for lightweight apps       | ✅ Better for lightweight apps           |
    | Cost                             | 💸💸💸                                       |                                     |
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

            st.write('##### A better display of results:')
            st.write(dfr.loc[dfr['MAL_ID'].isin(results['MAL_ID'])])


    except Exception as e:
        logging.exception(f"An error occurred: {e}")  # Log the full traceback
        st.error(f"An error occurred: {e}")  # Display error in Streamlit

# ---- Conclusion ----
st.write("## 🏁 Conclusion")
st.write(
    """
    Thank you for using **Your AI Anime Guide**! \n
    We hope you discovered some new anime to enjoy. \n
    Whether you're a fan of OpenAI or Sentence Transformers, this tool demonstrates the power of embeddings in creating personalized recommendations.
    """
)