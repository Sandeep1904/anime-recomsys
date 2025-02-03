# Import required libraries
import streamlit as st
import pandas as pd
import zipfile
import io
import openai
import faiss
import json
import numpy as np
import logging 


# ---- OpenAI API Setup ----
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

st.title("🎌 Your AI Anime Guide")
st.write("### Discover Your Next Favorite Anime!")



@st.cache_data
def load_csv(file):
    return pd.read_csv(file)  # Adjust the filename if necessary

@st.cache_data  # Cache the loaded DataFrame
def load_from_zip(file_path):
    with open(file_path, "rb") as f:  # Open in binary read mode
        zip_bytes = f.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zipf:
        with zipf.open("df.csv") as f:
            df = pd.read_csv(f)
            return df

inputdf = load_csv("anime.csv")
df = load_from_zip("df.zip")

url = "https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020"
st.write("## You can checkout the original dataset on [Kaggle](%s)" % url)
st.write("""# A lot preprocessing can be viewed in the Jupyter Notebook on my Github,
         But the final dataframe with the embeddings from OpenAI look like the following.""")
st.write(df.head(5))  # Show sample anime entries

# ---- FAISS Index Construction ----
@st.cache_resource
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



# Configure logging (do this once at the top of your app)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

st.button("testing button")

if st.button("Find Similar Anime"):
    logging.debug("Button clicked!")  # Log when the button is clicked
    try:
        with st.spinner("Generating embeddings..."):
            logging.debug("Entering embedding generation block")
            response = client.embeddings.create(
                input=user_input,
                model="text-embedding-3-small"
            )
            input_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            logging.debug("Embedding generated successfully")

            logging.debug("Entering FAISS search block")
            k = 5
            distances, indices = index.search(input_embedding, k)
            logging.debug("FAISS search completed")

            logging.debug("Entering result display block")
            st.write("### 🌟 Based on Your Input, You Might Like:")
            results = df.iloc[indices[0]][["MAL_ID", "content"]]  # This line is a prime suspect
            st.write(results)
            logging.debug("Results displayed")

            st.write("### 📊 Nearest Neighbors' Distances")
            st.write(distances[0])
            logging.debug("Distances displayed")

    except Exception as e:
        logging.exception(f"An error occurred: {e}")  # Log the full traceback
        st.error(f"An error occurred: {e}")  # Display error in Streamlit





