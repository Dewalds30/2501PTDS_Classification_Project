"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# ===============================================================
# Streamlit News Classifier App
# ===============================================================

# Streamlit dependencies
import streamlit as st
import joblib
import os
import re

# Data dependencies
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# ---------------------------------------------------------------
# Load the pickled model and vectorizer
# ---------------------------------------------------------------
PICKLE_PATH = os.path.join("pickled_files", "model_and_vectorizer.pkl")
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

model = data["model"]
vectorizer = data["vectorizer"]

# ---------------------------------------------------------------
# Load your processed data (if needed)
# ---------------------------------------------------------------
DATA_PATH = r"C:\Users\Dewald\Documents\GitHub\2501PTDS_Classification_Project\Data\processed\train.csv"

if os.path.exists(DATA_PATH):
    raw = pd.read_csv(DATA_PATH)
else:
    st.warning("Processed data file not found at: " + DATA_PATH)

# ---------------------------------------------------------------
# Text preprocessing function
# ---------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def process_text_pro(text: str) -> str:
    """Clean and preprocess input text."""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    # 3. Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # 4. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    # 5. Tokenize
    tokens = word_tokenize(text)
    # 6. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # 7. Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# ---------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------
def main():
    """News Classifier App with Streamlit"""

    st.title("📰 News Classifier Project")
    st.subheader("Analysing and classifying news articles")

    # Sidebar
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Information Page
    if selection == "Information":
        st.info("ℹ️ General Information")
        st.markdown("This app classifies news articles using a trained machine learning model.")

    # Prediction Page
    elif selection == "Prediction":
        st.info("🤖 Make a Prediction with the ML Model")

        # User input
        news_text = st.text_area("Enter News Text", "Type Here")

        if st.button("Classify"):
            # Preprocess text
            cleaned_text = process_text_pro(news_text)

            # Transform using the loaded vectorizer
            vect_text = vectorizer.transform([cleaned_text]).toarray()

            # Predict
            prediction = model.predict(vect_text)

            st.success(f"✅ Text Categorized as: **{prediction[0]}**")

# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()

