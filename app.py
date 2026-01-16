import streamlit as st
import pickle
import numpy as np
import re
import os
from scipy.sparse import hstack

# ---------- LOAD MODELS ----------

@st.cache_data(show_spinner=False)
def load_models():
    with gzip.open("tfidf_vectorizer.pkl.gz", "rb") as f:
        vectorizer = pickle.load(f)
    with gzip.open("knn_model.pkl.gz", "rb") as f:
        knn_model = pickle.load(f)
    with gzip.open("naive_bayes_model.pkl.gz", "rb") as f:
        nb_model = pickle.load(f)
    with gzip.open("decision_tree_model.pkl.gz", "rb") as f:
        dt_model = pickle.load(f)
    return vectorizer, knn_model, nb_model, dt_model

vectorizer, knn_model, nb_model, dt_model = load_models()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "knn_model.pkl"), "rb") as f:
    knn_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "naive_bayes_model.pkl"), "rb") as f:
    nb_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "decision_tree_model.pkl"), "rb") as f:
    dt_model = pickle.load(f)


# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Fake Job Posting Predictor", layout="wide")
st.title("Fake Job Posting Predictor")
st.write("Enter job details below to predict whether a job posting is **FAKE** or **REAL**.")

# ---------- USER INPUT ----------
job_text = st.text_area("Job Description", height=200)
has_logo = st.selectbox("Has Company Logo?", [0, 1])
has_questions = st.selectbox("Has Screening Questions?", [0, 1])

# ---------- PREDICTION ----------
if st.button("Predict"):
    if not job_text.strip():
        st.warning("Please enter a job description to predict.")
    else:
        # Clean and vectorize text
        job_text_clean = clean_text(job_text)
        text_vec = vectorizer.transform([job_text_clean])

        numeric_features = np.array([[has_logo, has_questions]])
        final_input = hstack([text_vec, numeric_features])

        # Get predictions
        pred_nb = nb_model.predict(final_input)[0]
        pred_dt = dt_model.predict(final_input)[0]
        pred_knn = knn_model.predict(final_input)[0]

        # Display results in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Naive Bayes")
            if pred_nb == 1:
                st.error("FAKE ❌")
            else:
                st.success("REAL ✅")

        with col2:
            st.subheader("Decision Tree")
            if pred_dt == 1:
                st.error("FAKE ❌")
            else:
                st.success("REAL ✅")

        with col3:
            st.subheader("KNN")
            if pred_knn == 1:
                st.error("FAKE ❌")
            else:
                st.success("REAL ✅")


