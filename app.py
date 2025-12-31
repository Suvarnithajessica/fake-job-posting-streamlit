import streamlit as st
import pickle
import numpy as np
import re
from scipy.sparse import hstack
import gzip

# ---------- LOAD MODELS ----------
with gzip.open("tfidf_vectorizer.pkl.gz", "rb") as f:
    vectorizer = pickle.load(f)

with gzip.open("knn_model.pkl.gz", "rb") as f:
    knn_model = pickle.load(f)

with gzip.open("naive_bayes_model.pkl.gz", "rb") as f:
    nb_model = pickle.load(f)

with gzip.open("decision_tree_model.pkl.gz", "rb") as f:
    dt_model = pickle.load(f)

# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------- STREAMLIT UI ----------
st.title("Fake Job Posting Predictor")

st.write("Enter job details below to predict whether a job posting is **FAKE** or **REAL**.")

job_text = st.text_area("Job Description", height=200)

has_logo = st.selectbox("Has Company Logo?", [0, 1])
has_questions = st.selectbox("Has Screening Questions?", [0, 1])

if st.button("Predict"):
    if job_text.strip() == "":
        st.warning("Please enter job description")
    else:
        job_text_clean = clean_text(job_text)

        text_vec = vectorizer.transform([job_text_clean])
        numeric_features = np.array([[has_logo, has_questions]])
        final_input = hstack([text_vec, numeric_features])

        pred_nb = nb_model.predict(final_input)[0]
        pred_dt = dt_model.predict(final_input)[0]
        pred_knn = knn_model.predict(final_input)[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Naive Bayes")
            st.error("FAKE ❌") if pred_nb == 1 else st.success("REAL ✅")

        with col2:
            st.subheader("Decision Tree")
            st.error("FAKE ❌") if pred_dt == 1 else st.success("REAL ✅")

        with col3:
            st.subheader("KNN")
            st.error("FAKE ❌") if pred_knn == 1 else st.success("REAL ✅")
