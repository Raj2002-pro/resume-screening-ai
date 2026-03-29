
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Resume Screening AI", layout="centered")
st.title("📄 Resume Screening AI")
st.write("Predict job role and match resume with job description using AI")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    file_path = "data/Resume_small.csv"

    if not os.path.exists(file_path):
        st.error("Dataset not found ❌")
        return None

    df = pd.read_csv(file_path)
    df = df[['Category', 'Resume_str']]

    # Rename for simplicity (optional but recommended)
    df = df.rename(columns={'Resume_str': 'Resume'})

    return df

df = load_data()


if df is None:
    st.stop()

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Resume'])
    y = df['Category']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return tfidf, model

tfidf, model = train_model(df)

# -------------------------------
# INPUT
# -------------------------------
st.subheader("📌 Enter Resume Text")
resume_text = st.text_area("Paste resume here", height=200)

st.subheader("📌 Enter Job Description")
job_desc = st.text_area("Paste job description", height=150)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Analyze Resume"):

    if resume_text.strip() == "":
        st.warning("Please enter resume text")
    else:
        resume_vec = tfidf.transform([resume_text])
        prediction = model.predict(resume_vec)[0]

        st.success(f"Predicted Role: {prediction}")

        if job_desc.strip() != "":
            jd_vec = tfidf.transform([job_desc])
            score = cosine_similarity(resume_vec, jd_vec)[0][0]

            st.info(f"Match Score: {round(score*100, 2)}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built using Streamlit & Machine Learning")
