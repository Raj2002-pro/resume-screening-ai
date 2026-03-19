import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -------- CLEAN FUNCTION --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    df = pd.read_json("data/resumes_dataset.jsonl", lines=True)
    df = df[['Text', 'Category', 'Skills']]
    df = df.dropna()

    df['combined'] = df['Text'] + " " + df['Skills']
    df['cleaned'] = df['combined'].apply(clean_text)

    return df

df = load_data()

# -------- TRAIN MODEL --------
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['Category']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, tfidf

model, tfidf = train_model(df)

# -------- UI DESIGN --------
st.set_page_config(page_title="Resume Screening AI", layout="centered")

st.markdown("<h1 style='text-align: center;'>📄 Resume Screening AI</h1>", unsafe_allow_html=True)
st.write("This AI model predicts job role and matches resume with job description using NLP techniques.")
st.markdown("<p style='text-align: center;'>Analyze resumes using AI and match with job descriptions</p>", unsafe_allow_html=True)

st.divider()

resume_input = st.text_area("📌 Paste Resume", height=200)
job_desc_input = st.text_area("📌 Paste Job Description", height=200)

if st.button("🚀 Analyze"):

    if resume_input.strip() == "":
        st.error("⚠️ Please enter a resume")
    else:
        cleaned_resume = clean_text(resume_input)

        vector = tfidf.transform([cleaned_resume])
        prediction = model.predict(vector)

        st.divider()
        st.subheader("🎯 Predicted Job Role")
        st.success(prediction[0])

        if job_desc_input.strip() != "":
            cleaned_job = clean_text(job_desc_input)

            vectors = tfidf.transform([cleaned_resume, cleaned_job])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])
            score = similarity[0][0] * 100

            st.subheader("📊 Match Score")
            st.progress(int(score))
            st.write(f"**{score:.2f}% match**")

            resume_words = set(cleaned_resume.split())
            job_words = set(cleaned_job.split())

            matched = resume_words.intersection(job_words)

            st.subheader("🔍 Matching Skills")
            if matched:
                st.write(", ".join(matched))
            else:
                st.write("No strong matches found")