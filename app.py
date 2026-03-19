import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Resume Screening AI", page_icon="📄", layout="centered")

st.title("📄 Resume Screening AI")
st.write("Upload or paste resume text and match it with job roles using AI 🚀")

# -------------------------------
# LOAD DATA (SAFE)
# -------------------------------
@st.cache_data
def load_data():
    # This creates a path relative to the script location, not the terminal
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", "resumes_dataset.jsonl")

    if not os.path.exists(file_path):
        st.error("❌ Dataset file not found! Please check your GitHub repo structure.")
        return None

    try:
        # Standard JSON array loading (no lines=True)
        df = pd.read_json(file_path, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# This line must be at the very edge of the left side (0 spaces)
df = load_data()

if df is None:
    st.stop()

# -------------------------------
# MODEL TRAINING
# -------------------------------
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Text'])
    y = df['Category']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return tfidf, model

tfidf, model = train_model(df)

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("📌 Enter Resume Text")

resume_text = st.text_area("Paste your resume here...", height=200)

st.subheader("📌 Enter Job Description (Optional)")
job_desc = st.text_area("Paste job description here...", height=150)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Analyze Resume"):

    if resume_text.strip() == "":
        st.warning("⚠️ Please enter resume text")
    else:
        # Predict Role
        resume_vec = tfidf.transform([resume_text])
        prediction = model.predict(resume_vec)[0]

        st.success(f"✅ Predicted Job Role: **{prediction}**")

        # Matching Score (if JD given)
        if job_desc.strip() != "":
            jd_vec = tfidf.transform([job_desc])
            score = cosine_similarity(resume_vec, jd_vec)[0][0]

            st.info(f"📊 Resume Match Score: **{round(score*100, 2)}%**")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
