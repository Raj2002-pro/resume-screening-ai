import streamlit as st
st.set_option('server.headless', True)
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
# -------------------------------
# LOAD DATA (SAFE & AUTO-DETECT)
# -------------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", "resumes_dataset.jsonl")

    if not os.path.exists(file_path):
        st.error(f"❌ Dataset file not found at: {file_path}")
        return None

    try:
        # Try Strategy 1: Standard JSON Array
        return pd.read_json(file_path, encoding='utf-8')
    except Exception:
        try:
            # Try Strategy 2: JSON Lines (JSONL)
            return pd.read_json(file_path, lines=True, encoding='utf-8')
        except Exception as e:
            st.error(f"❌ Critical Error loading dataset: {e}")
            return None

# Global variable - must have NO indentation (start at the very left)
df = load_data()

if df is None or df.empty:
    st.error("Dataset is empty or could not be loaded. Please check your data file.")
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
