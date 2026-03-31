import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# -------------------------------
# NAVBAR STYLE
# -------------------------------
st.markdown("""
<style>
body {
background-color: #0e1117;
}
.navbar {
background-color: #111;
padding: 15px;
border-radius: 10px;
}
.navbar h2 {
color: #4CAF50;
text-align: center;
}
.section {
padding: 20px;
border-radius: 10px;
background-color: #1c1f26;
margin-bottom: 20px;
}
.stButton>button {
background-color: #4CAF50;
color: white;
border-radius: 8px;
width: 100%;
height: 45px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# NAVBAR
# -------------------------------
st.markdown('<div class="navbar"><h2>📄 AI Resume Screening System</h2></div>', unsafe_allow_html=True)

# -------------------------------
# LOAD DATA (FIXED)
# -------------------------------
@st.cache_data
def load_data():
    # Get the directory where app.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "Resume_small.csv")
    
    if not os.path.exists(file_path):
        st.error("Dataset not found ❌")
        return None
    
    df = pd.read_csv(file_path)
    # ... rest of the code

	# ✅ Clean column names (important)
	df.columns = df.columns.str.strip()

	# ✅ Select correct columns
	df = df[['Category', 'Resume_str']]

	# ✅ Rename to match model usage
	df = df.rename(columns={'Resume_str': 'Resume'})

	return df

df = load_data()
if df is None:
	st.stop()

# -------------------------------
# TRAIN MODEL (FIXED)
# -------------------------------
@st.cache_resource
def train_model(df):
	tfidf = TfidfVectorizer(stop_words='english', max_features=3000)

	# ✅ Ensure text format (avoid errors)
	X = tfidf.fit_transform(df['Resume'].astype(str))
	y = df['Category']

	model = LogisticRegression(max_iter=1000)
	model.fit(X, y)

	return tfidf, model

tfidf, model = train_model(df)

# -------------------------------
# PDF READER FUNCTION
# -------------------------------
def extract_text_from_pdf(file):
	reader = PyPDF2.PdfReader(file)
	text = ""
	for page in reader.pages:
		if page.extract_text():
			text += page.extract_text()
	return text

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📥 Upload Resume or Paste Text")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
resume_text = st.text_area("Or Paste Resume Text", height=200)

job_desc = st.text_area("📌 Enter Job Description", height=150)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# BUTTON
# -------------------------------
if st.button("🚀 Analyze Resume"):

	# If PDF uploaded → extract text
	if uploaded_file is not None:
		resume_text = extract_text_from_pdf(uploaded_file)

	if resume_text.strip() == "":
		st.warning("Please upload or paste resume")
	else:
		resume_vec = tfidf.transform([resume_text])
		prediction = model.predict(resume_vec)[0]

		st.success(f"🎯 Predicted Role: {prediction}")

		# Match score
		if job_desc.strip() != "":
			jd_vec = tfidf.transform([job_desc])
			score = cosine_similarity(resume_vec, jd_vec)[0][0]

			st.progress(int(score * 100))
			st.info(f"📊 Match Score: {round(score*100, 2)}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🚀 AI-powered Resume Screening System")
