import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
import os

# Load NLP model for similarity scoring
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF resumes
def extract_text_from_pdf(file):
    return extract_text(file)

# Function to compute similarity between job description and resume
def compute_similarity(job_desc, resume_text):
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(job_embedding, resume_embedding).item()

# Streamlit UI Configuration
st.set_page_config(page_title="AI Resume Screening System", layout="wide")
st.title("AI Resume Screening & Candidate Ranking System")

st.header("Job Description")
job_description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("Ranking Resumes")
    results = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text:
            score = compute_similarity(job_description, text)
            results.append({"Candidate": file.name, "Match Score": round(score * 100, 2)})
    
    if results:
        df = pd.DataFrame(results).sort_values(by='Match Score', ascending=False)
        st.dataframe(df.style.format({"Match Score": "{:.2f}%"}))
    else:
        st.warning("No valid text extracted from resumes.")

# Run the app using: streamlit run sak.py
