
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np

st.title("AI Resume Shortlisting Tool")

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def simple_embedding(text):
    words = text.lower().split()
    vocab = list(set(words))
    vector = [words.count(w) for w in vocab]
    return np.array(vector)

def compute_similarity(v1, v2):
    min_len = min(len(v1), len(v2))
    v1, v2 = v1[:min_len], v2[:min_len]

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

jd = st.text_area("Paste Job Description")
files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)

if st.button("Analyze"):
    if not jd or not files:
        st.warning("Please provide JD and resumes")
    else:
        jd_vec = simple_embedding(jd)
        results = []

        for f in files:
            text = extract_text(f)
            res_vec = simple_embedding(text)
            score = compute_similarity(jd_vec, res_vec)
            results.append((f.name, score))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        st.subheader("Results")
        for name, score in results:
            st.write(f"{name} - {round(score*100, 2)}% match")
