
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import re

st.title("AI Resume Shortlisting Tool (ATS Version)")

# --- Extract Text ---
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# --- Extract Skills ---
def extract_skills(jd_text):
    skills_list = [
        "python", "c", "cpp", "embedded", "autosar", "ecu",
        "can", "lin", "matlab", "simulink", "linux",
        "testing", "debugging", "communication"
    ]

    found_skills = []
    for skill in skills_list:
        if re.search(rf"\\b{skill}\\b", jd_text):
            found_skills.append(skill)

    return found_skills

# --- Match Skills ---
def match_skills(jd_skills, resume_text):
    return [skill for skill in jd_skills if skill in resume_text]

# --- Missing Skills ---
def missing_skills(jd_skills, matched):
    return list(set(jd_skills) - set(matched))

# --- Skill Match Percentage ---
def skill_match_percent(jd_skills, matched):
    if len(jd_skills) == 0:
        return 0
    return (len(matched) / len(jd_skills)) * 100

# --- Highlight Skills ---
def highlight_skills(text, skills):
    for skill in skills:
        text = re.sub(
            rf"\\b({skill})\\b",
            r"**\\1**",
            text,
            flags=re.IGNORECASE
        )
    return text

# --- Embedding ---
def simple_embedding(text):
    words = text.split()
    vocab = list(set(words))
    vector = [words.count(w) for w in vocab]
    return np.array(vector)

# --- Similarity ---
def compute_similarity(v1, v2):
    min_len = min(len(v1), len(v2))
    v1, v2 = v1[:min_len], v2[:min_len]

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- UI ---
jd = st.text_area("Paste Job Description")
files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)

if st.button("Analyze"):
    if not jd or not files:
        st.warning("Please provide JD and upload resumes")
    else:
        jd_text = jd.lower()

        # Extract JD skills
        jd_skills = extract_skills(jd_text)

        st.subheader("Skills Identified from JD")
        st.write(jd_skills if jd_skills else "No skills detected")

        jd_vec = simple_embedding(jd_text)

        results = []

        for f in files:
            resume_text = extract_text(f)

            # similarity
            res_vec = simple_embedding(resume_text)
            score = compute_similarity(jd_vec, res_vec)

            # skills
            matched = match_skills(jd_skills, resume_text)
            missing = missing_skills(jd_skills, matched)
            skill_percent = skill_match_percent(jd_skills, matched)

            results.append((f.name, score, matched, missing, skill_percent, resume_text))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        st.subheader("Candidate Results")

        for name, score, matched, missing, skill_percent, resume_text in results:
            st.write(f"## {name}")
            st.write(f"🔹 Overall Match Score: {round(score*100, 2)}%")
            st.write(f"✅ Skill Match: {round(skill_percent, 2)}%")

            st.write("🟢 Matched Skills:", matched if matched else "None")
            st.write("🔴 Missing Skills:", missing if missing else "None")

            # Highlight skills
            highlighted_text = highlight_skills(resume_text[:1000], matched)

            st.write("📄 Resume Preview (Highlighted Skills)")
            st.markdown(highlighted_text)

            st.write("---")

