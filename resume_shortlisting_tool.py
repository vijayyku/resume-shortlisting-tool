
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import re

st.title("AI Resume Shortlisting Tool (Advanced ATS)")

# --- Extract Text ---
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# --- Dynamic Skill Extraction ---
def extract_skills(jd_text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower())

    stopwords = set([
        "the", "and", "with", "for", "you", "are", "this", "that",
        "will", "have", "has", "from", "our", "your", "job",
        "role", "work", "team", "experience", "years", "required",
        "good", "strong", "knowledge", "skills", "ability"
    ])

    filtered_words = [w for w in words if w not in stopwords]

    freq = {}
    for word in filtered_words:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Top 15 keywords as skills
    skills = [word for word, count in sorted_words[:15]]

    return skills

# --- Match Skills ---
def match_skills(jd_skills, resume_text):
    return [skill for skill in jd_skills if skill in resume_text]

# --- Missing Skills ---
def missing_skills(jd_skills, matched):
    return list(set(jd_skills) - set(matched))

# --- Skill Match % ---
def skill_match_percent(jd_skills, matched):
    if len(jd_skills) == 0:
        return 0
    return (len(matched) / len(jd_skills)) * 100

# --- Highlight Skills ---
def highlight_skills(text, skills):
    for skill in skills:
        text = re.sub(
            rf"\b({skill})\b",
            r"**\1**",
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

# --- Profile Evaluation ---
def evaluate_profile(score, skill_percent, missing):
    overall_score = (score * 100 * 0.4) + (skill_percent * 0.6)

    if overall_score >= 75 and len(missing) <= 2:
        return "✅ Strong Match", overall_score
    elif overall_score >= 50:
        return "⚠️ Moderate Match", overall_score
    else:
        return "❌ Low Match", overall_score

# --- UI ---
jd = st.text_area("Paste Job Description")
files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)

if st.button("Analyze"):
    if not jd or not files:
        st.warning("Please provide JD and upload resumes")
    else:
        jd_text = jd.lower()

        # Extract skills dynamically
        jd_skills = extract_skills(jd_text)

        st.subheader("Extracted Skills from JD")
        st.write(jd_skills if jd_skills else "No skills detected")

        jd_vec = simple_embedding(jd_text)

        results = []

        for f in files:
            resume_text = extract_text(f)

            # similarity
            res_vec = simple_embedding(resume_text)
            score = compute_similarity(jd_vec, res_vec)

            # skill analysis
            matched = match_skills(jd_skills, resume_text)
            missing = missing_skills(jd_skills, matched)
            skill_percent = skill_match_percent(jd_skills, matched)

            # evaluation
            label, overall_score = evaluate_profile(score, skill_percent, missing)

            results.append((
                f.name, score, skill_percent,
                matched, missing, label,
                overall_score, resume_text
            ))

        # sort by best candidates
        results = sorted(results, key=lambda x: x[6], reverse=True)

        st.subheader("Candidate Evaluation")

        for name, score, skill_percent, matched, missing, label, overall_score, resume_text in results:
            st.write(f"## {name}")
            st.write(f"🏁 Final Verdict: {label}")
            st.write(f"📊 Overall Score: {round(overall_score, 2)}%")
            st.write(f"🔹 Similarity Score: {round(score*100, 2)}%")
            st.write(f"✅ Skill Match: {round(skill_percent, 2)}%")

            st.write("🟢 Matched Skills:", matched if matched else "None")
            st.write("🔴 Missing Skills:", missing if missing else "None")

            highlighted = highlight_skills(resume_text[:1000], matched)

            st.write("📄 Resume Preview (Highlighted)")
            st.markdown(highlighted)

            st.write("---")

