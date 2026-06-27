import streamlit as st
from PyPDF2 import PdfReader
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🚀 Advanced ATS - AI Skill Matching")

# =========================
# ✅ DOMAIN SKILLS
# =========================
DOMAIN_SKILLS = [
    "c", "c++", "linux", "python", "java",
    "sap", "sap fica", "sap mm", "sap p2p",
    "sql", "api testing", "selenium", "jira"
]

# =========================
# 📄 Extract PDF
# =========================
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text.lower()

# =========================
# ✅ BUILD SKILL DB FROM JD
# =========================
def build_skill_database(jd_text):
    jd_text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', jd_text.lower())
    detected = set()

    for skill in DOMAIN_SKILLS:
        skill_lower = skill.lower()

        if skill_lower == "c":
            pattern = r'(?<!\w)c(?![\w\+])'
        elif any(ch in skill_lower for ch in ['+', '#', '.']):
            pattern = r'(?<!\w)' + re.escape(skill_lower)
        else:
            pattern = r'(?<!\w)' + re.escape(skill_lower) + r'(?!\w)'

        if re.search(pattern, jd_text):
            detected.add(skill)

    return list(detected)

# =========================
# ✅ SEMANTIC SKILL MATCH
# =========================
def semantic_skill_match(skill, resume_text):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([skill, resume_text])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except:
        return 0

# =========================
# ✅ ADVANCED MATCH SKILLS
# =========================
def match_skills_advanced(jd_db, resume_text):
    resume_text = resume_text.lower()

    def normalize(text):
        return re.sub(r'[\s\-_]', '', text)

    resume_norm = normalize(resume_text)
    resume_words = set(re.findall(r'[a-zA-Z0-9\+\#\.]+', resume_text))

    matched = set()
    partial_matches = {}

    for skill in jd_db:
        skill_lower = skill.lower()
        skill_norm = normalize(skill)

        # ✅ Exact match
        if skill_lower in resume_words:
            matched.add(skill)
            continue

        # ✅ Regex match
        pattern = r'(?<!\w)' + re.escape(skill_lower) + r'(?!\w)'
        if re.search(pattern, resume_text):
            matched.add(skill)
            continue

        # ✅ Multi-word normalized match
        if len(skill_lower) > 2 and " " in skill_lower:
            if skill_norm in resume_norm:
                matched.add(skill)
                continue

        # ✅ ✅ Semantic match (AI-like)
        score = semantic_skill_match(skill, resume_text)
        if score > 0.3:
            partial_matches[skill] = round(score, 2)

    missing = [s for s in jd_db if s not in matched and s not in partial_matches]

    total = len(jd_db)
    matched_score = len(matched) + 0.5 * len(partial_matches)

    percent = min(100, (matched_score / total) * 100) if total else 0

    return list(matched), partial_matches, missing, percent

# =========================
# 🎯 SEMANTIC JD vs RESUME
# =========================
def compute_similarity(jd, resume):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd, resume])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# =========================
# 📊 EXPERIENCE
# =========================
def extract_experience(text):
    matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(years|yrs)', text.lower())
    if matches:
        return float(max(matches, key=lambda x: float(x[0]))[0])
    return 0.0

# =========================
# 🏆 ADVANCED SCORING
# =========================
def evaluate_advanced(sim, skill_pct, jd_exp, res_exp, missing_skills):

    jd_exp = float(jd_exp) if jd_exp else 0.0
    res_exp = float(res_exp) if res_exp else 0.0

    sim_score = sim * 100
    skill_score = skill_pct

    # ✅ Experience scoring
    if jd_exp > 0:
        ratio = res_exp / jd_exp
        exp_score = min(100, 70 + ratio * 30)
    else:
        exp_score = 50

    # ✅ Penalty for missing key skills
    penalty = min(30, len(missing_skills) * 5)

    final = (0.6 * skill_score) + (0.2 * sim_score) + (0.2 * exp_score) - penalty

    return round(max(0, min(final, 100)), 2)

# =========================
# 🖥 UI
# =========================
jd = st.text_area("📌 Paste Job Description", height=200)
files = st.file_uploader("📂 Upload Resumes", accept_multiple_files=True)

if st.button("🚀 Run ATS"):
    if not jd or not files:
        st.warning("Provide JD and resumes")
    else:
        jd_text = jd.lower()

        # ✅ Build JD skill DB
        jd_skill_db = build_skill_database(jd_text)
        jd_exp = extract_experience(jd_text)

        st.subheader("🧠 Extracted JD Skills")
        st.write(jd_skill_db)

        results = []

        for f in files:
            resume_text = extract_text(f)

            # ✅ Advanced Matching
            matched, partial, missing, skill_pct = match_skills_advanced(jd_skill_db, resume_text)

            # ✅ Semantic similarity
            sim = compute_similarity(jd_text, resume_text)

            # ✅ Experience
            res_exp = extract_experience(resume_text)

            # ✅ Final scoring
            score = evaluate_advanced(sim, skill_pct, jd_exp, res_exp, missing)

            results.append({
                "Name": f.name,
                "Score": score,
                "Skill %": round(skill_pct, 2),
                "Similarity %": round(sim * 100, 2),
                "Experience": res_exp,
                "Matched Skills": ", ".join(matched),
                "Partial Matches": str(partial),
                "Missing Skills": ", ".join(missing),
            })

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        st.subheader("🏆 Candidate Ranking")
        st.dataframe(df, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Report", csv, "ATS_Report.csv")
