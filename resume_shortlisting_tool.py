import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Enterprise ATS AI", layout="wide")
st.title("🏢 Enterprise AI Resume Screening System")

# =========================
# 📄 Extract PDF Text
# =========================
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text.lower()

# =========================
# 🧠 Skill DB + Synonyms
# =========================
SKILL_DB = {
    "python": ["python"],
    "machine learning": ["machine learning", "ml"],
    "data science": ["data science"],
    "sql": ["sql"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "power bi": ["power bi"],
    "tableau": ["tableau"],
    "excel": ["excel"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "react": ["react"],
    "node.js": ["node", "nodejs"]
}

# =========================
# 🔍 Extract Skills
# =========================
def extract_skills(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    skills = []

    for skill, variants in SKILL_DB.items():
        for v in variants:
            if re.search(r'\b' + re.escape(v) + r'\b', text):
                skills.append(skill)
                break

    return list(set(skills))

# =========================
# 📅 Experience Extraction
# =========================
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    if matches:
        return max([int(m[0]) for m in matches])
    return 0

# =========================
# 🧩 Resume Section Detection
# =========================
def extract_sections(text):
    sections = {}

    patterns = {
        "projects": r'projects(.+?)(education|skills|experience|$)',
        "experience": r'experience(.+?)(education|skills|projects|$)',
        "skills": r'skills(.+?)(experience|education|projects|$)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        sections[key] = match.group(1) if match else ""

    return sections

# =========================
# 🎯 Semantic Similarity
# =========================
def compute_similarity(jd, resume):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd, resume])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# =========================
# ✅ Skill Matching
# =========================
def match_skills(jd, resume):
    matched = list(set(jd) & set(resume))
    missing = list(set(jd) - set(resume))
    percent = (len(matched)/len(jd)*100) if jd else 0
    return matched, missing, percent

# =========================
# 🏆 Enterprise Scoring
# =========================
def evaluate(sim, skill_pct, exp_jd, exp_res, project_weight):
    exp_score = min(exp_res/exp_jd, 1)*100 if exp_jd else 50

    final = (
        sim * 100 * 0.35 +
        skill_pct * 0.35 +
        exp_score * 0.2 +
        project_weight * 0.1
    )

    if final >= 80:
        label = "✅ Strong Hire"
    elif final >= 60:
        label = "⚠️ Consider"
    else:
        label = "❌ Reject"

    return label, final

# =========================
# 📊 Highlight Skills
# =========================
def highlight(text, skills):
    for s in skills:
        text = re.sub(rf"\b({re.escape(s)})\b", r"**\1**", text, flags=re.IGNORECASE)
    return text

# =========================
# 🖥 UI
# =========================
jd = st.text_area("📌 Job Description", height=200)
files = st.file_uploader("📂 Upload Resumes", accept_multiple_files=True)

if st.button("🚀 Run ATS AI"):
    if not jd or not files:
        st.warning("Provide JD + resumes")
    else:
        jd_text = jd.lower()

        jd_skills = extract_skills(jd_text)
        jd_exp = extract_experience(jd_text)

        st.subheader("📊 JD Analysis")
        st.write("Skills:", jd_skills)
        st.write("Experience:", jd_exp, "years")

        results = []

        for f in files:
            text = extract_text(f)

            res_skills = extract_skills(text)
            res_exp = extract_experience(text)

            sections = extract_sections(text)
            projects_len = len(sections["projects"])

            project_score = min(projects_len / 500, 1) * 100

            sim = compute_similarity(jd_text, text)

            matched, missing, skill_pct = match_skills(jd_skills, res_skills)

            verdict, final_score = evaluate(sim, skill_pct, jd_exp, res_exp, project_score)

            results.append({
                "Name": f.name,
                "Score": round(final_score,2),
                "Verdict": verdict,
                "Similarity": round(sim*100,2),
                "Skill %": round(skill_pct,2),
                "Experience": res_exp,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing)
            })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)

        st.subheader("🏆 Candidate Ranking")
        st.dataframe(df)

        # ✅ Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", csv, "ATS_Report.csv", "text/csv")

        # ✅ Detailed View
        st.subheader("🔍 Candidate Details")
        for r in results:
            st.markdown(f"### 👤 {r['Name']}")
            st.write(f"🏁 {r['Verdict']}")
            st.write(f"📊 Score: {r['Score']}%")
            st.write(f"✅ Skills: {r['Matched Skills']}")
            st.write(f"❌ Missing: {r['Missing Skills']}")
            st.write("---")


