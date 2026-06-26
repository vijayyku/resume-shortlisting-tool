import streamlit as st
from PyPDF2 import PdfReader
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(layout="wide")
st.title("🏢 Enterprise ATS - Domain + Semantic Matching")

# =========================
# ✅ DOMAIN SKILLS
# =========================
DOMAIN_SKILLS = [
    "sap", "sap p2p", "sap fico", "sap mm",
    "procure to pay", "accounts payable", "invoice processing",
    "vendor management", "purchase order",
    "java", "python", "rest api", "sql",
    "aws", "azure", "gcp"
]

# =========================
# ✅ DOMAIN SPECIALIZATION MAP
# =========================
DOMAIN_MAP = {
    "sap p2p": [
        "procure to pay", "procurement cycle", "purchase to pay",
        "accounts payable", "invoice processing", "vendor management",
        "vendor invoice", "purchase order", "po processing",
        "goods receipt", "grn", "3 way matching",
        "invoice verification", "sap mm", "materials management"
    ],
    "sap fico": [
        "general ledger", "accounts receivable",
        "accounts payable", "financial accounting"
    ]
}

# =========================
# 📄 Extract PDF
# =========================
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# =========================
# ✅ BUILD JD SKILLS
# =========================
def build_skill_database(jd_text):
    jd_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', jd_text.lower())

    detected = set()
    for skill in DOMAIN_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', jd_text):
            detected.add(skill)

    return list(detected)

# =========================
# ✅ DOMAIN + SEMANTIC MATCH
# =========================
def match_skills(jd_db, resume_text):

    resume_lines = resume_text.split("\n")

    matched = []
    missing = []

    # Encode resume once (optimization)
    resume_embeddings = model.encode(resume_lines)

    for jd_skill in jd_db:

        jd_skill_lower = jd_skill.lower()

        # ✅ Expand domain knowledge
        variants = [jd_skill_lower]
        if jd_skill_lower in DOMAIN_MAP:
            variants.extend(DOMAIN_MAP[jd_skill_lower])

        # ✅ Encode JD variants
        jd_embeddings = model.encode(variants)

        best_score = 0

        for jd_vec in jd_embeddings:
            sims = cosine_similarity([jd_vec], resume_embeddings)[0]
            best_score = max(best_score, max(sims))

        # ✅ Match decision
        if best_score >= 0.6:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    percent = (len(matched) / len(jd_db)) * 100 if jd_db else 0

    return matched, missing, percent

# =========================
# ✅ SEMANTIC JD vs RESUME
# =========================
def compute_similarity(jd, resume):
    embeddings = model.encode([jd, resume])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim

# =========================
# ✅ EXPERIENCE
# =========================
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    if matches:
        return max([int(m[0]) for m in matches])
    return 0

# =========================
# ✅ FINAL SCORING
# =========================
def evaluate(sim, skill_pct, jd_exp, res_exp):

    sim = max(0, min(sim, 1))
    sim_score = sim * 100

    # ✅ Boost weak similarity slightly
    if sim_score < 50:
        sim_score = 50 + (sim_score * 0.5)

    skill_pct = max(0, min(skill_pct, 100))

    # ✅ Experience scoring
    if jd_exp:
        ratio = res_exp / jd_exp
        if ratio >= 1:
            exp_score = 100 + min((ratio - 1) * 10, 10)
        else:
            exp_score = 70 + (ratio * 30)
    else:
        exp_score = 50

    final = (
        sim_score * 0.1 +
        skill_pct * 0.7 +
        exp_score * 0.2
    )

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

        # ✅ Step 1: Extract JD skills
        jd_skill_db = build_skill_database(jd_text)
        jd_exp = extract_experience(jd_text)

        st.subheader("🧠 Extracted JD Skills")
        st.write(jd_skill_db)

        results = []

        for f in files:
            resume_text = extract_text(f)

            # ✅ Step 2: Domain + semantic skill match
            matched, missing, skill_pct = match_skills(jd_skill_db, resume_text)

            # ✅ Step 3: Overall semantic similarity
            sim = compute_similarity(jd_text, resume_text)

            # ✅ Step 4: Experience
            res_exp = extract_experience(resume_text)

            # ✅ Step 5: Score
            score = evaluate(sim, skill_pct, jd_exp, res_exp)

            results.append({
                "Name": f.name,
                "Score": score,
                "Skill Match %": round(skill_pct, 2),
                "Similarity %": round(sim * 100, 2),
                "Experience": res_exp,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
            })

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        st.subheader("🏆 Ranking")
        st.dataframe(df, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download", csv, "ATS_Report.csv")
