import streamlit as st

# =========================
# ✅ Skill Match
# =========================
def match_skills(jd_skills, res_skills):
    matched = list(set(jd_skills) & set(res_skills))
    missing = list(set(jd_skills) - set(res_skills))
    percent = (len(matched) / len(jd_skills) * 100) if jd_skills else 0
    return matched, missing, percent

# =========================
# 🏆 Scoring Engine
# =========================
def evaluate(sim, skill_pct, exp_jd, exp_res, project_score):
    exp_score = min(exp_res / exp_jd, 1) * 100 if exp_jd else 50

    final = (
        sim * 100 * 0.3 +
        skill_pct * 0.45 +   # ✅ skills highest priority now
        exp_score * 0.15 +
        project_score * 0.1
    )

    if final >= 80:
        return "✅ Strong Hire", final
    elif final >= 60:
        return "⚠️ Consider", final
    else:
        return "❌ Reject", final

# =========================
# ✨ Highlight
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
        st.warning("⚠️ Provide JD + resumes")
    else:
        jd_text = jd.lower()

        jd_skills = extract_skills(jd_text)
        jd_exp = extract_experience(jd_text)

        st.subheader("📊 JD Analysis")
        st.write("Skills:", jd_skills)
        st.write("Experience Required:", jd_exp, "years")

        results = []

        for f in files:
            text = extract_text(f)

            res_skills = extract_skills(text)
            res_exp = extract_experience(text)

            sections = extract_sections(text)
            project_len = len(sections["projects"])
            project_score = min(project_len / 500, 1) * 100

            sim = compute_similarity(jd_text, text)

            matched, missing, skill_pct = match_skills(jd_skills, res_skills)

            verdict, final_score = evaluate(
                sim, skill_pct, jd_exp, res_exp, project_score
            )

            results.append({
                "Name": f.name,
                "Score": round(final_score, 2),
                "Verdict": verdict,
                "Similarity": round(sim * 100, 2),
                "Skill %": round(skill_pct, 2),
                "Experience": res_exp,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing)
            })

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        st.subheader("🏆 Candidate Ranking")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Report", csv, "ATS_Report.csv", "text/csv")

        st.subheader("🔍 Candidate Details")
        for r in results:
            st.markdown(f"### 👤 {r['Name']}")
            st.write(f"🏁 {r['Verdict']}")
            st.write(f"📊 Score: {r['Score']}%")
            st.write(f"✅ Skills: {r['Matched Skills']}")
            st.write(f"❌ Missing: {r['Missing Skills']}")
            st.write("---")
from PyPDF2 import PdfReader
import pandas as pd
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
# 🧠 MERGED SKILL DATABASE
# =========================
SKILL_DB = {

    # ✅ GENERAL (OLD DB)
    "programming": [
        "python", "java", "c++", "c", "c#", "go", "ruby", "php", "rust"
    ],

    "data_science": [
        "machine learning", "deep learning", "data science", "nlp",
        "computer vision", "pandas", "numpy", "scikit-learn",
        "tensorflow", "pytorch", "xgboost"
    ],

    "data_engineering": [
        "sql", "mysql", "postgresql", "mongodb",
        "hadoop", "spark", "kafka", "etl", "data warehouse"
    ],

    "frontend": [
        "html", "css", "javascript", "react", "angular", "vue"
    ],

    "backend_general": [
        "node.js", "spring boot", "django", "flask", "fastapi",
        "rest api", "microservices"
    ],

    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes",
        "jenkins", "terraform", "ansible"
    ],

    "tools": [
        "git", "github", "linux", "bash", "jira"
    ],

    "analytics": [
        "excel", "power bi", "tableau"
    ],

    # ✅ NEW (YOUR JD-SPECIFIC ENTERPRISE DB)

    "programming_backend": [
        "java", "j2ee", "spring boot", "jdbc", "sql"
    ],

    "integration_middleware": [
        "ibm sterling b2b integrator",
        "axway b2bi",
        "mulesoft",
        "dell boomi",
        "tibco",
        "api integrations",
        "b2b integrations",
        "edi workflows"
    ],

    "edi_data_standards": [
        "edi", "x12", "edifact", "map editor",
        "itx", "ibm transformation extender"
    ],

    "api_webservices": [
        "rest api", "rest apis", "soap web services",
        "web services security", "oauth", "jwt", "ssl"
    ],

    "data_formats": [
        "json", "xml", "xsd"
    ],

    "workflow_bp": [
        "bpml",
        "business process modeling language",
        "business processes",
        "routing rules"
    ],

    "protocols": [
        "as2", "as3", "sftp", "ftps",
        "https", "cifs", "webdav"
    ],

    "os_extended": [
        "linux", "unix", "shell scripting"
    ],

    "enterprise_sap": [
        "sap integrations", "idocs",
        "mft solutions", "managed file transfer"
    ],

    "methodologies": [
        "agile", "scrum"
    ]
}

# =========================
# 🔍 Build Skill Lookup
# =========================
def build_skill_lookup():
    lookup = {}

    for _, skills in SKILL_DB.items():
        for skill in skills:
            base = skill.lower()

            lookup[base] = base
            lookup[base.replace(".", "")] = base
            lookup[base.replace(" ", "")] = base

    return lookup

SKILL_LOOKUP = build_skill_lookup()

# =========================
# 🧠 Extract Skills
# =========================
def extract_skills(text):
    text = re.sub(r'[^a-zA-Z0-9\s/]', ' ', text.lower())
    found = set()

    for term in SKILL_LOOKUP:
        if re.search(r'\b' + re.escape(term) + r'\b', text):
            found.add(SKILL_LOOKUP[term])

    return list(found)

# =========================
# 📅 Extract Experience
# =========================
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    if matches:
        return max([int(m[0]) for m in matches])
    return 0

# =========================
# 🧩 Section Extraction
# =========================
def extract_sections(text):
    sections = {}

    patterns = {
        "projects": r'projects(.+?)(education|skills|experience|$)',
        "experience": r'experience(.+?)(education|skills|projects|$)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        sections[key] = match.group(1) if match else ""

    return sections

# =========================
# 🎯 Similarity
# =========================
def compute_similarity(jd, resume):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd, resume])
    
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    return similarity
