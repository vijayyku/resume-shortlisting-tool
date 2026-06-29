import streamlit as st
from PyPDF2 import PdfReader
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🏢 Enterprise ATS - JD Driven Skill Matching")

# =========================
# ✅ DOMAIN SKILL LIBRARY (ENTERPRISE)
# =========================
DOMAIN_SKILLS = [
    # Core Programming
    "java", "j2ee", "spring boot", "jdbc", "sql", "rest",
    "python", "c", "c++", "c/c++","c#", "javascript", "typescript",
    "go", "rust", "kotlin", "scala", "swift", "c++11", "c++14", "MISRA C++",
    "php", "ruby", "dart", "r", "matlab", "Embedded C,C++", "Embedded C", "Embedded C++", 
    "Autosar", "ASPICE", "ISO 26262", "CAN", "CAN TP", "CAPL", "CANOE", "CAN ANALYZER", "UDS", "Android",
    "iOS", "Mobile Apps", "rtos", "bsw", "rte", "com", "mcal", "classic autosar", "adaptive autosar", "davinci",

    #All RTOS
    "FreeRTOS", "VxWorks", "QNX Neutrino RTOS", "RTLinux", "Zephyr RTOS", "ThreadX (Azure RTOS)",
    "NuttX RTOS", "eCos (Embedded Configurable Operating System)", "µC/OS-II", "µC/OS-III", "TI-RTOS",
    "ChibiOS", "Integrity RTOS", "SafeRTOS", "PikeOS", "Contiki RTOS", "RIOT OS", "Mbed OS", "LiteOS (Huawei LiteOS)",
    "Apache Mynewt", "Nano-RK", "Keil RTX (RTX5)", "OSEK/VDX", "AUTOSAR OS", "RTEMS (Real-Time Executive for Multiprocessor Systems)",
   
    # Integration / Middleware
    "ibm sterling b2b integrator", "axway b2bi",
    "mulesoft", "dell boomi", "tibco",
    "b2b integrations", "edi workflows",
    
   "ibm sterling b2b integrator", "axway b2bi",
   "mulesoft", "dell boomi", "tibco", "webmethods",
   "oracle soa suite", "sap pi/po", "snaplogic",

   "b2b integrations", "edi workflows", "edi mapping",
   "x12", "edifact", "tradacoms", "xml", "json",

    "enterprise service bus", "event-driven architecture",
   "message queues", "kafka", "rabbitmq", "activemq", "ibm mq",

   "data transformation", "data mapping", "schema validation",
   "integration patterns", "api management", "api gateway",

   "microservices integration", "cloud integration",
   "hybrid integration platforms", "ipaas",
    
    # EDI Standards
    "edi", "x12", "edifact", "itx",
    "ibm transformation extender", "map editor",
    
   # Web Development
   "html", "css", "javascript", "typescript",
   "react", "angular", "vue.js", "next.js",
   "bootstrap", "tailwind css", "material ui",
   "jquery", "ajax", "dom manipulation",

   "node.js", "express.js", "spring mvc", "spring web",
   "restful apis", "graphql", "web services", "web service",
   "json", "xml",

   "frontend development", "backend development", "full stack development",
   "responsive design", "cross-browser compatibility",

   "web security", "cors", "xss", "csrf",
   "session management", "cookies",

   "webpack", "vite", "babel",
   "npm", "yarn",

   "api integration", "third-party integrations",
   "seo basics", "performance optimization",

   # Web Testing
   "selenium", "cypress", "playwright", "puppeteer",
   "manual testing", "automation testing", "functional testing",
   "regression testing", "smoke testing", "sanity testing",

   "ui testing", "frontend testing", "cross-browser testing",
   "cross-device testing",

   "api testing", "postman", "rest assured", "soap ui",
   "unit testing", "integration testing", "end-to-end testing",

   "jest", "mocha", "chai", "junit", "testng",

   "bdd", "cucumber", "gherkin", "pytest", "testng",

   "test case design", "test execution", "defect tracking",

   "performance testing", "load testing", "jmeter", "gatling",
 
    # APIs & Security
    "rest apis", "soap web services",
    "oauth", "jwt", "ssl", "web services security",
    "web services", "soap",
    
    # Data formats
    "json", "xml", "xsd",

    # Workflow
    "bpml", "business processes", "routing rules",

    # Protocols
    "as2", "as3", "sftp", "ftps", "http/https", "cifs", "webdav",

    # DevOps / Tools
    "git", "jenkins", "ci/cd",

    # Cloud  
    "aws", "azure", "gcp", "dell boomi",
    "aws ec2", "aws s3", "aws lambda", "aws rds", "aws cloudwatch",
    "aws api gateway", "aws sqs", "aws sns", "AWS AppFlow", "AWS Glue", "DynamoDB",

    "azure virtual machines", "azure blob storage", "azure functions",
    "azure sql database", "azure devops", "azure service bus",

    "gcp compute engine", "gcp cloud storage", "gcp pub/sub",
    "cloud architecture", "cloud deployment", "cloud migration",
    "cloud security", "cloud monitoring",

    "docker", "kubernetes", "helm",
    "terraform", "cloudformation", "infrastructure as code",

    "serverless computing", "microservices", "containerization",
    "ci/cd pipelines", "github actions", "gitlab ci",

    "identity and access management", "iam", "openid connect",

    # OS
    "linux", "unix", "shell scripting",
    "windows server", "macos",
    "bash", "ksh", "zsh",

   "process management", "thread management",
   "memory management", "file system management",

   "cron jobs", "systemd", "init scripts",

   "user management", "permission management",
   "ssh", "scp", "rsync",

   "log monitoring", "system monitoring",
   "top", "htop", "vmstat", "iostat",

   "network configuration", "tcp/ip", "dns", "firewall",

   "package management", "yum", "apt", "rpm",

   "system administration", "troubleshooting", "performance tuning",

    # Enterprise
    "sap integrations", "idocs", "mft solutions",

    # Methodologies
    "agile", "scrum",

    # Generic
    "software testing", "test planning", "test execution", "test automation",
    "automation frameworks", "defect tracking", "performance testing", "sdlc",
    "generative ai", "ai testing", "bias detection", "test strategy",
    "analytical skills", "problem solving", "communication", "collaboration",
     
# =========================
# 🏢 ERP CORE
# =========================
"erp", "enterprise resource planning", "sap", "oracle erp",
"microsoft dynamics", "dynamics 365", "netsuite", "odoo",
"jd edwards", "peoplesoft",

# =========================
# 🏢 ERP FUNCTIONAL (BUSINESS MODULES)
# =========================

# Finance
"sap fico", "sap fica", "sap finance", "general ledger", "accounts payable",
"accounts receivable", "fixed assets", "cost accounting",
"financial accounting", "taxation", "treasury", "SAP P2P", "P2P",

# Supply Chain
"sap mm", "sap sd", "procurement", "purchase to pay",
"order to cash", "inventory management", "warehouse management",
"logistics", "demand planning", "supply chain management",

# Manufacturing
"sap pp", "production planning", "manufacturing execution",
"mrp", "bill of materials",

# HR
"sap hcm", "successfactors", "payroll", "talent management",
"recruitment", "learning management system",

# CRM
"sap crm", "salesforce", "customer relationship management",
"lead management", "opportunity management",

# =========================
# 🏢 ERP TECHNICAL
# =========================
"sap abap", "abap", "sap hana", "s4hana", "sap bw", "sap bo",
"sap fiori", "sap ui5", "sap basis", "SAP UX", "SAP UX", "SAP UX XML",
"oracle pl/sql", "oracle fusion", "SAP UI5 framework",
"ms dynamics crm development", "x++",
"erp customization", "erp integration", "Object-Oriented Programming", "oops",
"Adobe Forms", "Workflow", "BADI","IDOC", "BAPIs", "User exits", "RAP (Restful ABAP Programming)", "RAP",
"SAP Business Technology Platform", "SAP BTP",

# =========================
# 🔗 ERP INTEGRATION & MIDDLEWARE
# =========================
"idoc", "bapi", "rfc", "odata", "soap", "rest api",
"mule soft", "boomi", "sap pi", "sap po", "cpi",
"integration suite", "api management",

# =========================
# ☁️ CLOUD ERP & MODERN ERP
# =========================
"sap s4hana", "s4 hana cloud", "oracle cloud erp",
"dynamics 365 cloud", "netsuite cloud",
"cloud migration", "erp transformation",

# =========================
# 🤖 AUTOMATION IN ERP (NEW AGE)
# =========================
"rpa", "uipath", "automation anywhere", "blue prism",
"intelligent automation", "workflow automation",
"process mining", "celonis", "tosca",

# =========================
# 📊 ERP ANALYTICS & REPORTING
# =========================
"sap bw hana", "sap analytics cloud",
"power bi", "tableau", "data visualization",
"financial reporting", "dashboarding",

# =========================
# 🔄 ERP IMPLEMENTATION LIFECYCLE
# =========================
"erp implementation", "erp rollout", "erp migration",
"greenfield implementation", "brownfield implementation",
"blueprint", "gap analysis", "business process mapping",
"fit gap analysis", "uat", "cutover", "go live support",

# =========================
# 🔐 ERP SECURITY
# =========================
"sap security", "roles and authorization",
"identity access management", "gRC", "sox compliance",

# =========================
# 📊 DATA & AI IN ERP (LATEST TREND 🚀)
# =========================
"embedded analytics", "ai in erp", "predictive analytics",
"generative ai", "copilot", "chatbots", "intelligent erp",
"machine learning in erp",

# =========================
# 🧠 SOFT / CONSULTING SKILLS
# =========================
"stakeholder management", "requirement gathering",
"functional consulting", "solution design",
"change management", "client communication",

# SAP Utilities & Revenue Management
"SAP IS-U", "SAP S/4HANA Utilities", "SAPS/4HANA", "SAP S/4 HANA Retail", "S/4HANA", "SAPUI5",
"SAP BRIM (Billing and Revenue Innovation Management)",

# Core Financials (FI-CA)
"contract accounting (fi-ca)", "accounts receivable",
"dunning", "collections management", "credit management",
"dispute management", "cash application", "payment processing",

# Billing & Invoicing
"billing", "invoicing", "rate determination",
"tariff configuration", "bill simulation",
"convergent invoicing (ci)", "invoice processing",

# project or program management skills 
"Project Planning", "Project Execution",
        "Project Lifecycle Management",
        "Scope Management",
        "Work Breakdown Structure (WBS)",
        "Milestone Tracking",
        "Deliverables Management",
        "Project Governance",
        "Change Management",
        "Issue Management",
        "Risk Management",
       "Stakeholder Management",
       "project management", 
        "Program Management",
        "Program Strategy",
        "Roadmapping",
        "Portfolio Management",
        "Benefits Realization",
        "Dependency Management",
        "Cross-Functional Coordination",
        "Business Transformation",

    "Agile Project Management",
        "Scrum",
        "Kanban",
        "SAFe (Scaled Agile Framework)",
        "Sprint Planning",
        "Backlog Grooming",
        "User Story Mapping",
        "Release Planning",
        "Daily Standups",
        "Sprint Reviews",
        "Retrospectives",
    "Budget Planning",
        "Cost Control",
        "Financial Forecasting",
        "Resource Allocation",
        "Resource Optimization",
        "Earned Value Management (EVM)",
        "ROI Analysis",
        "Vendor Management",
        "Procurement Management",
    "Risk Assessment",
        "Risk Mitigation",
        "Issue Resolution",
        "Escalation Management",
        "Quality Assurance (QA)",
        "Compliance Management",
        "Audit Readiness",
        "Process Improvement",
        "Lean",
        "Six Sigma",
    "Stakeholder Communication",
        "Executive Reporting",
        "Dashboard Reporting",
        "Team Leadership",
        "Conflict Resolution",
        "Decision Making",
        "Negotiation",
        "Influence Without Authority",
    "Microsoft Project",
        "Jira",
        "Jira Align",
        "Asana",
        "Trello",
        "Smartsheet",
        "Monday.com",
        "Microsoft Teams",
        "Slack",
        "Confluence",
        "SharePoint",
        "Power BI",
        "Tableau",
        "Advanced Excel",
    "SDLC",
        "Agile SDLC",
        "Cloud Projects (Azure, AWS)",
        "DevOps Coordination",
        "API Integration",
        "Digital Transformation",
   "PMP",
        "PRINCE2",
        "Certified Scrum Master (CSM)",
        "PMI-ACP",
        "SAFe Agilist",
        "Lean Six Sigma" 
]

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
# ✅ BUILD SKILL DB FROM JD
# =========================
def build_skill_database(jd_text):
    # ✅ Normalize text
    jd_text = jd_text.lower()
    jd_text = re.sub(r'[-_/]', ' ', jd_text)  # normalize separators like sap-ui5 → sap ui5
    jd_text = re.sub(r'[^a-zA-Z0-9\s\+#\.]', ' ', jd_text)

    detected = set()

    for skill in DOMAIN_SKILLS:
        skill_lower = skill.lower()
        skill_norm = re.sub(r'[-_/]', ' ', skill_lower).strip()

        # ✅ Special handling for "c"
        if skill_norm == "c":
            pattern = r'(?<![a-zA-Z])c(?![a-zA-Z+])'

        # ✅ Skills with special characters (C++, C#, Node.js)
        elif any(ch in skill_norm for ch in ['+', '#', '.']):
            pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill_norm)

        # ✅ Alphanumeric skills (sapui5, aws3, etc.)
        elif any(ch.isdigit() for ch in skill_norm):
            pattern = r'(?<![a-zA-Z])' + re.escape(skill_norm) + r'(?![a-zA-Z])'

        # ✅ Normal words
        else:
            pattern = r'\b' + re.escape(skill_norm) + r'\b'

        # ✅ Search in JD text
        if re.search(pattern, jd_text):
            detected.add(skill)

        # ✅ Extra fallback: handle spaced variations (sap ui5 vs sapui5)
        elif " " in skill_norm:
            compact_skill = skill_norm.replace(" ", "")
            if compact_skill in jd_text.replace(" ", ""):
                detected.add(skill)

    return list(detected)
    
# =========================
# ✅ MATCH SKILLS
# =========================
def match_skills(jd_db, resume_text):
    resume_text = resume_text.lower()

    # ✅ Normalize function
    def normalize(text):
        return re.sub(r'[\s\-_]', '', text.lower())

    resume_norm = normalize(resume_text)

    # ✅ ✅ Tokenization (CRITICAL FIX)
    resume_words = set(re.findall(r'[a-zA-Z0-9\+\#\.]+', resume_text))

    # ✅ Skill synonyms / related skills map
    SKILL_MAP = {
        "rest api": ["rest apis", "restful", "api", "apis", "integration"],
        "sap ui5": ["sapui5", "ui5"],
        "javascript": ["js", "java script"],
        "json": ["js object notation"],
        "xml": ["xml"],
        "odata": ["o data", "odata services"],
        "cds": ["cds views", "core data services"],
        "git": ["version control", "github"],
        "html5": ["html"],
        "css": ["css3"],
        "sap fiori": ["fiori"],
        "ci/cd": ["ci/cd", "ci cd pipelines"], 
        "c": ["c", "embedded c", "misra c"],
        "c++": ["c++", "c++11", "c++14", "misra c++"],
        "capl": ["canoe", "can analyzer", "capl"],
        "autosar": ["autosar", "bsw", "rte", "com", "mcal", "classic autosar", "adaptive autosar", "davinci"],
        "uds": ["uds", "capl", "python"],
        "can": ["can", "can tp"],

        # ✅ SAP P2P mapping
        "sap p2p": [
            "sap p2p", "p2p", "procure to pay", "procurement cycle",
            "purchase to pay", "accounts payable", "ap",
            "invoice processing", "vendor invoice", "vendor management",
            "purchase order", "po processing", "goods receipt",
            "grn", "3 way matching", "invoice verification",
            "sap mm", "materials management"
        ],
        "sap gateway": ["gateway"],
        "ui annotation": ["annotations", "ui annotations"]
    }

    matched = set()

    #New Code
    
matched = set()

for skill in jd_db:
    skill_lower = skill.lower()
    skill_norm = normalize(skill)

    # ✅ ✅ 1. STRICT regex match (primary fix for C, C++, etc.)
    if skill_lower == "c":
        # exact 'c' only (prevents React, CSS, etc.)
        if re.search(r'\bc\b', resume_text):
            matched.add(skill)
            continue

    elif skill_lower in ["c++", "c#"]:
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, resume_text):
            matched.add(skill)
            continue

    else:
        # general safe boundary match
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, resume_text):
            matched.add(skill)
            continue

    # ✅ ✅ 2. Token match (fallback, still safe)
    if skill_lower in resume_words:
        matched.add(skill)
        continue

    # ✅ ✅ 3. Multi-word normalized match (clean fix)
    if len(skill_lower) > 2 and " " in skill_lower and skill_norm in resume_norm:
        matched.add(skill)
        continue

    # ✅ ✅ 4. Synonym matching (fixed for boundaries)
    for syn in SKILL_MAP.get(skill_lower, []):
        syn_lower = syn.lower()
        syn_norm = normalize(syn)

        # safe regex instead of direct 'in'
        pattern = r'\b' + re.escape(syn_lower) + r'\b'

        if (
            re.search(pattern, resume_text) or
            syn_norm in resume_norm
        ):
            matched.add(skill)
            break

   #End of new code

    # ✅ Missing skills
    missing = [skill for skill in jd_db if skill not in matched]

    # ✅ Percentage
    percent = min(100, (len(matched) / len(jd_db)) * 120) if jd_db else 0

    return matched, missing, percent

# =========================
# 🎯 SEMANTIC MATCH
# =========================
def compute_similarity(jd, resume):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd, resume])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# =========================
# 📊 EXPERIENCE
# =========================
def extract_experience(text):
    # matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(years|yrs)', text)
    if matches:
        #return max([int(m[0]) for m in matches])
        #return max(matches, key=lambda x: float(x[0]))[0] if matches else "0"
        return float(max(matches, key=lambda x: float(x[0]))[0])
    #return 0.0

# =========================
# 🏆 SCORING
# =========================
def evaluate(sim, skill_pct, jd_exp, res_exp,
             missing_critical_skills=0,
             skill_weight=0.7,
             sim_weight=0.1,
             exp_weight=0.2):

    
    # ✅ Convert safely (VERY IMPORTANT FIX)
    try:
        jd_exp = float(jd_exp)
    except:
        jd_exp = 0.0

    try:
        res_exp = float(res_exp)
    except:
        res_exp = 0.0
                 
    # ✅ Normalize similarity
    sim = max(0, min(sim, 1))
    sim_score = sim * 100

    # ✅ Strong normalization: boost weak similarity
    if sim_score < 50:
        sim_score = 50 + (sim_score * 0.5)

    # ✅ Normalize skills
    skill_pct = max(0, min(skill_pct, 100))

    # ✅ Experience scoring (reward higher experience slightly)
    if jd_exp > 0:
        ratio = res_exp / jd_exp
        if ratio >= 1:
            exp_score = 100 + min((ratio - 1) * 10, 10)  # bonus up to 110
        else:
            exp_score = 70 + (ratio * 30)
    else:
        exp_score = 50

    # ✅ Final weighted score (NO penalty applied)
    final = (
        sim_score * sim_weight +
        skill_pct * skill_weight +
        exp_score * exp_weight
    )

    # ✅ Clamp result
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

        # ✅ Step 1: Build Skill DB from JD
        jd_skill_db = build_skill_database(jd_text)
        jd_exp = extract_experience(jd_text)

        st.subheader("🧠 Extracted Skill Database (from JD)")
        st.write(jd_skill_db)

        results = []
      
        for f in files:
            resume_text = extract_text(f)

            # ✅ Step 2: Match skills
            matched, missing, skill_pct = match_skills(jd_skill_db, resume_text)

            # ✅ Step 3: Semantic match
            sim = compute_similarity(jd_text, resume_text)

            # ✅ Step 4: Experience
            res_exp = extract_experience(resume_text)

            # ✅ Step 5: Final Score
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

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download", csv, "ATS_Report.csv")
