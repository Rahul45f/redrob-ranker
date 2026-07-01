"""
features.py
Extract all scoring-relevant features from a raw candidate dict (from candidates.jsonl).
No external dependencies beyond stdlib + datetime.
"""

from datetime import datetime, date
from typing import Dict, List, Tuple, Any

# ── Reference date (July 2026) ─────────────────────────────────────────────────
TODAY = date(2026, 7, 1)

# ── JD Relevant Skills ─────────────────────────────────────────────────────────
# Tier 1: "Things you absolutely need" (per JD)
CORE_SKILLS = {
    # Embeddings / retrieval
    "embeddings", "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "embedding", "dense retrieval", "semantic search",
    # Vector DBs / hybrid search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector search", "vector database", "hybrid search",
    "approximate nearest neighbor", "ann",
    # Python
    "python",
    # Ranking eval
    "ndcg", "mrr", "map", "mean average precision", "learning to rank",
    "ltr", "ranking", "information retrieval", "ir", "evaluation framework",
}

# Tier 2: "nice to have" (per JD)
NICE_SKILLS = {
    "fine-tuning", "fine tuning", "lora", "qlora", "peft", "llm fine-tuning",
    "xgboost", "lightgbm", "gradient boosting", "neural ranking",
    "distributed systems", "large-scale inference", "inference optimization",
    "pytorch", "tensorflow", "huggingface", "transformers", "bert",
    "nlp", "natural language processing", "rag", "retrieval augmented",
    "recommendation systems", "recommendation", "search",
    "a/b testing", "ab testing", "mlops", "mlflow",
}

# Tier 3: supportive ML/data skills
SUPPORT_SKILLS = {
    "scikit-learn", "sklearn", "pandas", "numpy", "spark", "sql",
    "docker", "kubernetes", "aws", "gcp", "azure", "git",
    "kafka", "airflow", "redis", "postgresql",
}

ALL_RELEVANT = CORE_SKILLS | NICE_SKILLS | SUPPORT_SKILLS

# ── Career Title Classification ────────────────────────────────────────────────
ML_AI_TITLE_KEYWORDS = [
    "ml engineer", "machine learning engineer", "ai engineer", "artificial intelligence engineer",
    "nlp engineer", "natural language", "search engineer", "ranking engineer",
    "recommendation", "applied scientist", "applied ml", "applied ai",
    "research scientist", "research engineer", "data scientist",
    "retrieval", "information retrieval", "senior ai", "principal ai",
    "staff ml", "staff ai", "founding engineer", "intelligence engineer",
]

DATA_ADJACENT_TITLES = [
    "data engineer", "backend engineer", "software engineer", "platform engineer",
    "full stack", "fullstack", "senior engineer", "software developer",
    "senior developer", "senior software", "devops engineer", "sre",
    "site reliability",
]

# JD explicitly disqualifies these as primary career patterns
IRRELEVANT_TITLES = [
    "operations manager", "marketing manager", "hr manager", "human resources",
    "accountant", "civil engineer", "mechanical engineer", "content writer",
    "customer support", "sales", "graphic designer", "project manager",
    "business analyst", "qa engineer", "quality assurance", "recruiter",
    "financial analyst", "frontend engineer", "ui developer", "ux designer",
    ".net developer", "java developer",
]

# JD explicitly penalises entire-career-consulting
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hexaware", "mindtree", "mphasis",
    "hcl", "hcl technologies", "l&t infotech", "l&t technology", "niit",
    "kpit", "mphasis", "persistent", "birlasoft",
}

# JD wants product/startup experience
AI_PRODUCT_INDUSTRIES = {
    "ai/ml", "software", "saas", "fintech", "edtech", "healthtech",
    "e-commerce", "food delivery", "transportation", "media tech", "adtech",
    "recruitment tech", "hr tech", "marketplace",
}


# ── Proficiency weights ────────────────────────────────────────────────────────
PROFICIENCY_WEIGHT = {
    "expert":       1.0,
    "advanced":     0.80,
    "intermediate": 0.50,
    "beginner":     0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# HONEYPOT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_honeypot(c: dict) -> Tuple[bool, str]:
    """
    Return (is_honeypot, reason).
    Checks for impossible profiles as documented in submission_spec.
    """
    profile = c.get("profile", {})
    skills = c.get("skills", [])
    career = c.get("career_history", [])

    # Flag 1: many "expert"/"advanced" skills with 0 months of use
    expert_zero = [s for s in skills
                   if s.get("proficiency") in ("expert", "advanced")
                   and s.get("duration_months", 1) == 0]
    if len(expert_zero) >= 5:
        return True, f"HP:expert+0mo x{len(expert_zero)}"

    # Flag 2: sum of career duration_months hugely exceeds claimed experience
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    claimed_years = profile.get("years_of_experience", 0)
    if claimed_years > 0 and total_career_months > claimed_years * 12 * 2.0:
        return True, f"HP:career_months({total_career_months})>>claimed({claimed_years}yr)"

    # Flag 3: job start_date before plausible date (e.g., started in 2015 at company
    # but profile says founded 2020 — we check if start > end)
    for job in career:
        start_str = job.get("start_date", "")
        end_str = job.get("end_date")
        if start_str and end_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                if start > end:
                    return True, f"HP:job_dates_impossible({start}>{end})"
            except ValueError:
                pass

    # Flag 4: claimed experience years > (TODAY - earliest possible start)
    # Someone with 15+ years but only 1 career entry of 12 months
    if len(career) <= 1 and claimed_years >= 8:
        total_duration = sum(j.get("duration_months", 0) for j in career)
        if total_duration < claimed_years * 12 * 0.4:
            return True, f"HP:sparse_career({total_duration}mo) vs claimed({claimed_years}yr)"

    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# CAREER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def classify_title(title: str) -> str:
    """Classify a job title. Returns: 'ml_ai', 'data_adjacent', 'irrelevant'."""
    t = title.lower()
    if any(kw in t for kw in ML_AI_TITLE_KEYWORDS):
        return "ml_ai"
    if any(kw in t for kw in DATA_ADJACENT_TITLES):
        return "data_adjacent"
    return "irrelevant"


def analyze_career(c: dict) -> Dict[str, Any]:
    """
    Return career fitness features:
    - ml_ai_months: months in ML/AI/NLP/Search roles
    - data_adjacent_months: months in SWE/data-eng adjacent roles
    - product_company_months: months at non-consulting product companies
    - entire_career_consulting: True if ALL companies are consulting firms
    - has_product_ml_experience: shipped ML at a product company
    - current_title_class: ml_ai / data_adjacent / irrelevant
    - career_trajectory: improving / declining / flat / unclear
    """
    career = c.get("career_history", [])
    profile = c.get("profile", {})

    ml_months = 0
    data_months = 0
    product_ml = False
    total_months = 0
    consulting_months = 0
    non_consulting_companies = 0

    for job in career:
        title_class = classify_title(job.get("title", ""))
        dur = job.get("duration_months", 0)
        company = job.get("company", "").lower().strip()
        industry = job.get("industry", "").lower().strip()
        total_months += dur

        is_consulting = any(cc in company for cc in CONSULTING_COMPANIES)
        if is_consulting:
            consulting_months += dur
        else:
            non_consulting_companies += 1

        if title_class == "ml_ai":
            ml_months += dur
            if not is_consulting:
                product_ml = True
        elif title_class == "data_adjacent":
            data_months += dur

    entire_consulting = (total_months > 0 and consulting_months / total_months > 0.90)
    current_class = classify_title(profile.get("current_title", ""))

    return {
        "ml_ai_months":              ml_months,
        "data_adjacent_months":      data_months,
        "product_ml_experience":     product_ml,
        "entire_career_consulting":  entire_consulting,
        "current_title_class":       current_class,
        "total_career_months":       total_months,
        "non_consulting_roles":      non_consulting_companies,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_skills(c: dict) -> Dict[str, Any]:
    """
    Compute trust-weighted skill scores.
    Trust = proficiency_weight * clamp(duration_months/24, 0, 1) * clamp(endorsements/10, 0, 1.5)
    This catches keyword stuffers who list skills with 0 duration and 0 endorsements.
    """
    skills = c.get("skills", [])
    sig = c.get("redrob_signals", {})
    assessment_scores = sig.get("skill_assessment_scores", {})

    core_score = 0.0
    nice_score = 0.0
    support_score = 0.0
    has_python = False

    for sk in skills:
        name_lower = sk.get("name", "").lower().strip()
        proficiency = sk.get("proficiency", "beginner")
        dur = sk.get("duration_months", 0)
        endorse = sk.get("endorsements", 0)

        prof_w = PROFICIENCY_WEIGHT.get(proficiency, 0.2)
        dur_w = min(dur / 24.0, 1.0)           # saturates at 24 months
        end_w = min(endorse / 10.0, 1.5)       # bonus for many endorsements, caps at 1.5

        # Boost if Redrob assessment score available
        assess_bonus = 0.0
        for akey, ascore in assessment_scores.items():
            if akey.lower() in name_lower or name_lower in akey.lower():
                assess_bonus = (ascore / 100.0) * 0.5   # up to 0.5 extra
                break

        trust = prof_w * (0.4 + 0.6 * dur_w) * (0.5 + 0.5 * end_w) + assess_bonus

        if name_lower == "python" or "python" in name_lower:
            has_python = True

        if name_lower in CORE_SKILLS:
            core_score += trust
        elif name_lower in NICE_SKILLS:
            nice_score += trust * 0.7
        elif name_lower in SUPPORT_SKILLS:
            support_score += trust * 0.3

    return {
        "core_skill_score":    min(core_score, 5.0),   # cap to avoid runaway
        "nice_skill_score":    min(nice_score, 3.0),
        "support_skill_score": min(support_score, 2.0),
        "has_python":          has_python,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AVAILABILITY & BEHAVIORAL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_signals(c: dict) -> Dict[str, Any]:
    """
    Extract behavioral availability signals.
    A technically strong candidate who is not reachable is not hireable.
    """
    sig = c.get("redrob_signals", {})

    # Days since last active
    try:
        last_active = datetime.strptime(sig.get("last_active_date", "2020-01-01"), "%Y-%m-%d").date()
        days_since_active = (TODAY - last_active).days
    except ValueError:
        days_since_active = 365

    # Recency score: active within 30d=1.0, 90d=0.7, 180d=0.4, 365d=0.2, older=0.0
    if days_since_active <= 30:
        recency = 1.0
    elif days_since_active <= 90:
        recency = 0.7
    elif days_since_active <= 180:
        recency = 0.4
    elif days_since_active <= 365:
        recency = 0.2
    else:
        recency = 0.0

    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.4

    response_rate = float(sig.get("recruiter_response_rate", 0.0))

    # Notice period: <=30d=1.0, <=60d=0.7, <=90d=0.4, >90d=0.2
    notice = int(sig.get("notice_period_days", 90))
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    interview_rate = float(sig.get("interview_completion_rate", 0.5))
    offer_rate = sig.get("offer_acceptance_rate", -1)
    offer_score = float(offer_rate) if offer_rate != -1 else 0.5

    github = float(sig.get("github_activity_score", -1))
    github_score = (github / 100.0) if github >= 0 else 0.3   # no GitHub = neutral

    # Composite availability
    availability = (
        0.30 * recency +
        0.20 * open_to_work +
        0.20 * response_rate +
        0.15 * notice_score +
        0.10 * interview_rate +
        0.05 * offer_score
    )

    return {
        "availability_score":   availability,
        "recency_score":        recency,
        "response_rate":        response_rate,
        "notice_days":          notice,
        "github_score":         github_score,
        "open_to_work":         sig.get("open_to_work_flag", False),
        "profile_completeness": float(sig.get("profile_completeness_score", 0)) / 100.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE & EDUCATION
# ══════════════════════════════════════════════════════════════════════════════

def analyze_experience(c: dict) -> Dict[str, Any]:
    """Score years of experience against JD sweet-spot (5-9 yr, ideal 6-8)."""
    years = float(c.get("profile", {}).get("years_of_experience", 0))

    if 6 <= years <= 8:
        exp_score = 1.0
    elif 5 <= years < 6 or 8 < years <= 9:
        exp_score = 0.85
    elif 4 <= years < 5 or 9 < years <= 11:
        exp_score = 0.65
    elif 3 <= years < 4 or 11 < years <= 13:
        exp_score = 0.40
    else:
        exp_score = 0.15

    return {"experience_score": exp_score, "years_of_experience": years}


EDU_TIER_SCORE = {
    "tier_1":  1.0,
    "tier_2":  0.85,
    "tier_3":  0.65,
    "tier_4":  0.45,
    "unknown": 0.55,
}

DEGREE_SCORE = {
    "phd": 1.0, "ph.d": 1.0, "doctorate": 1.0,
    "m.tech": 0.9, "mtech": 0.9, "m.e.": 0.9,
    "msc": 0.85, "m.sc": 0.85, "m.s.": 0.85, "ms": 0.85,
    "mba": 0.75,
    "b.tech": 0.75, "btech": 0.75, "b.e.": 0.75, "be": 0.75,
    "bsc": 0.70, "b.sc": 0.70,
}


def analyze_education(c: dict) -> Dict[str, Any]:
    edu_list = c.get("education", [])
    if not edu_list:
        return {"education_score": 0.5, "best_tier": "unknown"}

    best_score = 0.0
    best_tier = "unknown"
    for edu in edu_list:
        tier = edu.get("tier", "unknown")
        tier_score = EDU_TIER_SCORE.get(tier, 0.55)
        degree = edu.get("degree", "").lower().strip().rstrip(".")
        deg_score = DEGREE_SCORE.get(degree, 0.70)
        combined = tier_score * 0.7 + deg_score * 0.3
        if combined > best_score:
            best_score = combined
            best_tier = tier

    return {"education_score": best_score, "best_tier": best_tier}


# ══════════════════════════════════════════════════════════════════════════════
# LOCATION
# ══════════════════════════════════════════════════════════════════════════════

PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "gurugram",
    "gurgaon", "bangalore", "bengaluru", "chennai", "kolkata",
}


def analyze_location(c: dict) -> Dict[str, Any]:
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})

    country = profile.get("country", "").lower().strip()
    location = profile.get("location", "").lower().strip()
    willing_to_relocate = sig.get("willing_to_relocate", False)

    in_india = (country == "india")
    in_preferred_city = any(city in location for city in PREFERRED_CITIES)

    if in_india and in_preferred_city:
        loc_score = 1.0
    elif in_india:
        loc_score = 0.85
    elif willing_to_relocate:
        loc_score = 0.60
    else:
        loc_score = 0.30

    return {
        "location_score":    loc_score,
        "in_india":          in_india,
        "preferred_city":    in_preferred_city,
        "willing_relocate":  willing_to_relocate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(c: dict) -> dict:
    """Full feature extraction for one candidate dict."""
    hp_flag, hp_reason = detect_honeypot(c)
    career  = analyze_career(c)
    skills  = analyze_skills(c)
    signals = analyze_signals(c)
    exp     = analyze_experience(c)
    edu     = analyze_education(c)
    loc     = analyze_location(c)

    return {
        "candidate_id": c.get("candidate_id", ""),
        "is_honeypot":  hp_flag,
        "hp_reason":    hp_reason,
        **career,
        **skills,
        **signals,
        **exp,
        **edu,
        **loc,
    }
