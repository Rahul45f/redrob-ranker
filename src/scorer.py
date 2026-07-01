"""
scorer.py
Combines extracted features into a final composite score (0.0 – 1.0).

Score = career_fitness  (0.40)
      + skills_fitness  (0.28)
      + availability    (0.18)
      + experience      (0.08)
      + education       (0.04)
      + location        (0.02)

Honeypot candidates get score = 0.0.

Design rationale (per JD):
- Career fitness is the decisive signal against keyword stuffers
  ("all the AI keywords but title is Marketing Manager = not a fit")
- Availability is high-weight because "perfect on paper but not reachable = not hireable"
- Education is low-weight because JD says it's not the primary filter
"""


def compute_score(feat: dict) -> float:
    """
    Compute final composite score for a candidate.
    Returns float in [0.0, 1.0].
    """
    # ── Hard disqualifiers ────────────────────────────────────────────────────
    if feat.get("is_honeypot"):
        return 0.0

    # ── Career Fitness (0.40 weight) ──────────────────────────────────────────
    ml_months    = feat.get("ml_ai_months", 0)
    data_months  = feat.get("data_adjacent_months", 0)
    product_ml   = feat.get("product_ml_experience", False)
    consulting   = feat.get("entire_career_consulting", False)
    cur_class    = feat.get("current_title_class", "irrelevant")
    total_months = max(feat.get("total_career_months", 1), 1)

    ml_fraction   = min(ml_months / total_months, 1.0)
    data_fraction = min(data_months / total_months, 1.0)

    career_base = ml_fraction * 1.0 + data_fraction * 0.4

    # Current title bonus
    if cur_class == "ml_ai":
        career_base += 0.30
    elif cur_class == "data_adjacent":
        career_base += 0.10

    # Product company bonus
    if product_ml:
        career_base += 0.25

    # Consulting penalty (JD explicitly penalises entire-career-consulting)
    if consulting:
        career_base *= 0.35

    career_score = min(career_base, 1.0)

    # ── Skills Fitness (0.28 weight) ──────────────────────────────────────────
    core    = feat.get("core_skill_score", 0.0)     # 0-5 scale
    nice    = feat.get("nice_skill_score", 0.0)     # 0-3 scale
    support = feat.get("support_skill_score", 0.0)  # 0-2 scale
    python  = 0.10 if feat.get("has_python") else 0.0  # Python is required

    # Normalise: core saturates at 3, nice at 2, support at 1
    skills_raw = core / 3.0 * 0.55 + nice / 2.0 * 0.25 + support * 0.10 + python
    skills_score = min(skills_raw, 1.0)

    # Penalise keyword stuffers: many skills listed but zero career ML evidence
    if ml_months == 0 and data_months == 0 and cur_class == "irrelevant":
        skills_score *= 0.20   # strong down-weight — these are likely stuffers

    # ── Availability (0.18 weight) ────────────────────────────────────────────
    availability_score = feat.get("availability_score", 0.5)

    # GitHub is a nice bonus for this specific role (engineering work evidence)
    github = feat.get("github_score", 0.3)
    availability_score = availability_score * 0.85 + github * 0.15

    # ── Experience (0.08 weight) ──────────────────────────────────────────────
    experience_score = feat.get("experience_score", 0.0)

    # ── Education (0.04 weight) ───────────────────────────────────────────────
    education_score = feat.get("education_score", 0.5)

    # ── Location (0.02 weight) ────────────────────────────────────────────────
    location_score = feat.get("location_score", 0.5)

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = (
        0.40 * career_score       +
        0.28 * skills_score       +
        0.18 * availability_score +
        0.08 * experience_score   +
        0.04 * education_score    +
        0.02 * location_score
    )

    # Keep a thin floor for valid candidates so scores stay differentiable
    return max(0.001, min(composite, 1.0))
