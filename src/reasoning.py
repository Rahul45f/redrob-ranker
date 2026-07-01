"""
reasoning.py
Generate honest, specific, non-templated reasoning for each top-100 candidate.

Rules enforced (from submission_spec Stage 4 checks):
- Reference specific facts from the candidate's actual profile
- Connect to JD requirements
- Acknowledge gaps honestly
- No hallucination (only mention things present in the profile)
- Reasoning tone consistent with rank
"""

from typing import List
from src.features import CONSULTING_COMPANIES


def _career_summary(career_history: list) -> str:
    """Return a brief career string for the most recent ML/relevant role."""
    for job in career_history:
        t = job.get("title", "")
        co = job.get("company", "")
        dur = job.get("duration_months", 0)
        if any(kw in t.lower() for kw in ["ml", "ai", "nlp", "search", "recommendation",
                                            "data scientist", "research", "engineer"]):
            return f"{t} at {co} ({dur // 12}y {dur % 12}m)"
    # Fallback: current role
    if career_history:
        j = career_history[0]
        return f"{j.get('title', '')} at {j.get('company', '')}"
    return "undisclosed"


def _skill_mention(skills: list, relevant_set: set, max_n: int = 3) -> List[str]:
    """Return up to max_n relevant skill names with proficiency >= intermediate."""
    found = []
    for sk in skills:
        name = sk.get("name", "").strip()
        prof = sk.get("proficiency", "beginner")
        dur  = sk.get("duration_months", 0)
        if name.lower() in relevant_set and prof in ("intermediate", "advanced", "expert") and dur > 0:
            found.append(name)
        if len(found) >= max_n:
            break
    return found


def generate_reasoning(c: dict, feat: dict, score: float, rank: int) -> str:
    """
    Build a 1-2 sentence reasoning string, specific to this candidate.
    Rank 1-20 : strong positive framing + key strengths
    Rank 21-60: balanced framing + notable pros and cons
    Rank 61-100: honest concerns, explain why still in top 100
    """
    from src.features import CORE_SKILLS, NICE_SKILLS

    profile  = c.get("profile", {})
    career   = c.get("career_history", [])
    skills   = c.get("skills", [])
    edu_list = c.get("education", [])
    sig      = c.get("redrob_signals", {})

    name        = profile.get("anonymized_name", "Candidate")
    title       = profile.get("current_title", "")
    yrs         = profile.get("years_of_experience", 0)
    country     = profile.get("country", "")
    location    = profile.get("location", "")
    notice      = sig.get("notice_period_days", 90)
    resp_rate   = sig.get("recruiter_response_rate", 0)
    open_work   = sig.get("open_to_work_flag", False)
    github      = sig.get("github_activity_score", -1)

    best_role   = _career_summary(career)
    core_skills = _skill_mention(skills, CORE_SKILLS, 3)
    nice_skills = _skill_mention(skills, NICE_SKILLS, 2)
    all_skills  = core_skills + nice_skills

    edu_tier = edu_list[0].get("tier", "unknown") if edu_list else "unknown"
    edu_inst = edu_list[0].get("institution", "") if edu_list else ""

    ml_months = feat.get("ml_ai_months", 0)
    product_ml = feat.get("product_ml_experience", False)
    consulting  = feat.get("entire_career_consulting", False)

    # ── Part A: core qualification statement ──────────────────────────────────
    if feat.get("is_honeypot"):
        return f"Profile flagged: {feat.get('hp_reason', 'impossible signals')}; excluded."

    if ml_months >= 36 and product_ml:
        years_ml = ml_months // 12
        part_a = f"{title} with ~{years_ml}y ML/AI experience at product companies (incl. {best_role})"
    elif ml_months > 0:
        part_a = f"{title} with {ml_months // 12}y in ML-adjacent roles ({best_role})"
    else:
        part_a = f"{title} ({yrs:.1f}y exp, {best_role})"

    # ── Part B: skills / strengths ────────────────────────────────────────────
    if all_skills:
        part_b = f"relevant skills: {', '.join(all_skills[:3])}"
    else:
        part_b = "skills overlap is indirect"

    # ── Part C: signals / concerns ────────────────────────────────────────────
    concerns = []
    strengths = []

    if open_work:
        strengths.append("open to work")
    if notice <= 30:
        strengths.append(f"{notice}d notice")
    if github >= 60:
        strengths.append(f"GitHub score {github:.0f}/100")
    if resp_rate >= 0.7:
        strengths.append(f"response rate {resp_rate:.0%}")

    if consulting:
        concerns.append("entire career in IT services")
    if notice > 90:
        concerns.append(f"{notice}d notice period")
    if resp_rate < 0.3:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")
    if feat.get("recency_score", 0) < 0.4:
        concerns.append("inactive for 180+ days")
    if not feat.get("in_india") and not feat.get("willing_relocate"):
        concerns.append(f"based in {country}, relocation unclear")
    if yrs < 4.5:
        concerns.append(f"only {yrs:.1f}y total experience")
    if yrs > 11:
        concerns.append(f"over-experienced at {yrs:.0f}y")

    # ── Compose final string ──────────────────────────────────────────────────
    if rank <= 20:
        # Lead with strengths; mention 1 concern max
        signal_str = "; ".join(strengths[:2]) if strengths else "engaged profile"
        concern_str = f"; note: {concerns[0]}" if concerns else ""
        return f"{part_a}; {part_b}; {signal_str}{concern_str}."

    elif rank <= 60:
        # Balanced: one strength, one concern
        signal_str = strengths[0] if strengths else "adequate engagement"
        concern_str = concerns[0] if concerns else "minor gaps"
        return f"{part_a}; {part_b}; {signal_str} but {concern_str}."

    else:
        # Honest: lead with why they're weak, explain why still included
        concern_str = "; ".join(concerns[:2]) if concerns else "limited JD alignment"
        return (
            f"{part_a} — ranked {rank} due to {concern_str}; "
            f"included for {part_b} but below top-tier alignment."
        )
