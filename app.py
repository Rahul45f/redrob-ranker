"""
app.py — Redrob Ranker Streamlit Sandbox

Required by submission_spec Section 10.5.
Accepts ≤100 candidates as JSON/JSONL upload, runs the ranking system,
and produces a downloadable CSV.
"""

import json
import csv
import io
import streamlit as st
import pandas as pd

from src.features import extract_features
from src.scorer import compute_score
from src.reasoning import generate_reasoning

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker — Sandbox",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.score-high { color: #22c55e; font-weight: bold; }
.score-mid  { color: #f59e0b; font-weight: bold; }
.score-low  { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎯 Redrob Candidate Ranker")
st.markdown(
    "**INDIA.RUNS Hackathon — Intelligent Candidate Discovery**  \n"
    "Upload a JSON/JSONL candidate file (≤100 candidates) to rank them for the "
    "**Senior AI Engineer** role."
)
st.info(
    "This is the sandbox environment. The production ranker runs the same logic "
    "on 100,000 candidates via `python rank.py --candidates candidates.jsonl --out submission.csv`.",
    icon="ℹ️",
)
st.markdown("---")


# ── File upload ────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("📁 Upload Candidates")
    uploaded = st.file_uploader(
        "Upload candidates file (JSON array or JSONL — up to 500 MB)",
        type=["json", "jsonl"],
        help="Upload candidates.jsonl (full 100K pool) or sample_candidates.json for a quick test.",
    )

with col_info:
    st.subheader("⚙️ JD Target")
    st.markdown("""
    **Role:** Senior AI Engineer — Founding Team  
    **Company:** Redrob AI  
    **Location:** Pune / Noida (Hybrid)  
    **Experience:** 5–9 years  
    
    **Must-haves:**
    - Embeddings / semantic search (production)  
    - Vector DB (FAISS, Pinecone, Qdrant, etc.)  
    - Python  
    - Ranking eval (NDCG, MRR, MAP)
    """)

st.markdown("---")

if uploaded is not None:
    # ── Load candidates ────────────────────────────────────────────────────────
    raw = uploaded.read().decode("utf-8")
    candidates = []
    try:
        # Try JSON array first
        parsed = json.loads(raw)
        candidates = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        # Try JSONL
        for line in raw.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not candidates:
        st.error("Could not parse the uploaded file. Please upload a valid JSON array or JSONL file.")
        st.stop()

    st.success(f"✅ Loaded **{len(candidates):,}** candidates.")

    with st.spinner("🔍 Extracting features and scoring candidates..."):
        results = []
        for c in candidates:
            feat  = extract_features(c)
            score = compute_score(feat)
            results.append({
                "candidate":     c,
                "feat":          feat,
                "score":         score,
                "candidate_id":  feat["candidate_id"],
            })

        results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

        top_k = min(100, len(results))
        top_results = results[:top_k]

        prev_score = float("inf")
        for r in top_results:
            if r["score"] > prev_score:
                r["score"] = prev_score
            prev_score = r["score"]

        output_rows = []
        for rank_pos, r in enumerate(top_results, start=1):
            feat = r["feat"]
            reasoning = generate_reasoning(
                c=r["candidate"],
                feat=feat,
                score=r["score"],
                rank=rank_pos,
            )
            output_rows.append({
                "candidate_id":        r["candidate_id"],
                "rank":                rank_pos,
                "score":               round(r["score"], 6),
                "reasoning":           reasoning,
                # Extra display-only columns
                "_title":              r["candidate"].get("profile", {}).get("current_title", ""),
                "_years":              r["candidate"].get("profile", {}).get("years_of_experience", 0),
                "_country":            r["candidate"].get("profile", {}).get("country", ""),
                "_ml_months":          feat.get("ml_ai_months", 0),
                "_product_ml":         feat.get("product_ml_experience", False),
                "_consulting":         feat.get("entire_career_consulting", False),
                "_open_to_work":       feat.get("open_to_work", False),
                "_notice_days":        feat.get("notice_days", 90),
                "_honeypot":           feat.get("is_honeypot", False),
                "_availability":       round(feat.get("availability_score", 0), 2),
            })

    # ── KPI row ────────────────────────────────────────────────────────────────
    n_strong = sum(1 for r in output_rows if r["score"] >= 0.65)
    n_hp     = sum(1 for r in output_rows if r["_honeypot"])
    hp_rate  = n_hp / len(output_rows) if output_rows else 0
    avg_sc   = sum(r["score"] for r in output_rows) / len(output_rows) if output_rows else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Candidates ranked",  len(output_rows))
    k2.metric("Strong matches",     n_strong)
    k3.metric("Avg score",          f"{avg_sc:.3f}")
    k4.metric("Honeypots detected", n_hp)
    k5.metric("Honeypot rate",      f"{hp_rate:.1%}",
              delta="✅ safe" if hp_rate <= 0.10 else "⚠️ disqualify",
              delta_color="normal" if hp_rate <= 0.10 else "inverse")

    st.markdown("---")

    # ── Results table ──────────────────────────────────────────────────────────
    st.subheader("🏆 Ranked Candidates")

    display_df = pd.DataFrame([{
        "Rank":         r["rank"],
        "ID":           r["candidate_id"],
        "Score":        r["score"],
        "Title":        r["_title"],
        "Yrs":          r["_years"],
        "Country":      r["_country"],
        "ML Months":    r["_ml_months"],
        "Product ML":   "✅" if r["_product_ml"] else "❌",
        "Consulting":   "⚠️" if r["_consulting"] else "—",
        "Open":         "✅" if r["_open_to_work"] else "—",
        "Notice (d)":   r["_notice_days"],
        "Avail.":       r["_availability"],
        "🍯":           "🍯" if r["_honeypot"] else "",
        "Reasoning":    r["reasoning"],
    } for r in output_rows])

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Score breakdown for top candidate ─────────────────────────────────────
    if output_rows:
        top = output_rows[0]
        top_feat = results[0]["feat"]
        st.markdown("---")
        st.subheader(f"🔍 Deep Dive: Rank #1 — {top['_title']} ({top['candidate_id']})")
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Overall",        f"{top['score']:.3f}")
        d2.metric("ML Months",      top["_ml_months"])
        d3.metric("Product ML",     "Yes" if top["_product_ml"] else "No")
        d4.metric("Availability",   f"{top['_availability']:.2f}")
        d5.metric("Notice",         f"{top['_notice_days']}d")
        d6.metric("Country",        top["_country"])
        st.markdown(f"**Reasoning:** {top['reasoning']}")

    # ── CSV download ───────────────────────────────────────────────────────────
    st.markdown("---")
    csv_rows = [{"candidate_id": r["candidate_id"], "rank": r["rank"],
                 "score": r["score"], "reasoning": r["reasoning"]}
                for r in output_rows]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_bytes = buf.getvalue().encode("utf-8")

    st.download_button(
        label="⬇️ Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )

else:
    st.markdown(
        "**Upload a candidates file above to start.** "
        "Use `sample_candidates.json` from the hackathon bundle to test."
    )

st.markdown("---")
st.caption(
    "Redrob Intelligent Candidate Discovery · INDIA.RUNS Hackathon · "
    "CPU-only · No network during ranking · <5 min for 100K candidates"
)
