# Redrob Intelligent Candidate Discovery & Ranking

**INDIA.RUNS Hackathon — Redrob × H2S**  
Challenge: Data & AI — Intelligent Candidate Discovery  
Participant: Rahul (IIT Kharagpur) · [@Rahul45f](https://github.com/Rahul45f)

---

## Reproduce the Submission (Single Command)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: **~30 seconds** for 100,000 candidates on CPU | Memory: <2 GB | No network.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Rahul45f/redrob-ranker.git
cd redrob-ranker

# 2. Python 3.9+ required (no GPU, no network needed)
pip install -r requirements.txt

# 3. Place candidates.jsonl in the repo root (from hackathon bundle)
# OR use the sample:
python rank.py --candidates sample_candidates.json --out submission.csv --sample
```

---

## Project Structure

```
redrob-ranker/
├── rank.py                       ← Main ranking script (produces submission.csv)
├── app.py                        ← Streamlit sandbox (≤100 candidates)
├── requirements.txt
├── submission_metadata.yaml
├── validate_submission.py        ← Official format validator (from hackathon bundle)
├── README.md
└── src/
    ├── __init__.py
    ├── features.py               ← Feature extraction from candidate JSON
    ├── scorer.py                 ← Weighted composite scoring
    └── reasoning.py              ← Per-candidate reasoning generation
```

---

## Architecture

```
candidates.jsonl (100,000 candidates)
         │
         ▼  load_candidates() → generator, no full file in memory at once
         │
         ▼  extract_features(candidate)
         │    ├─ detect_honeypot()      → flag impossible profiles
         │    ├─ analyze_career()       → classify job titles, detect consulting
         │    ├─ analyze_skills()       → trust-weighted skill matching
         │    ├─ analyze_signals()      → behavioral availability composite
         │    ├─ analyze_experience()   → years vs JD sweet-spot (5-9yr)
         │    ├─ analyze_education()    → tier + degree scoring
         │    └─ analyze_location()     → India / preferred city / relocation
         │
         ▼  compute_score(features)     → float [0, 1]
         │
         ▼  sort by score (descending), take top 100
         │
         ▼  generate_reasoning(candidate, features, rank)
         │
         ▼  submission.csv
              candidate_id, rank, score, reasoning
```

---

## Scoring Methodology

| Component | Weight | Key Logic |
|-----------|--------|-----------|
| **Career fitness** | 40% | Fraction of career months in ML/AI/NLP/Search roles at product companies. Entire-career-consulting (TCS, Infosys, Wipro, etc.) → 65% penalty. Current title classified as `ml_ai`, `data_adjacent`, or `irrelevant`. |
| **Skills fitness** | 28% | Trust-weighted: `proficiency_factor × f(duration_months) × f(endorsements)`. Expert skills with 0 months used are down-weighted. Keyword stuffers (high skill count but zero career ML evidence) get a 5× penalty. |
| **Behavioral availability** | 18% | Composite of last_active_date recency, open_to_work_flag, recruiter_response_rate, notice_period_days, interview_completion_rate, GitHub activity score. |
| **Experience range** | 8% | JD sweet-spot is 5-9 years (ideal 6-8). Score peaks at 6-8 years, diminishes outside. |
| **Education** | 4% | Institution tier (tier_1=1.0 → tier_4=0.45) × degree level. |
| **Location** | 2% | India in preferred cities (Pune/Noida/Hyderabad/Mumbai/Delhi NCR) = 1.0. Willing to relocate = 0.6. |

### Honeypot Detection

Per `submission_spec.md Section 7`, the dataset contains ~80 honeypot candidates with impossible profiles. The ranker flags them at score = 0.0:

- Expert/advanced proficiency on ≥5 skills with 0 months duration
- Sum of career duration_months > claimed years × 24 (2× expected)
- Job start_date after end_date
- Single-job career with claimed experience far exceeding the job's duration

**Result on full dataset:** 0 honeypots in top 100 (0.0% rate — well below 10% disqualification threshold).

### Why Career Fitness Is the Dominant Signal

The JD explicitly states: *"The right answer involves reasoning about the gap between what the JD says and what the JD means. A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit, no matter how perfect their skill list looks."*

This is implemented by computing the fraction of actual career months spent in ML/AI roles at product companies, and using that as a strong prior that overrides skill keyword matching.

---

## Streamlit Sandbox

```bash
streamlit run app.py
```

Accepts a JSON or JSONL file of ≤100 candidates, runs the full ranking pipeline, and provides a downloadable submission.csv.

Live sandbox: **https://rahul45f-redrob-ranker.streamlit.app**  
*(deploy to Streamlit Cloud free tier — instructions in Deployment section below)*

---

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub as a **public** repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → `app.py` as the main file
4. Click **Deploy**
5. Copy the URL and paste into `submission_metadata.yaml` → `sandbox_link`

---

## Compute Constraints Compliance

| Constraint | Limit | Actual |
|-----------|-------|--------|
| Runtime | ≤ 5 min | ~30 sec |
| Memory | ≤ 16 GB | < 2 GB |
| GPU | None | None used |
| Network | None during ranking | None (stdlib only) |

---

## Validate Before Submitting

```bash
python validate_submission.py submission.csv
# Expected: "Submission is valid."
```

---

## Full CLI Options

```bash
python rank.py --help

# Full dataset
python rank.py --candidates candidates.jsonl --out submission.csv

# Gzipped dataset
python rank.py --candidates candidates.jsonl.gz --out submission.csv

# Sample JSON (50 candidates from bundle)
python rank.py --candidates sample_candidates.json --out test.csv --sample

# Verbose mode (prints progress every 10k candidates)
python rank.py --candidates candidates.jsonl --out submission.csv --verbose
```
