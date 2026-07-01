#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery

Single command that produces submission.csv from candidates.jsonl within
the compute constraints (5 min, 16 GB RAM, CPU only, no network).

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
    python rank.py --candidates candidates.jsonl.gz --out submission.csv
    python rank.py --candidates candidates.jsonl --out submission.csv --top_k 100
    python rank.py --candidates sample_candidates.json --out submission.csv --sample
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

from src.features import extract_features
from src.scorer import compute_score
from src.reasoning import generate_reasoning


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    p.add_argument("--candidates", required=True,
                   help="Path to candidates.jsonl, candidates.jsonl.gz, or sample JSON")
    p.add_argument("--out",        required=True,
                   help="Output CSV path (e.g. submission.csv)")
    p.add_argument("--top_k",      type=int, default=100,
                   help="Number of candidates to output (default 100)")
    p.add_argument("--sample",     action="store_true",
                   help="Input is a JSON array (sample_candidates.json format)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-candidate scores while running")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_candidates(path: str, is_sample: bool = False):
    """
    Generator that yields candidate dicts.
    Handles: .jsonl, .jsonl.gz, and JSON array (sample_candidates.json).
    """
    path = Path(path)

    if is_sample or path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return

    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")

    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ── Main ranking pipeline ─────────────────────────────────────────────────────
def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("  Redrob Intelligent Candidate Ranker")
    print("=" * 60)
    print(f"\n📂 Loading candidates from: {args.candidates}")

    # ── Step 1: Extract features for all candidates ───────────────────────────
    results = []
    n_loaded = 0
    n_honeypot = 0

    for c in load_candidates(args.candidates, is_sample=args.sample):
        n_loaded += 1
        feat  = extract_features(c)
        score = compute_score(feat)

        if feat.get("is_honeypot"):
            n_honeypot += 1

        results.append({
            "candidate_raw": c,
            "feat":          feat,
            "score":         score,
            "candidate_id":  feat["candidate_id"],
        })

        if args.verbose and n_loaded % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {n_loaded:,} | elapsed {elapsed:.1f}s")

    print(f"✅ Loaded & scored {n_loaded:,} candidates in {time.time()-t0:.1f}s")
    print(f"   Honeypot candidates flagged: {n_honeypot:,}")

    # ── Step 2: Sort by score (descending) ────────────────────────────────────
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # ── Step 3: Take top K ────────────────────────────────────────────────────
    top_k = min(args.top_k, len(results))
    top_results = results[:top_k]

    # Verify honeypot rate in top 100
    hp_in_top = sum(1 for r in top_results if r["feat"].get("is_honeypot"))
    hp_rate = hp_in_top / top_k if top_k > 0 else 0
    print(f"   Honeypots in top {top_k}: {hp_in_top} ({hp_rate:.1%}) — {'⚠️ WARNING' if hp_rate > 0.10 else '✅ OK'}")

    # ── Step 4: Assign ranks and ensure non-increasing scores ─────────────────
    # Scores are already sorted; normalise to [0, 1] range  
    # and ensure strict non-increase where there are floating point ties
    top_scores = [r["score"] for r in top_results]
    prev_score = float("inf")
    for i, r in enumerate(top_results):
        # Enforce non-increasing: if tie, keep same score; never increase
        if r["score"] > prev_score:
            r["score"] = prev_score
        prev_score = r["score"]

    # ── Step 5: Generate reasoning for each candidate ─────────────────────────
    print(f"\n📝 Generating reasoning strings for top {top_k} candidates...")
    output_rows = []
    for rank_pos, r in enumerate(top_results, start=1):
        reasoning = generate_reasoning(
            c=r["candidate_raw"],
            feat=r["feat"],
            score=r["score"],
            rank=rank_pos,
        )
        output_rows.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank_pos,
            "score":        round(r["score"], 6),
            "reasoning":    reasoning,
        })

    # ── Step 6: Write CSV ─────────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(output_rows)

    elapsed = time.time() - t0
    print(f"\n✅ Submission written to: {out_path}")
    print(f"   Rows written : {len(output_rows)}")
    print(f"   Total runtime: {elapsed:.2f}s")
    print(f"\nTop 10 candidates:")
    for row in output_rows[:10]:
        print(f"  #{row['rank']:>3}  {row['candidate_id']}  score={row['score']:.4f}  {row['reasoning'][:70]}...")
    print()


if __name__ == "__main__":
    main()
