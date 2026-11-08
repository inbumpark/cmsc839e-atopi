"""Rank SCIN conditions by how rarely dermatologists list them together with Eczema.

    python scripts/rank_mcq_distractors.py [--min-cases 30] [--top 15]

For each condition X in configs/mcq_label_pool.json with at least --min-cases SCIN cases, prints
the share of cases with X in their dermatologist differential that also list Eczema. A low share
means dermatologists (almost) never confuse X with Eczema. The top 3 are the fixed distractors
in configs/mcq_far.yaml (Acne 4.1%, Rosacea 5.8%, Herpes Zoster 8.5% as of Nov 2026).

Pearson correlation of label weights is not used: it mostly reflects how common a label is (Insect
Bite and Urticaria come out "most negative" yet share a differential with Eczema in ~21% of cases).
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

URL = "https://storage.googleapis.com/dx-scin-public-data/dataset/scin_labels.csv"
POOL = Path(__file__).resolve().parent.parent / "configs" / "mcq_label_pool.json"
TARGET = "Eczema"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-cases", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    pool = json.loads(POOL.read_text())["labels"]
    df = pd.read_csv(URL, dtype={"case_id": str})
    diffs = [d for d in map(ast.literal_eval, df["weighted_skin_condition_label"].dropna()) if d]
    n, with_target = Counter(), Counter()
    for d in diffs:
        for name in d:
            n[name] += 1
            with_target[name] += TARGET in d
    rows = sorted((with_target[x] / n[x], x) for x in n if x in pool and n[x] >= args.min_cases)
    print(f"{len(diffs)} SCIN cases with a dermatologist differential\n")
    print(f"{'condition':40} {'cases':>5} {'also ' + TARGET:>12} {'share':>6}")
    for share, x in rows[: args.top]:
        print(f"{x:40} {n[x]:5} {with_target[x]:12} {share * 100:5.1f}%")


if __name__ == "__main__":
    main()
