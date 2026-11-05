"""Build configs/mcq_label_pool.json: the SCIN condition names MCQ distractors are drawn from.

    python scripts/make_mcq_pool.py

Reads the public SCIN labels CSV over HTTPS (no Google Cloud login needed) and keeps every
dermatologist-given condition name with the number of cases it appears in. Names that are Eczema
under another name, or unspecified dermatitis that would also be a fair answer for an eczema case,
are excluded so that exactly one option is correct; they are listed in the file under "excluded".
"""
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

URL = "https://storage.googleapis.com/dx-scin-public-data/dataset/scin_labels.csv"
OUT = Path(__file__).resolve().parent.parent / "configs" / "mcq_label_pool.json"
TARGET = "Eczema"

# Unspecified dermatitis: not wrong for an eczema case, so not a fair distractor.
NONSPECIFIC_DERMATITIS = {
    "Acute dermatitis", "Acute dermatitis, NOS", "Chronic dermatitis", "Chronic dermatitis, NOS",
    "Acute and chronic dermatitis", "Acute vesicular dermatitis", "Pruritic dermatitis",
    "resolving dermatitis",
}


def excluded_reason(name):
    if name == TARGET:
        return "target"
    if re.search(r"eczem", name, re.I):
        return "eczema variant"
    if name in NONSPECIFIC_DERMATITIS:
        return "nonspecific dermatitis"
    return None


def main():
    df = pd.read_csv(URL, dtype={"case_id": str})
    counts = Counter()
    for cell in df["dermatologist_skin_condition_on_label_name"].dropna():
        counts.update(set(ast.literal_eval(cell)))
    labels, excluded = {}, {}
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        reason = excluded_reason(name)
        if reason:
            excluded[name] = reason
        else:
            labels[name] = n
    OUT.write_text(json.dumps({
        "source": URL,
        "target": TARGET,
        "note": "labels: condition name -> number of SCIN cases with that dermatologist label",
        "labels": labels,
        "excluded": excluded,
    }, indent=1) + "\n")
    print(f"{len(labels)} distractor labels, {len(excluded)} excluded -> {OUT}")
    for name, reason in excluded.items():
        print(f"  excluded ({reason}): {name}")


if __name__ == "__main__":
    main()
