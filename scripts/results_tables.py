"""Print the README results tables (markdown) for MCQ runs, on cases that have context.

    python scripts/results_tables.py                      # random distractors, all Eczema cases
    python scripts/results_tables.py --set far            # fixed far distractors, certain cases
    python scripts/results_tables.py --runs "Qwen3-VL 8B=mcq" "Gemma 3 4B=mcq_gemma3_4b"

Each run name R reads results/R/raw.jsonl plus results/R_chains/raw.jsonl if present (paths
1->2->4, 1->3->4). Accuracy is per answer, averaged per case; [95% bootstrap CI over cases];
paired p-values are sign-flip permutation tests over cases.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETS = {
    "random": {"Qwen3-VL 8B": "mcq", "MedGemma 4B": "mcq_medgemma4b", "Gemma 3 4B": "mcq_gemma3_4b"},
    "far": {"Qwen3-VL 8B": "mcq_far", "MedGemma 4B": "mcq_far_medgemma4b", "Gemma 3 4B": "mcq_far_gemma3_4b"},
}
PATHS = [("1 image only", "shared", 1), ("2 random field (new chat)", "independent", 2),
         ("3 model-chosen field (new chat)", "independent", 3), ("4 all context (new chat)", "independent", 4),
         ("1→2", "cascade", 2), ("1→3", "cascade", 3), ("1→4", "cascade", 4),
         ("1→2→4", "chain_124", 4), ("1→3→4", "chain_134", 4)]
PREV = {("cascade", 2): ("shared", 1), ("cascade", 3): ("shared", 1), ("cascade", 4): ("shared", 1),
        ("chain_124", 4): ("cascade", 2), ("chain_134", 4): ("cascade", 3)}
COMPARISONS = [("2 vs 1 (new chat)", ("independent", 2), ("shared", 1)),
               ("3 vs 2 (new chat)", ("independent", 3), ("independent", 2)),
               ("4 vs 1 (new chat)", ("independent", 4), ("shared", 1)),
               ("1→3 vs 1→2", ("cascade", 3), ("cascade", 2)),
               ("1→4 vs 1", ("cascade", 4), ("shared", 1)),
               ("1→2→4 vs 1→4", ("chain_124", 4), ("cascade", 4)),
               ("1→3→4 vs 1→4", ("chain_134", 4), ("cascade", 4))]
FIELDS = ["symptoms", "texture", "body_location", "other_symptoms", "sex", "age", "fitzpatrick", "duration"]
RNG = np.random.default_rng(0)


def load(run):
    recs = []
    for r in (run, run + "_chains"):
        p = RESULTS / r / "raw.jsonl"
        if p.exists():
            recs += [json.loads(l) for l in p.open() if l.strip()]
    no_ctx = {d["case_id"] for d in recs if d.get("error") == "no context fields"}
    by = defaultdict(dict)
    for d in recs:
        if d["case_id"] not in no_ctx:
            by[(d["mode"], d["stage"])][(d["case_id"], d["sample"])] = d
    return by, len({d["case_id"] for d in recs} - no_ctx)


def correct(d):
    return d.get("answer") is not None and d["answer"] == d["correct_answer"]


def per_case(recs):
    g = defaultdict(list)
    for (cid, _), d in recs.items():
        g[cid].append(correct(d))
    return {c: float(np.mean(v)) for c, v in g.items()}


def cell(by, mode, stage):
    recs = by.get((mode, stage))
    if not recs:
        return None
    v = [d for d in recs.values() if d["parse_ok"]]
    y = np.array([correct(d) for d in v])
    c = np.array([d["confidence"] for d in v])
    pc = np.array(list(per_case(recs).values()))
    boots = pc[RNG.integers(0, len(pc), (2000, len(pc)))].mean(1)
    pos, neg = c[y], c[~y]
    out = {"acc": pc.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
           "conf": c.mean(), "five": np.mean(c == 5),
           "c_right": pos.mean() if len(pos) else np.nan, "c_wrong": neg.mean() if len(neg) else np.nan,
           "auroc": ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
           if len(pos) and len(neg) else np.nan}
    if (mode, stage) in PREV:
        prev = by[PREV[(mode, stage)]]
        out["changed"] = np.mean([d["answer"] != prev[k]["answer"] for k, d in recs.items() if k in prev])
    return out


def table(title, names, cells, fmt):
    print(f"\n{title}\n\n| path | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for label, mode, stage in PATHS:
        row = [cells[(n, label)] for n in names]
        if all(x is None for x in row):
            continue
        print(f"| {label} | " + " | ".join("–" if x is None else fmt(x) for x in row) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=SETS, default="random")
    ap.add_argument("--runs", nargs="+", metavar="LABEL=RUN", help="override the model -> run mapping")
    args = ap.parse_args()
    models = dict(r.split("=", 1) for r in args.runs) if args.runs else SETS[args.set]
    data = {n: load(r) for n, r in models.items()}
    names = list(models)
    print("cases with context:", {n: data[n][1] for n in names})
    cells = {(n, p[0]): cell(data[n][0], p[1], p[2]) for n in names for p in PATHS}

    table("Accuracy (% of answers that picked Eczema) [95% CI]", names, cells,
          lambda x: f"{x['acc'] * 100:.1f} [{x['lo'] * 100:.1f}, {x['hi'] * 100:.1f}]")
    table("Mean confidence (% rated 5)", names, cells, lambda x: f"{x['conf']:.2f} ({x['five'] * 100:.0f}%)")
    table("AUROC of confidence vs correct (mean confidence right / wrong)", names, cells,
          lambda x: f"{x['auroc']:.2f} ({x['c_right']:.2f} / {x['c_wrong']:.2f})")
    table("Answer changed by the last turn", names, cells,
          lambda x: f"{x['changed'] * 100:.0f}%" if "changed" in x else "–")

    print("\nPaired differences (percentage points)\n\n| comparison | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for label, a, b in COMPARISONS:
        row = []
        for n in names:
            by = data[n][0]
            if a not in by or b not in by:
                row.append("–"); continue
            A, B = per_case(by[a]), per_case(by[b])
            d = np.array([A[c] - B[c] for c in A if c in B])
            flips = RNG.choice([-1.0, 1.0], (10000, len(d)))
            p = np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12)
            row.append(f"{d.mean() * 100:+.1f} ({'p<0.001' if p < 0.001 else f'p={p:.3f}'})")
        print(f"| {label} | " + " | ".join(row) + " |")

    print("\nPer field: accuracy change when given at random (stage 2 vs same sample's stage 1) | share asked in stage 3\n")
    print("| field | " + " | ".join(f"{n} Δ" for n in names) + " | " + " | ".join(f"{n} asks" for n in names)
          + " |\n|---|" + "---|" * 2 * len(names))
    for f in FIELDS:
        eff, ask = [], []
        for n in names:
            by = data[n][0]
            s1 = by[("shared", 1)]
            v = [d for d in by[("independent", 2)].values() if d["fields_given"] == [f]]
            eff.append(f"{np.mean([correct(d) - correct(s1[(d['case_id'], d['sample'])]) for d in v]) * 100:+.1f} (n={len(v)})"
                       if v else "–")
            asked = [d["requested_field"] for d in by[("independent", 3)].values() if d["requested_field"]]
            ask.append(f"{Counter(asked)[f] / len(asked) * 100:.0f}%")
        print(f"| {f} | " + " | ".join(eff) + " | " + " | ".join(ask) + " |")

    print("\nPosition bias and wrong picks (all paths)")
    for n in names:
        v = [d for recs in data[n][0].values() for d in recs.values() if d["parse_ok"]]
        picks = Counter(d["answer"] for d in v)
        acc = defaultdict(list)
        for d in v:
            acc[d["correct_answer"]].append(correct(d))
        wrong = Counter(d["diagnosis"] for d in v if not correct(d))
        print(f"- {n}: picks " + " ".join(f"{l}={picks[l] / len(v) * 100:.0f}%" for l in "ABCD")
              + " | acc when Eczema is " + " ".join(f"{l}={np.mean(acc[l]) * 100:.0f}%" for l in "ABCD" if acc[l])
              + " | wrong picks: " + ", ".join(f"{w} {c / len(v) * 100:.0f}%" for w, c in wrong.most_common(3)))


if __name__ == "__main__":
    main()
