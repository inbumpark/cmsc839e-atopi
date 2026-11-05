"""Compare cascade paths: 1, 1->2, 1->3, 1->4 (from a regular run) and 1->2->4, 1->3->4 (chain run).

    python -m derm_eval.chains results/<base_run> results/<chain_run> [--exclude-no-context] [--n-boot 2000]

<chain_run> is a run with modes chain_124,chain_134 made with --history-from <base_run>, so the
1->2 and 1->3 conversations it continues are exactly the ones scored in <base_run>.
Writes chain_accuracy.csv, chain_paired.csv and summary.txt to <chain_run>/analysis[_with_context]/.
Accuracy is per answer, averaged per case first; CIs and paired tests resample cases.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import NO_CONTEXT, load_run
from .config import resolve

# path label -> (run, mode, stage); "base" is the regular run, "chain" the chain run
PATHS = {
    "1": ("base", "cascade", 1),
    "1->2": ("base", "cascade", 2),
    "1->3": ("base", "cascade", 3),
    "1->4": ("base", "cascade", 4),
    "1->2->4": ("chain", "chain_124", 4),
    "1->3->4": ("chain", "chain_134", 4),
    "4 (fresh chat)": ("base", "independent", 4),
}
# (path, compared against): is the extra step worth it, and does the end point depend on the route?
COMPARISONS = [
    ("1->2->4", "1->4"), ("1->3->4", "1->4"), ("1->3->4", "1->2->4"),
    ("1->2->4", "1->2"), ("1->3->4", "1->3"), ("1->4", "1"),
]
PREVIOUS = {"1->2": "1", "1->3": "1", "1->4": "1", "1->2->4": "1->2", "1->3->4": "1->3"}


def per_case(df, path):
    run, mode, stage = PATHS[path]
    g = df[(df.run == run) & (df["mode"] == mode) & (df.stage == stage)]
    return g.groupby("case_id").correct.mean()


def paired(a, b, n_boot, rng):
    """Mean of a-b over shared cases, bootstrap 95% CI and two-sided sign-flip permutation p."""
    d = (a - b).dropna().to_numpy()
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    flips = rng.choice([-1.0, 1.0], size=(n_boot * 5, len(d)))
    p = float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12))
    return d.mean(), *np.percentile(boots, [2.5, 97.5]), p, len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_run")
    ap.add_argument("chain_run")
    ap.add_argument("--exclude-no-context", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    base, cfg = load_run(resolve(args.base_run))
    chain, ccfg = load_run(resolve(args.chain_run))
    if cfg["model"] != ccfg["model"]:
        raise SystemExit(f"model differs: {cfg['model']} vs {ccfg['model']}")
    df = pd.concat([base.assign(run="base"), chain.assign(run="chain")], ignore_index=True)
    note = ""
    if args.exclude_no_context:
        no_ctx = set(df.case_id[df.error == NO_CONTEXT])
        df = df[~df.case_id.isin(no_ctx)]
        note = f" (excluded {len(no_ctx)} cases with no context fields)"
    rng = np.random.default_rng(0)

    acc_rows = []
    for path, (run, mode, stage) in PATHS.items():
        g = df[(df.run == run) & (df["mode"] == mode) & (df.stage == stage)]
        pc = g.groupby("case_id").correct.mean().to_numpy()
        boots = pc[rng.integers(0, len(pc), (args.n_boot, len(pc)))].mean(1)
        row = {"path": path, "n_cases": len(pc), "n_answers": len(g), "acc": pc.mean(),
               "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
               "parse_fail": (~g.parse_ok).mean(), "mean_confidence": g.confidence.mean()}
        if path in PREVIOUS:      # how often the last step changed the answer of the same sample
            prun, pmode, pstage = PATHS[PREVIOUS[path]]
            prev = df[(df.run == prun) & (df["mode"] == pmode) & (df.stage == pstage)]
            m = g.merge(prev, on=["case_id", "sample"], suffixes=("", "_prev"))
            row["changed_vs_previous"] = (m.pred != m.pred_prev).mean()
        acc_rows.append(row)
    acc = pd.DataFrame(acc_rows)

    pair_rows = []
    for a, b in COMPARISONS:
        diff, lo, hi, p, n = paired(per_case(df, a), per_case(df, b), args.n_boot, rng)
        pair_rows.append({"path": a, "vs": b, "n_cases": n, "diff": diff, "ci_low": lo, "ci_high": hi, "p": p})
    pairs = pd.DataFrame(pair_rows)

    out = Path(resolve(args.chain_run)) / ("analysis_with_context" if args.exclude_no_context else "analysis")
    out.mkdir(exist_ok=True)
    acc.to_csv(out / "chain_accuracy.csv", index=False)
    pairs.to_csv(out / "chain_paired.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        summary = (f"model: {cfg['model']}   base: {args.base_run}   chains: {args.chain_run}{note}\n\n"
                   f"== accuracy by path ==\n{acc.to_string(index=False)}\n\n"
                   f"== paired differences (path - vs) ==\n{pairs.to_string(index=False)}\n")
    (out / "summary.txt").write_text(summary)
    print(summary + f"\nwrote {out}")


if __name__ == "__main__":
    main()
