"""Analyze a confidence-only run (answer_format: confidence, e.g. configs/conf.yaml).

    python -m derm_eval.conf results/<run_name> [--n-boot 2000]

Each answer is the model's 1-5 confidence that the stated diagnosis (eczema) is right. Tables are
by path (1, 1->2, 1->3, 1->2->4, 1->3->4, and any other mode x stage the run has). Every case in
the data is labelled Eczema, so higher confidence is "more right". Two dermatologist references
are compared with it: their own 1-5 confidence in the label (label_confidence) and their weight
for Eczema in the case's differential (1.0 = every dermatologist said only Eczema).

Cases without any context field are left out (only stage 1 can run for them). Unparseable answers
are left out of the means and counted in parse_fail. Statistics average the samples of a case
first; CIs and p-values resample cases. Writes CSVs and summary.txt to results/<run_name>/conf_analysis/.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .config import resolve
from .schema import load_cases
from .stages import SHARED

NO_CONTEXT = "no context fields"
# path label -> (mode, stage), in the order the questions are asked
PATHS = {
    "1": (SHARED, 1),
    "1->2": ("cascade", 2), "1->3": ("cascade", 3), "1->4": ("cascade", 4),
    "1->2->4": ("chain_124", 4), "1->3->4": ("chain_134", 4),
    "2 (new chat)": ("independent", 2), "3 (new chat)": ("independent", 3), "4 (new chat)": ("independent", 4),
}
PREVIOUS = {"1->2": "1", "1->3": "1", "1->4": "1", "1->2->4": "1->2", "1->3->4": "1->3"}
COMPARISONS = [("1->3", "1->2"), ("1->2->4", "1"), ("1->3->4", "1"), ("1->3->4", "1->2->4"),
               ("1->2->4", "1->4"), ("1->3->4", "1->4")]
WEIGHT_BINS = [0, 0.25, 0.5, 0.75, 0.999, 1.0]
WEIGHT_LABELS = ["<=0.25", "0.25-0.5", "0.5-0.75", "0.75-<1", "1.0"]


def load_run(run_dir):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if cfg.get("answer_format") != "confidence":
        raise SystemExit(f"{run_dir} is a {cfg.get('answer_format', 'free_text')} run; use derm_eval.analyze")
    rows = [json.loads(l) for l in (run_dir / "raw.jsonl").open() if l.strip()]
    if not rows:
        raise SystemExit("raw.jsonl is empty")
    df = pd.DataFrame(rows)
    cases = {c.case_id: c for c in load_cases(resolve(cfg["data_path"]))}
    df = df[df.case_id.isin(cases)].copy()
    target = cfg.get("target_label", "Eczema")
    df["derm_weight"] = df.case_id.map(lambda c: float(cases[c].differential.get(target, np.nan))
                                       if cases[c].differential else np.nan)
    df["derm_confidence"] = pd.to_numeric(df.case_id.map(lambda c: cases[c].label_confidence), errors="coerce")
    df["confidence"] = pd.to_numeric(df.confidence, errors="coerce")
    df["field"] = [f[0] if s in (2, 3) and f else None for f, s in zip(df.fields_given, df.stage)]
    by_key = {v: k for k, v in PATHS.items()}
    df["path"] = [by_key.get((m, s), f"{m} stage {s}") for m, s in zip(df["mode"], df.stage)]
    return df, cfg


def boot_ci(values, n_boot, rng):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    return tuple(np.percentile(v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1), [2.5, 97.5]))


def paired(d, n_boot, rng):
    """Mean of per-case differences d, bootstrap 95% CI and two-sided sign-flip permutation p."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return {"n_cases": len(d), "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    lo, hi = boot_ci(d, n_boot, rng)
    flips = rng.choice([-1.0, 1.0], size=(n_boot * 5, len(d)))
    p = float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12))
    return {"n_cases": len(d), "diff": d.mean(), "ci_low": lo, "ci_high": hi, "p": p}


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or len(set(a[ok])) < 2 or len(set(b[ok])) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stats.spearmanr(a[ok], b[ok]).statistic


def path_table(df, order, n_boot, rng):
    """Level of the confidence on each path, and how it compares with the dermatologists."""
    out = []
    for path in order:
        g = df[df.path == path]
        pc = g.groupby("case_id").agg(c=("confidence", "mean"), sd=("confidence", "std"),
                                      w=("derm_weight", "first"), dc=("derm_confidence", "first"))
        lo, hi = boot_ci(pc.c, n_boot, rng)
        share = g.confidence.dropna().value_counts(normalize=True)
        row = {"path": path, "n_cases": len(pc), "n_answers": len(g), "parse_fail": (~g.parse_ok).mean(),
               "mean_conf": pc.c.mean(), "ci_low": lo, "ci_high": hi}
        row.update({f"conf_{i}": share.get(i, 0.0) for i in range(1, 6)})       # share of answers at each level
        row.update({
            "sd_across_seeds": pc.sd.mean(),                  # spread of a case's samples (uncertainty)
            "mean_derm_conf": pc.dc.mean(), "diff_vs_derm_conf": (pc.c - pc.dc).mean(),
            "mae_vs_derm_conf": (pc.c - pc.dc).abs().mean(),
            "spearman_vs_derm_conf": spearman(pc.c, pc.dc),
            "spearman_vs_derm_weight": spearman(pc.c, pc.w),
        })
        out.append(row)
    return pd.DataFrame(out)


def step_table(df, order, n_boot, rng):
    """What the last step of each path did to the same sample's previous rating."""
    out = []
    for path in order:
        if PREVIOUS.get(path) not in order:
            continue
        a = df[df.path == path][["case_id", "sample", "confidence"]]
        b = df[df.path == PREVIOUS[path]][["case_id", "sample", "confidence"]]
        m = a.merge(b, on=["case_id", "sample"], suffixes=("", "_prev")).dropna()
        m["d"] = m.confidence - m.confidence_prev
        row = {"path": path, "previous": PREVIOUS[path], "n_answers": len(m),
               **paired(m.groupby("case_id").d.mean(), n_boot, rng)}
        row.update(mean_abs_change=m.d.abs().mean(), raised=(m.d > 0).mean(), lowered=(m.d < 0).mean(),
                   unchanged=(m.d == 0).mean())
        out.append(row)
    return pd.DataFrame(out).rename(columns={"diff": "mean_change"})


def comparison_table(df, order, n_boot, rng):
    pc = df.groupby(["path", "case_id"]).confidence.mean().unstack(0)
    out = []
    for a, b in COMPARISONS:
        if a in order and b in order:
            out.append({"path": a, "vs": b, **paired(pc[a] - pc[b], n_boot, rng)})
    return pd.DataFrame(out)


def field_table(df):
    """Stages 2 and 3 by the single field that was given, with the change from the stage-1 rating."""
    s1 = df[df.path == "1"][["case_id", "sample", "confidence"]].rename(columns={"confidence": "c1"})
    d = df[df.stage.isin([2, 3]) & df.field.notna()].merge(s1, on=["case_id", "sample"], how="left")
    d["change"] = d.confidence - d.c1
    t = (d.groupby(["path", "field"])
         .agg(n=("confidence", "size"), mean_conf=("confidence", "mean"), mean_change_vs_1=("change", "mean"))
         .reset_index())
    t["share_of_path"] = t.n / t.groupby("path").n.transform("sum")
    return t.sort_values(["path", "n"], ascending=[True, False])


def by_reference(df, order, col):
    """Mean model confidence per path, by a dermatologist reference column (already binned)."""
    d = df.dropna(subset=[col])
    pc = d.groupby([col, "path", "case_id"], observed=True).confidence.mean()
    t = pc.groupby(level=[0, 1], observed=True).mean().unstack(1).reindex(columns=order)
    t.insert(0, "n_cases", d.groupby(col, observed=True).case_id.nunique())
    return t.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    run_dir = resolve(args.run_dir)
    df, cfg = load_run(run_dir)
    no_ctx = set(df.case_id[df.error == NO_CONTEXT])
    if df.stage.max() > 1:
        no_ctx |= set(df.case_id) - set(df.case_id[df.stage > 1])   # later stages have not run yet
    df = df[~df.case_id.isin(no_ctx)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no cases with context fields in this run")
    order = [p for p in PATHS if p in set(df.path)] + sorted(set(df.path) - set(PATHS))
    rng = np.random.default_rng(0)
    df["derm_weight_bin"] = pd.cut(df.derm_weight, WEIGHT_BINS, labels=WEIGHT_LABELS, include_lowest=True)

    tables = {
        "conf_by_path": path_table(df, order, args.n_boot, rng),
        "conf_step_change": step_table(df, order, args.n_boot, rng),
        "conf_paired_paths": comparison_table(df, order, args.n_boot, rng),
        "conf_by_field": field_table(df),
        "conf_by_derm_confidence": by_reference(df, order, "derm_confidence"),
        "conf_by_derm_weight": by_reference(df, order, "derm_weight_bin"),
    }
    out_dir = run_dir / "conf_analysis"
    out_dir.mkdir(exist_ok=True)
    df.drop(columns=["raw", "select_raw"]).to_csv(out_dir / "answers.csv", index=False)
    lines = [f"run: {args.run_dir}   model: {cfg['model']}   stated diagnosis: {cfg.get('target_label', 'Eczema')}   "
             f"samples/case: {cfg['n_samples']}   cases: {df.case_id.nunique()}   answers: {len(df)}   "
             f"(left out {len(no_ctx)} cases with no context or unfinished)", ""]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        for name, t in tables.items():
            t.to_csv(out_dir / f"{name}.csv", index=False)
            lines += [f"== {name} ==", t.to_string(index=False) if len(t) else "(empty)", ""]
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"wrote tables and summary.txt to {out_dir}")


if __name__ == "__main__":
    main()
