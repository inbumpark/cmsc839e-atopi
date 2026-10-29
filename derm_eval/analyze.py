"""Compute RQ1-RQ3 metrics and plots from a run directory.

    python -m derm_eval.analyze results/<run_name> [--labels configs/labels.yaml] [--n-boot 2000]

Writes CSV tables, PNG plots and summary.txt to results/<run_name>/analysis/. Grading happens here
(not at query time), so after extending labels.yaml you can re-run this without re-querying the model.
Unparseable answers count as incorrect.
"""
import argparse
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .config import resolve
from .normalize import UNMATCHED, Normalizer
from .schema import load_cases
from .stages import SHARED

UNPARSED = "UNPARSED"
MODE_COLORS = {"independent": "#2a78d6", "cascade": "#eb6834"}          # categorical slots 1-2
STAGE_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}  # categorical slots 1-4
STAGE_NAMES = {1: "1 image only", 2: "2 random field", 3: "3 model-chosen field", 4: "4 all context"}


# ---- loading ---------------------------------------------------------------------------------
def load_run(run_dir, labels_path=None):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    rows = [json.loads(l) for l in (run_dir / "raw.jsonl").open() if l.strip()]
    if not rows:
        raise SystemExit("raw.jsonl is empty")
    df = pd.DataFrame(rows)
    cases = {c.case_id: c for c in load_cases(resolve(cfg["data_path"]))}
    df = df[df.case_id.isin(cases)].copy()

    norm = Normalizer(labels_path or resolve(cfg["labels_path"]), extra_labels={c.label for c in cases.values()})
    target = cfg.get("target_label", "Eczema")
    df["label"] = df.case_id.map(lambda c: cases[c].label)
    df["label_confidence"] = df.case_id.map(lambda c: cases[c].label_confidence)
    df["pred"] = [norm(d) if ok else UNPARSED for d, ok in zip(df.diagnosis, df.parse_ok)]
    df["correct"] = df.pred == df.label
    df["in_differential"] = [p == l or p in cases[c].differential for p, l, c in zip(df.pred, df.label, df.case_id)]
    df["pred_target"] = df.pred == target
    df["true_target"] = df.label == target
    df["field"] = [f[0] if s in (2, 3) and f else None for f, s in zip(df.fields_given, df.stage)]

    # Stage 1 is shared by both modes; duplicate it so every mode has stages 1-4.
    modes = [m for m in cfg["modes"]] or ["independent"]
    shared = df[df["mode"] == SHARED]
    df = pd.concat([df[df["mode"] != SHARED]] + [shared.assign(mode=m) for m in modes], ignore_index=True)
    return df.sort_values(["mode", "stage", "case_id", "sample"]).reset_index(drop=True), cfg


# ---- statistics helpers ---------------------------------------------------------------------
def bootstrap_ci(values, n_boot, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def prf(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return precision, recall


def entropy(counts):
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def auroc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


# ---- tables -------------------------------------------------------------------------------
def accuracy_table(df, n_boot):
    out = []
    for (mode, stage), g in df.groupby(["mode", "stage"]):
        per_case = g.groupby("case_id")[["correct", "in_differential"]].mean()
        p, r = prf(g.pred_target, g.true_target)
        out.append({
            "mode": mode, "stage": stage, "n_cases": g.case_id.nunique(), "n_answers": len(g),
            "parse_rate": g.parse_ok.mean(),
            "acc_exact": per_case.correct.mean(), **dict(zip(["acc_exact_lo", "acc_exact_hi"],
                                                            bootstrap_ci(per_case.correct, n_boot))),
            "acc_in_differential": per_case.in_differential.mean(),
            "eczema_acc": (g.pred_target == g.true_target).mean(), "eczema_precision": p, "eczema_recall": r,
            "mean_confidence": g.confidence.mean(),
        })
    return pd.DataFrame(out)


def majority_votes(df):
    out = []
    for (mode, stage, case_id), g in df.groupby(["mode", "stage", "case_id"]):
        counts = Counter(g.pred)
        pred, top = counts.most_common(1)[0]
        first = g.iloc[0]
        out.append({"mode": mode, "stage": stage, "case_id": case_id, "label": first.label,
                    "majority_pred": pred, "n": len(g), "agreement": top / len(g),
                    "entropy_bits": entropy(counts.values()), "n_distinct": len(counts),
                    "correct": pred == first.label,
                    "mean_confidence": g.confidence.mean(), "label_confidence": first.label_confidence})
    return pd.DataFrame(out)


def majority_table(maj, target):
    out = []
    for (mode, stage), g in maj.groupby(["mode", "stage"]):
        pred_t = g.majority_pred == target
        true_t = g.label == target
        p, r = prf(pred_t, true_t)
        out.append({"mode": mode, "stage": stage, "n_cases": len(g), "majority_acc": g.correct.mean(),
                    "eczema_acc": (pred_t == true_t).mean(), "eczema_precision": p, "eczema_recall": r,
                    "mean_agreement": g.agreement.mean(), "mean_entropy_bits": g.entropy_bits.mean()})
    return pd.DataFrame(out)


def rq2_table(df, maj, n_boot):
    """Stage 2 (random field) vs stage 3 (model-chosen field), paired by case."""
    out = []
    for mode in sorted(df["mode"].unique()):
        m = maj[maj["mode"] == mode].pivot(index="case_id", columns="stage", values="correct")
        s = df[df["mode"] == mode].groupby(["case_id", "stage"]).correct.mean().unstack()
        if not {2, 3} <= set(m.columns):
            continue
        m, s = m[[2, 3]].dropna().astype(bool), s[[2, 3]].dropna()
        b = int((m[2] & ~m[3]).sum())   # random right, chosen wrong
        c = int((~m[2] & m[3]).sum())   # chosen right, random wrong
        diff = s[3] - s[2]
        lo, hi = bootstrap_ci(diff, n_boot)
        out.append({"mode": mode, "n_cases": len(s), "acc_stage2": s[2].mean(), "acc_stage3": s[3].mean(),
                    "diff_3_minus_2": diff.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "majority_only_random_right": b, "majority_only_chosen_right": c,
                    "mcnemar_exact_p": stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0})
    return pd.DataFrame(out)


def field_tables(df):
    d = df[df.stage.isin([2, 3]) & df.field.notna()]
    choice = (d[d.stage == 3].groupby(["mode", "field"]).size().rename("n").reset_index())
    choice["share"] = choice.n / choice.groupby("mode").n.transform("sum")
    acc = (d.groupby(["mode", "stage", "field"])
           .agg(n=("correct", "size"), acc=("correct", "mean"), mean_confidence=("confidence", "mean"))
           .reset_index())
    return choice.sort_values(["mode", "n"], ascending=[True, False]), acc


def confidence_tables(df, maj):
    rel = (df[df.parse_ok].groupby(["mode", "stage", "confidence"])
           .agg(n=("correct", "size"), acc=("correct", "mean")).reset_index())
    out = []
    for (mode, stage), g in df[df.parse_ok].groupby(["mode", "stage"]):
        conf = g.confidence.astype(float)
        p = conf / 5.0   # map 1-5 to a probability for ECE (assumption: 5 = certain)
        ece = sum(len(gg) / len(g) * abs(gg.correct.mean() - (gg.confidence.mean() / 5.0))
                  for _, gg in g.groupby("confidence"))
        mg = maj[(maj["mode"] == mode) & (maj.stage == stage)].dropna(subset=["label_confidence", "mean_confidence"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, rho_p = (stats.spearmanr(mg.mean_confidence, mg.label_confidence) if len(mg) > 2 else (np.nan, np.nan))
        out.append({"mode": mode, "stage": stage, "mean_confidence": conf.mean(),
                    "conf_when_correct": conf[g.correct].mean(), "conf_when_wrong": conf[~g.correct].mean(),
                    "accuracy": g.correct.mean(), "mean_conf_as_prob": p.mean(), "ece": ece,
                    "auroc_conf_vs_correct": auroc(conf.values, g.correct.values),
                    "spearman_vs_derm_conf": rho, "spearman_p": rho_p,
                    "mean_derm_confidence": mg.label_confidence.mean()})
    return rel, pd.DataFrame(out)


def flip_table(df):
    base = df[df.stage == 1][["mode", "case_id", "sample", "pred", "confidence", "correct"]]
    merged = df[df.stage > 1].merge(base, on=["mode", "case_id", "sample"], suffixes=("", "_s1"))
    merged["flipped"] = merged.pred != merged.pred_s1
    merged["conf_change"] = merged.confidence - merged.confidence_s1
    return (merged.groupby(["mode", "stage"]).agg(
        n=("flipped", "size"), flip_rate=("flipped", "mean"),
        wrong_to_right=("correct", lambda s: (s & ~merged.loc[s.index, "correct_s1"]).mean()),
        right_to_wrong=("correct", lambda s: (~s & merged.loc[s.index, "correct_s1"]).mean()),
        mean_conf_change=("conf_change", "mean")).reset_index())


def uncertainty_table(maj):
    return (maj.groupby(["mode", "stage", "correct"])
            .agg(n_cases=("case_id", "size"), mean_agreement=("agreement", "mean"),
                 mean_entropy_bits=("entropy_bits", "mean"), mean_confidence=("mean_confidence", "mean"))
            .reset_index())


# ---- plots --------------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ae"); ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_all(acc, rel, choice, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": "#52514e", "text.color": "#0b0b0b"})

    # 1. Accuracy by stage, one bar per mode, 95% bootstrap CI.
    modes = [m for m in MODE_COLORS if m in set(acc["mode"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(modes)
    for i, mode in enumerate(modes):
        g = acc[acc["mode"] == mode].sort_values("stage")
        x = g.stage + (i - (len(modes) - 1) / 2) * width
        err = np.array([g.acc_exact - g.acc_exact_lo, g.acc_exact_hi - g.acc_exact]).clip(min=0)
        ax.bar(x, g.acc_exact, width * 0.92, color=MODE_COLORS.get(mode), label=mode,
               yerr=np.nan_to_num(err), error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 3})
    ax.set_xticks(list(STAGE_NAMES), [STAGE_NAMES[s] for s in STAGE_NAMES])
    ax.set_ylabel("accuracy (exact label)")
    ax.set_ylim(0, 1)
    ax.set_title("Diagnosis accuracy by information stage", loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(out_dir / "accuracy_by_stage.png", dpi=200); plt.close(fig)

    # 2. Reliability: accuracy at each self-reported confidence level, one panel per mode.
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        ax.plot([1, 5], [0.2, 1.0], color="#b5b4ae", linewidth=1, linestyle="--", label="confidence / 5")
        for stage, g in rel[rel["mode"] == mode].groupby("stage"):
            ax.plot(g.confidence, g.acc, color=STAGE_COLORS[stage], linewidth=2, marker="o", markersize=5,
                    label=STAGE_NAMES[stage])
        ax.set_title(mode, loc="left"); ax.set_xlabel("self-reported confidence"); ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1); _style(ax)
    axes[0][0].set_ylabel("accuracy")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "reliability.png", dpi=200); plt.close(fig)

    # 3. Which field the model asks for in stage 3.
    if len(choice):
        fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.5), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            g = choice[choice["mode"] == mode].sort_values("share")
            ax.barh(g.field, g.share, color="#2a78d6", height=0.6)
            for y, (v, n) in enumerate(zip(g.share, g.n)):
                ax.text(v + 0.01, y, f"{v:.0%} (n={n})", va="center", fontsize=8, color="#52514e")
            ax.set_xlim(0, 1.15); ax.set_title(f"Stage 3 requested field: {mode}", loc="left")
            ax.grid(axis="x", color="#e6e5e0"); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.tight_layout(); fig.savefig(out_dir / "stage3_field_choice.png", dpi=200); plt.close(fig)


# ---- main ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--labels", help="override labels.yaml (default: the one in the run config)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df, cfg = load_run(args.run_dir, args.labels)
    out_dir = Path(args.run_dir) / "analysis"
    out_dir.mkdir(exist_ok=True)
    target = cfg.get("target_label", "Eczema")

    maj = majority_votes(df)
    acc = accuracy_table(df, args.n_boot)
    choice, field_acc = field_tables(df)
    rel, conf = confidence_tables(df, maj)
    tables = {
        "rq1_accuracy": acc,
        "rq1_majority_vote": majority_table(maj, target),
        "rq2_stage2_vs_stage3": rq2_table(df, maj, args.n_boot),
        "rq2_stage3_field_choice": choice,
        "rq2_accuracy_by_field": field_acc,
        "rq3_confidence": conf,
        "rq3_reliability": rel,
        "change_vs_stage1": flip_table(df),
        "uncertainty_seed_agreement": uncertainty_table(maj),
    }
    unmatched = (df[df.pred == UNMATCHED].groupby("diagnosis").size().rename("count")
                 .sort_values(ascending=False).reset_index())

    df.drop(columns=["raw", "select_raw"]).to_csv(out_dir / "graded_answers.csv", index=False)
    maj.to_csv(out_dir / "majority_votes.csv", index=False)
    unmatched.to_csv(out_dir / "unmatched_diagnoses.csv", index=False)
    lines = [f"run: {args.run_dir}   model: {cfg['model']}   samples/case: {cfg['n_samples']}   "
             f"cases: {df.case_id.nunique()}   answers: {len(df)}", ""]
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        for name, t in tables.items():
            t.to_csv(out_dir / f"{name}.csv", index=False)
            lines += [f"== {name} ==", t.to_string(index=False) if len(t) else "(empty)", ""]
    lines.append(f"unmatched diagnoses: {len(unmatched)} distinct strings ({int(unmatched['count'].sum()) if len(unmatched) else 0} "
                 f"answers) -> unmatched_diagnoses.csv; add aliases to labels.yaml and re-run analyze.")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary)
    plot_all(acc, rel, choice, out_dir)
    print(summary)
    print(f"\nwrote tables, plots and summary.txt to {out_dir}")


if __name__ == "__main__":
    main()
