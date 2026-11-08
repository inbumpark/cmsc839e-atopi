"""Preprocessed case format: loading and validation.

Run as a CLI to check a dataset before an experiment:
    python -m derm_eval.schema data/cases.jsonl [--config configs/default.yaml]
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import load_config, resolve


@dataclass
class Case:
    case_id: str
    images: list            # absolute paths
    label: str              # dermatologist reference label (ground truth)
    label_confidence: float | None = None   # dermatologist confidence for `label`, 1-5
    differential: dict = field(default_factory=dict)  # label -> weight (optional)
    context: dict = field(default_factory=dict)       # field key -> human-readable value

    def available_fields(self, field_order):
        """Context fields (in config order) that this case actually has a value for."""
        return [k for k in field_order if _present(self.context.get(k))]


def _present(v):
    return v is not None and str(v).strip() not in ("", "nan", "None")


def load_cases(path, max_cases=None):
    path = Path(path)
    cases = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from None
            for key in ("case_id", "images", "label"):
                if key not in d:
                    raise ValueError(f"{path}:{line_no}: missing required key '{key}'")
            images = d["images"] if isinstance(d["images"], list) else [d["images"]]
            cases.append(Case(
                case_id=str(d["case_id"]),
                images=[str((path.parent / p).resolve()) for p in images],
                label=d["label"],
                label_confidence=d.get("label_confidence"),
                differential=d.get("differential") or {},
                context={k: v for k, v in (d.get("context") or {}).items() if _present(v)},
            ))
            if max_cases and len(cases) >= max_cases:
                break
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case_id values")
    return cases


def label_weight(case):
    """Dermatologist weight of the case's own label in its differential (0 if not listed)."""
    return float(case.differential.get(case.label, 0.0))


def select_cases(cases, cfg):
    """Cases a run uses: label weight >= min_label_weight (if set), then the first max_cases."""
    w = cfg.get("min_label_weight")
    if w is not None:
        cases = [c for c in cases if label_weight(c) >= float(w) - 1e-9]
    if cfg.get("max_cases"):
        cases = cases[: cfg["max_cases"]]
    return cases


def validate(cases, field_order, known_labels):
    """Return (errors, warnings) as lists of strings."""
    errors, warnings = [], []
    for c in cases:
        if not c.images:
            errors.append(f"{c.case_id}: no images")
        for p in c.images:
            if not Path(p).exists():
                errors.append(f"{c.case_id}: image not found: {p}")
        unknown = set(c.context) - set(field_order)
        if unknown:
            warnings.append(f"{c.case_id}: context fields not in config (ignored): {sorted(unknown)}")
        n = len(c.available_fields(field_order))
        if n < 2:
            warnings.append(f"{c.case_id}: only {n} usable context field(s); stages 2-4 barely differ")
        if c.label not in known_labels:
            warnings.append(f"{c.case_id}: label '{c.label}' has no aliases in labels.yaml")
        if c.label_confidence is not None and not (1 <= float(c.label_confidence) <= 5):
            warnings.append(f"{c.case_id}: label_confidence {c.label_confidence} outside 1-5")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_path")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    field_order = list(cfg["context_fields"])
    known_labels = set(yaml.safe_load(resolve(cfg["labels_path"]).read_text()))
    cases = load_cases(args.data_path)
    errors, warnings = validate(cases, field_order, known_labels)

    labels = {}
    for c in cases:
        labels[c.label] = labels.get(c.label, 0) + 1
    print(f"{len(cases)} cases")
    print("labels:", dict(sorted(labels.items(), key=lambda kv: -kv[1])))
    print("field coverage:")
    for k in field_order:
        n = sum(1 for c in cases if k in c.context)
        print(f"  {k:16s} {n}/{len(cases)}")
    for w in warnings:
        print("WARNING", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
