"""Merge the shards of a split run into results/<run_name>/raw.jsonl and config.yaml.

    python -m derm_eval.merge results/<run_name> [--allow-incomplete]

Reads every results/<run_name>/shards/<I>of<N>/ written by `derm_eval.run --shard I/N`, checks
that all N shards were run with the same settings and that every planned query is present, then
writes the combined raw.jsonl (plus any records already in it) for `derm_eval.analyze`.
Safe to re-run: records are de-duplicated by (case, mode, stage, sample).
"""
import argparse
import json
import re
from pathlib import Path

import yaml

from .config import resolve
from .run import load_done, plan_tasks, record_key, shard_cases
from .schema import load_cases

# Settings that must match across shards for their answers to be comparable.
MUST_MATCH = ("model", "temperature", "base_seed", "think", "image_max_side", "images_per_case",
              "num_ctx", "n_samples", "stages", "modes", "data_path", "max_cases", "context_fields",
              "answer_format", "mcq_choices", "mcq_seed", "mcq_weighting", "mcq_pool_path", "history_from")


def find_shards(run_dir):
    """{i: path} for run_dir/shards/<i>of<n>/, plus n. Errors if shards disagree on n or some are missing."""
    found = {}
    for p in sorted((Path(run_dir) / "shards").glob("*of*")):
        m = re.fullmatch(r"(\d+)of(\d+)", p.name)
        if m and (p / "config.yaml").exists():
            found[(int(m.group(1)), int(m.group(2)))] = p
    if not found:
        raise SystemExit(f"no shards found under {Path(run_dir) / 'shards'}")
    ns = {n for _, n in found}
    if len(ns) > 1:
        raise SystemExit(f"shards were split different ways ({sorted(ns)}); keep one split per run name")
    n = ns.pop()
    missing = sorted(set(range(n)) - {i for i, _ in found})
    if missing:
        raise SystemExit(f"missing shard(s) {missing} of {n}")
    return {i: p for (i, _), p in found.items()}, n


def merge(run_dir, allow_incomplete=False):
    run_dir = Path(run_dir)
    shards, n = find_shards(run_dir)
    cfgs = {i: yaml.safe_load((p / "config.yaml").read_text()) for i, p in shards.items()}
    base = cfgs[0]
    for i, cfg in cfgs.items():
        diff = [k for k in MUST_MATCH if cfg.get(k) != base.get(k)]
        if diff:
            raise SystemExit(f"shard {i} differs from shard 0 in {diff}; refusing to mix results")

    cases = load_cases(resolve(base["data_path"]), base.get("max_cases"))
    raw_path = run_dir / "raw.jsonl"
    merged = load_done(raw_path)          # keep anything already merged or run unsharded
    n_missing = 0
    for i, p in sorted(shards.items()):
        done = load_done(p / "raw.jsonl")
        planned = {(c.case_id, mode, stage, k) for c, k, stage, mode in plan_tasks(shard_cases(cases, i, n), base)}
        missing = planned - done.keys()
        n_missing += len(missing)
        print(f"shard {i}of{n}: {len(done)} records, {len(missing)} of {len(planned)} planned queries missing")
        for key, rec in done.items():
            merged.setdefault(key, rec)
    if n_missing and not allow_incomplete:
        raise SystemExit(f"{n_missing} planned queries missing; resume the unfinished shard(s) "
                         "or pass --allow-incomplete")

    cfg = {k: v for k, v in base.items() if k not in ("shard", "ollama_host")}
    cfg["run_name"] = run_dir.name
    cfg["merged_from_shards"] = n
    tmp = raw_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for rec in merged.values():
            f.write(json.dumps(rec) + "\n")
    tmp.replace(raw_path)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {len(merged)} records to {raw_path}. Analyze with: python -m derm_eval.analyze {run_dir}")
    return len(merged), n_missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--allow-incomplete", action="store_true", help="merge even if some queries are missing")
    args = ap.parse_args()
    merge(resolve(args.run_dir), args.allow_incomplete)


if __name__ == "__main__":
    main()
