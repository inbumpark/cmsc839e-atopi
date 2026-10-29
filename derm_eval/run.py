"""Run the experiment and append one JSON record per model answer to results/<run_name>/raw.jsonl.

    python -m derm_eval.run --config configs/default.yaml [--max-cases N] [--stages 1,2,3,4]
                            [--modes independent,cascade] [--n-samples K] [--run-name NAME]

Re-running the same command resumes: completed (case, mode, stage, sample) entries are skipped.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from .client import OllamaClient
from .config import load_config, resolve
from .schema import load_cases
from .stages import SHARED, StageRunner


def record_key(r):
    return (r["case_id"], r["mode"], r["stage"], r["sample"])


def load_done(raw_path):
    done = {}
    if raw_path.exists():
        for line in raw_path.open():
            if line.strip():
                r = json.loads(line)
                done[record_key(r)] = r
    return done


def plan_tasks(cases, cfg):
    """Ordered list of (case, sample, stage, mode). Stage 1 is shared by both modes."""
    stages, modes = cfg["stages"], cfg["modes"]
    need_stage1 = 1 in stages or "cascade" in modes
    tasks = []
    for case in cases:
        for k in range(cfg["n_samples"]):
            if need_stage1:
                tasks.append((case, k, 1, SHARED))
            for mode in modes:
                for stage in stages:
                    if stage != 1:
                        tasks.append((case, k, stage, mode))
    return tasks


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run-name")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--n-samples", type=int)
    ap.add_argument("--stages", help="comma-separated, e.g. 1,2,3,4")
    ap.add_argument("--modes", help="comma-separated: independent,cascade")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.max_cases is not None:
        cfg["max_cases"] = args.max_cases
    if args.n_samples is not None:
        cfg["n_samples"] = args.n_samples
    if args.stages:
        cfg["stages"] = [int(s) for s in args.stages.split(",")]
    if args.modes:
        cfg["modes"] = args.modes.split(",")
    for m in cfg["modes"]:
        if m not in ("independent", "cascade"):
            raise SystemExit(f"unknown mode '{m}'")

    cases = load_cases(resolve(cfg["data_path"]), cfg.get("max_cases"))
    out_dir = resolve(cfg["results_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    cfg_path = out_dir / "config.yaml"
    if cfg_path.exists():
        prev = yaml.safe_load(cfg_path.read_text())
        for key in ("model", "temperature", "base_seed", "think", "image_max_side", "images_per_case"):
            if prev.get(key) != cfg.get(key):
                raise SystemExit(f"{out_dir} was run with {key}={prev.get(key)!r}, now {cfg.get(key)!r}. "
                                 "Use a different --run-name to avoid mixing results.")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    client = OllamaClient(cfg)
    client.check()
    runner = StageRunner(client, cfg)

    done = load_done(raw_path)
    tasks = plan_tasks(cases, cfg)
    todo = [t for t in tasks if (t[0].case_id, t[3], t[2], t[1]) not in done]
    print(f"{len(cases)} cases, {len(tasks)} queries planned, {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to run -> {raw_path}")

    t_start, n_fail = time.time(), 0
    with raw_path.open("a") as f:
        for i, (case, k, stage, mode) in enumerate(todo, 1):
            stage1 = done.get((case.case_id, SHARED, 1, k)) if mode == "cascade" else None
            rec = runner.run(case, k, stage, mode, stage1_record=stage1)
            rec.update(run_name=cfg["run_name"], model=cfg["model"], timestamp=datetime.now().isoformat())
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[record_key(rec)] = rec
            n_fail += not rec["parse_ok"]
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(todo) - i)
            ans = f"{rec['diagnosis']} ({rec['confidence']})" if rec["parse_ok"] else f"FAILED {rec['error'] or ''}"
            extra = f" asked={rec['requested_field']}" if stage == 3 else ""
            print(f"[{i}/{len(todo)}] case={case.case_id} s{k} stage{stage} {mode:11s} "
                  f"{rec['latency_s']:5.1f}s{extra} -> {ans}  | ETA {fmt_time(eta)}", flush=True)

    print(f"done in {fmt_time(time.time() - t_start)}; {n_fail} unparsed answers. "
          f"Analyze with: python -m derm_eval.analyze {out_dir}")


if __name__ == "__main__":
    main()
