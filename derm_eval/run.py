"""Run the experiment and append one JSON record per model answer to results/<run_name>/raw.jsonl.

    python -m derm_eval.run --config configs/default.yaml [--max-cases N] [--stages 1,2,3,4]
                            [--modes independent,cascade] [--n-samples K] [--run-name NAME]
                            [--shard I/N] [--ollama-host URL] [--history-from RUN_DIR]

Re-running the same command resumes: completed (case, mode, stage, sample) entries are skipped.

--shard I/N runs only every N-th case starting at case I (0-based) and writes to
results/<run_name>/shards/<I>of<N>/, so N processes (e.g. one per GPU, each with its own Ollama
server via --ollama-host) can split a run. Combine them with: python -m derm_eval.merge results/<run_name>

Chains 1->2->4 and 1->3->4 (modes chain_124, chain_134; stage 4 only) continue the cascade
conversations of a finished run, read from --history-from (its merged raw.jsonl), e.g.
    python -m derm_eval.run --config configs/mcq.yaml --run-name mcq_chains --stages 4 \
        --modes chain_124,chain_134 --history-from results/mcq
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
from .stages import CHAINS, SHARED, StageRunner


# Settings that must not change when resuming a run (absent keys fall back to KEY_DEFAULTS, so
# runs made before a setting existed still resume).
RESUME_MUST_MATCH = ("model", "temperature", "base_seed", "think", "image_max_side", "images_per_case",
                     "answer_format", "mcq_choices", "mcq_seed", "mcq_weighting", "mcq_pool_path")
KEY_DEFAULTS = {"answer_format": "free_text"}


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
                    if stage != 1 and (mode not in CHAINS or stage == 4):
                        tasks.append((case, k, stage, mode))
    return tasks


def load_history(cfg, todo):
    """Records of the run that chain modes continue; checks it used the same model and settings."""
    src = resolve(cfg["history_from"])
    if not (src / "raw.jsonl").exists():
        raise SystemExit(f"--history-from {src}: no raw.jsonl (merge that run first)")
    prev = yaml.safe_load((src / "config.yaml").read_text())
    for key in RESUME_MUST_MATCH + ("num_ctx", "context_fields", "data_path"):
        if prev.get(key, KEY_DEFAULTS.get(key)) != cfg.get(key, KEY_DEFAULTS.get(key)):
            raise SystemExit(f"--history-from {src} was run with {key}={prev.get(key)!r}, now {cfg.get(key)!r}")
    history = load_done(src / "raw.jsonl")
    need = {(c.case_id, SHARED, 1, k) for c, k, _, m in todo if m in CHAINS} | \
           {(c.case_id, "cascade", CHAINS[m], k) for c, k, _, m in todo if m in CHAINS}
    missing = need - history.keys()
    if missing:
        raise SystemExit(f"--history-from {src} lacks {len(missing)} records chains need, e.g. {sorted(missing)[0]}")
    return history


def parse_shard(spec):
    """'I/N' -> (I, N) with 0 <= I < N."""
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like I/N (e.g. 0/4), got {spec!r}")
    if not 0 <= i < n:
        raise SystemExit(f"--shard {spec}: need 0 <= I < N")
    return i, n


def shard_cases(cases, i, n):
    """Round-robin split, so each shard gets a similar mix of cases."""
    return cases[i::n]


def shard_dir(run_dir, i, n):
    return Path(run_dir) / "shards" / f"{i}of{n}"


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
    ap.add_argument("--shard", help="I/N: run only shard I of N (0-based), see module docstring")
    ap.add_argument("--ollama-host", help="override ollama_host from the config")
    ap.add_argument("--history-from", help="finished run dir whose stage-1/cascade records chain modes continue")
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
    if args.ollama_host:
        cfg["ollama_host"] = args.ollama_host
    if args.history_from:
        cfg["history_from"] = args.history_from
    for m in cfg["modes"]:
        if m not in ("independent", "cascade", *CHAINS):
            raise SystemExit(f"unknown mode '{m}'")
    chains = [m for m in cfg["modes"] if m in CHAINS]
    if chains and not cfg.get("history_from"):
        raise SystemExit(f"modes {chains} continue an earlier run's cascades; pass --history-from RUN_DIR")

    cases = load_cases(resolve(cfg["data_path"]), cfg.get("max_cases"))
    out_dir = resolve(cfg["results_dir"]) / cfg["run_name"]
    if args.shard:
        i, n = parse_shard(args.shard)
        cases = shard_cases(cases, i, n)
        cfg["shard"] = [i, n]
        out_dir = shard_dir(out_dir, i, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    cfg_path = out_dir / "config.yaml"
    if cfg_path.exists():
        prev = yaml.safe_load(cfg_path.read_text())
        for key in RESUME_MUST_MATCH:
            if prev.get(key, KEY_DEFAULTS.get(key)) != cfg.get(key, KEY_DEFAULTS.get(key)):
                raise SystemExit(f"{out_dir} was run with {key}={prev.get(key)!r}, now {cfg.get(key)!r}. "
                                 "Use a different --run-name to avoid mixing results.")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    client = OllamaClient(cfg)
    client.check()
    runner = StageRunner(client, cfg)

    done = load_done(raw_path)
    tasks = plan_tasks(cases, cfg)
    todo = [t for t in tasks if (t[0].case_id, t[3], t[2], t[1]) not in done]
    history = load_history(cfg, todo) if chains else {}
    print(f"{len(cases)} cases, {len(tasks)} queries planned, {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to run -> {raw_path}")

    t_start, n_fail = time.time(), 0
    with raw_path.open("a") as f:
        for i, (case, k, stage, mode) in enumerate(todo, 1):
            if mode in CHAINS:
                stage1 = history[(case.case_id, SHARED, 1, k)]
                via = history[(case.case_id, "cascade", CHAINS[mode], k)]
                rec = runner.run(case, k, stage, mode, stage1_record=stage1, via_record=via)
            else:
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
            extra += f" via={rec.get('via_diagnosis')}" if mode in CHAINS and rec["parse_ok"] else ""
            print(f"[{i}/{len(todo)}] case={case.case_id} s{k} stage{stage} {mode:11s} "
                  f"{rec['latency_s']:5.1f}s{extra} -> {ans}  | ETA {fmt_time(eta)}", flush=True)

    print(f"done in {fmt_time(time.time() - t_start)}; {n_fail} unparsed answers. "
          f"Analyze with: python -m derm_eval.analyze {out_dir}")


if __name__ == "__main__":
    main()
