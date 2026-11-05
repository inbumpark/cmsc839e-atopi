import json
from pathlib import Path

import pytest
import yaml

from derm_eval import prompts
from derm_eval.client import parse_json
from derm_eval.config import load_config, resolve
from derm_eval.normalize import UNMATCHED, Normalizer
from derm_eval.run import plan_tasks
from derm_eval.schema import Case, load_cases, validate
from derm_eval.stages import SHARED, StageRunner

CFG = load_config(resolve("configs/default.yaml"))
FIELDS = CFG["context_fields"]


# ---- normalize -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def norm():
    return Normalizer(resolve("configs/labels.yaml"), extra_labels={"Some Rare Label"})


@pytest.mark.parametrize("text,label", [
    ("Atopic Dermatitis", "Eczema"),
    ("eczema", "Eczema"),
    ("Likely nummular eczema (discoid)", "Eczema"),
    ("Seborrheic dermatitis", "Seborrheic Dermatitis"),
    ("allergic contact dermatitis", "Allergic Contact Dermatitis"),
    ("contact dermatitis", "CD - Contact dermatitis"),
    ("Hives (acute urticaria)", "Urticaria"),
    ("tinea corporis", "Tinea"),
    ("psoriasis vs eczema", "Psoriasis"),          # earliest mention wins
    ("Psoriasiss", "Psoriasis"),                   # fuzzy
    ("some rare label", "Some Rare Label"),        # labels from the data are always matchable
    ("Erythema", UNMATCHED),
    ("", UNMATCHED),
])
def test_normalize(norm, text, label):
    assert norm(text) == label


def test_parse_json():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json here") is None


# ---- schema --------------------------------------------------------------------------------
def write_cases(tmp_path, rows):
    (tmp_path / "img.png").write_bytes(b"x")
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_load_and_validate(tmp_path):
    p = write_cases(tmp_path, [
        {"case_id": 1, "images": ["img.png"], "label": "Eczema", "label_confidence": 4,
         "context": {"duration": "1 week", "texture": "", "bogus": "x"}},
        {"case_id": 2, "images": ["missing.png"], "label": "Nope", "context": {}},
    ])
    cases = load_cases(p)
    assert cases[0].case_id == "1" and Path(cases[0].images[0]).is_absolute()
    assert cases[0].available_fields(list(FIELDS)) == ["duration"]     # empty values dropped
    errors, warnings = validate(cases, list(FIELDS), {"Eczema"})
    assert any("missing.png" in e for e in errors)
    assert any("bogus" in w for w in warnings)
    assert any("'Nope'" in w for w in warnings)


def test_duplicate_ids(tmp_path):
    p = write_cases(tmp_path, [{"case_id": "a", "images": ["img.png"], "label": "x"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)


# ---- stages with a mock client -------------------------------------------------------------
class MockClient:
    def __init__(self, bad_first=False):
        self.calls = []
        self.bad_first = bad_first

    def images(self, paths):
        return ["b64"] * len(paths)

    def chat(self, messages, schema, seed):
        self.calls.append({"messages": messages, "schema": schema, "seed": seed})
        if self.bad_first and len(self.calls) == 1:
            return "not json", 0.1
        if "answer" in schema["properties"]:                       # MCQ: always pick option B
            return json.dumps({"answer": "B", "reason": "looks like it", "confidence": 3}), 0.1
        if "requested_field" in schema["properties"]:
            return json.dumps({"requested_field": schema["properties"]["requested_field"]["enum"][-1],
                               "why": "useful"}), 0.1
        return json.dumps({"diagnosis": "eczema", "reason": "itchy", "confidence": 4}), 0.1


CASE = Case(case_id="c1", images=["/x.png"], label="Eczema", label_confidence=4,
            context={"duration": "2 weeks", "body_location": "elbow", "symptoms": "itching"})


def runner(client):
    return StageRunner(client, {**CFG, "max_retries": 2})


def test_stage1_image_only():
    c = MockClient()
    rec = runner(c).run(CASE, 0, 1, SHARED)
    msg = c.calls[0]["messages"]
    assert len(msg) == 1 and msg[0]["images"] == ["b64"]
    assert "Patient context" not in msg[0]["content"] and "Here is a photo from" in msg[0]["content"]
    assert rec["parse_ok"] and rec["fields_given"] == [] and rec["confidence"] == 4


def test_stage2_random_field_deterministic_and_shared_across_modes():
    r = runner(MockClient())
    picks = {r.random_field(CASE, k) for k in range(20)}
    assert picks <= set(CASE.context) and len(picks) > 1
    s1 = r.run(CASE, 3, 1, SHARED)
    ind = r.run(CASE, 3, 2, "independent")
    cas = r.run(CASE, 3, 2, "cascade", stage1_record=s1)
    assert ind["fields_given"] == cas["fields_given"] == [r.random_field(CASE, 3)]


def test_stage3_independent_two_turns():
    c = MockClient()
    rec = runner(c).run(CASE, 0, 3, "independent")
    assert len(c.calls) == 2
    assert c.calls[0]["schema"]["properties"]["requested_field"]["enum"] == ["duration", "body_location", "symptoms"]
    final = c.calls[1]["messages"]
    assert [m["role"] for m in final] == ["user", "assistant", "user"]
    assert "itching" in final[-1]["content"]
    assert rec["requested_field"] == "symptoms" and rec["fields_given"] == ["symptoms"]


def test_stage4_all_fields():
    c = MockClient()
    rec = runner(c).run(CASE, 0, 4, "independent")
    assert rec["fields_given"] == ["duration", "body_location", "symptoms"]
    content = c.calls[0]["messages"][0]["content"]
    assert all(v in content for v in CASE.context.values())


def test_cascade_branches_from_stage1():
    c = MockClient()
    r = runner(c)
    s1 = r.run(CASE, 0, 1, SHARED)
    r.run(CASE, 0, 3, "cascade", stage1_record=s1)
    final = c.calls[-1]["messages"]
    assert [m["role"] for m in final] == ["user", "assistant", "user", "assistant", "user"]
    assert final[1]["content"] == s1["raw"]
    with pytest.raises(ValueError):
        r.run(CASE, 0, 2, "cascade")


def test_retry_on_bad_json():
    c = MockClient(bad_first=True)
    rec = runner(c).run(CASE, 0, 1, SHARED)
    assert rec["parse_ok"] and rec["attempts"] == 2
    assert c.calls[1]["seed"] != c.calls[0]["seed"]


def test_no_context_fields():
    case = Case(case_id="c2", images=["/x.png"], label="Eczema")
    rec = runner(MockClient()).run(case, 0, 2, "independent")
    assert not rec["parse_ok"] and rec["error"] == "no context fields"


def test_plan_tasks_counts():
    cfg = {**CFG, "n_samples": 2}
    tasks = plan_tasks([CASE], cfg)
    assert len(tasks) == 2 * (1 + 2 * 3)       # per sample: shared stage 1 + 2 modes x stages 2-4
    assert tasks[0][2] == 1 and tasks[0][3] == SHARED


def test_prompt_matches_proposal():
    text = prompts.diagnosis_prompt(1, {"duration": "x"}, ["duration"], FIELDS)
    assert "You are a skilled dermatologist with multiple years of experience" in text
    assert "Here is a photo and some context from a patient" in text
    assert "confidence score of the diagnosis on a scale of 1 to 5" in text


# ---- sharding + merge ----------------------------------------------------------------------
from derm_eval.merge import merge
from derm_eval.run import parse_shard, shard_cases, shard_dir


def test_shard_cases_partition():
    cases = list(range(10))
    shards = [shard_cases(cases, i, 3) for i in range(3)]
    assert sorted(sum(shards, [])) == cases                  # disjoint and complete
    assert parse_shard("2/3") == (2, 3)
    for bad in ("3/3", "x", "1"):
        with pytest.raises(SystemExit):
            parse_shard(bad)


def _fake_split_run(tmp_path, n, drop=0, **overrides):
    """Write a run split into n shards with mock records; drop the last `drop` records of shard 0."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = write_cases(tmp_path, [{"case_id": f"c{j}", "images": ["img.png"], "label": "Eczema",
                                   "context": {"duration": "1 week"}} for j in range(5)])
    cfg = {**CFG, "data_path": str(data), "max_cases": None, "n_samples": 2, "ollama_host": "x"}
    cases = load_cases(data)
    run_dir = tmp_path / "results" / "split"
    for i in range(n):
        d = shard_dir(run_dir, i, n)
        d.mkdir(parents=True)
        shard_cfg = {**cfg, "shard": [i, n], **(overrides if i == 1 else {})}
        (d / "config.yaml").write_text(yaml.safe_dump(shard_cfg))
        recs = [{"case_id": c.case_id, "mode": m, "stage": s, "sample": k, "parse_ok": True}
                for c, k, s, m in plan_tasks(shard_cases(cases, i, n), cfg)]
        if i == 0 and drop:
            recs = recs[:-drop]
        (d / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return run_dir, len(plan_tasks(cases, cfg))


def test_merge_complete(tmp_path):
    run_dir, planned = _fake_split_run(tmp_path, 3)
    assert merge(run_dir) == (planned, 0)
    assert merge(run_dir) == (planned, 0)                     # idempotent
    merged_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert merged_cfg["run_name"] == "split" and "shard" not in merged_cfg


def test_merge_refuses_incomplete_or_mismatched(tmp_path):
    run_dir, planned = _fake_split_run(tmp_path / "a", 2, drop=3)
    with pytest.raises(SystemExit, match="3 planned queries missing"):
        merge(run_dir)
    assert merge(run_dir, allow_incomplete=True) == (planned - 3, 3)
    run_dir, _ = _fake_split_run(tmp_path / "b", 2, temperature=0.1)
    with pytest.raises(SystemExit, match="temperature"):
        merge(run_dir)


# ---- multiple choice -----------------------------------------------------------------------
from derm_eval.mcq import build_options, load_pool
from derm_eval.stages import _validator_mcq

MCQ_CFG = load_config(resolve("configs/mcq.yaml"))
POOL = load_pool(MCQ_CFG)


def test_mcq_pool_has_no_eczema_synonyms():
    norm = Normalizer(resolve("configs/labels.yaml"))
    assert "Eczema" not in POOL
    assert not [name for name in POOL if norm(name) == "Eczema" or "eczem" in name.lower()]


@pytest.mark.parametrize("weighting", ["uniform", "frequency"])
def test_build_options(weighting):
    opts = build_options("c1", "Eczema", POOL, 4, seed=0, weighting=weighting)
    assert len(opts) == 4 and len(set(opts)) == 4 and opts.count("Eczema") == 1
    assert opts == build_options("c1", "Eczema", POOL, 4, seed=0, weighting=weighting)   # deterministic
    positions = {build_options(f"c{i}", "Eczema", POOL, 4, 0, weighting).index("Eczema") for i in range(40)}
    assert positions == {0, 1, 2, 3}                                    # shuffled across cases


def test_mcq_validator():
    v = _validator_mcq(["Psoriasis", "Eczema", "Tinea", "Acne"])
    for ans in ("B", "b", "B.", "(B)", "B. Eczema", "Eczema"):
        assert v({"answer": ans, "reason": "r", "confidence": 4})["diagnosis"] == "Eczema", ans
    assert v({"answer": "E", "reason": "r", "confidence": 4}) is None
    assert v({"answer": "B", "reason": "r", "confidence": 9}) is None


def test_mcq_stages_use_same_options_everywhere():
    r = StageRunner(MockClient(), {**MCQ_CFG, "max_retries": 0})
    opts = r.options(CASE)
    s1 = r.run(CASE, 0, 1, SHARED)
    assert s1["options"] == opts and s1["answer"] == "B" and s1["diagnosis"] == opts[1]
    assert s1["correct_answer"] == "ABCD"[opts.index("Eczema")]
    for stage in (2, 3, 4):
        for mode in ("independent", "cascade"):
            client = MockClient()
            rec = StageRunner(client, {**MCQ_CFG, "max_retries": 0}).run(CASE, 0, stage, mode, stage1_record=s1)
            assert rec["parse_ok"] and rec["options"] == opts
            last = client.calls[-1]["messages"][-1]["content"]
            assert all(f"{l}. {o}" in last for l, o in zip("ABCD", opts))      # options re-listed each turn


def test_generation_aborted_is_retried_then_recorded():
    from derm_eval.client import GenerationAborted

    class Looping(MockClient):
        def __init__(self, fail_times):
            super().__init__()
            self.fail_times = fail_times

        def chat(self, messages, schema, seed):
            if len(self.calls) < self.fail_times:
                self.calls.append({"seed": seed})
                raise GenerationAborted("prediction aborted, token repeat limit reached")
            return super().chat(messages, schema, seed)

    c = Looping(fail_times=1)                  # first try loops, retry succeeds
    rec = runner(c).run(CASE, 0, 1, SHARED)
    assert rec["parse_ok"] and rec["attempts"] == 2
    assert c.calls[0]["seed"] != c.calls[1]["seed"]
    c = Looping(fail_times=99)                 # every try loops: recorded as a parse failure, no crash
    rec = runner(c).run(CASE, 0, 1, SHARED)
    assert not rec["parse_ok"] and rec["attempts"] == 3 and "generation aborted" in rec["raw"]


# ---- chains 1->2->4 and 1->3->4 --------------------------------------------------------------
@pytest.mark.parametrize("cfg", [CFG, MCQ_CFG], ids=["free_text", "mcq"])
@pytest.mark.parametrize("mode,via", [("chain_124", 2), ("chain_134", 3)])
def test_chain_continues_exact_cascade_conversation(cfg, mode, via):
    from derm_eval.stages import CHAINS
    assert CHAINS[mode] == via
    cfg = {**cfg, "max_retries": 0}
    s1 = StageRunner(MockClient(), cfg).run(CASE, 0, 1, SHARED)
    c_via = MockClient()
    via_rec = StageRunner(c_via, cfg).run(CASE, 0, via, "cascade", stage1_record=s1)
    sent_via = c_via.calls[-1]["messages"] + [{"role": "assistant", "content": via_rec["raw"]}]

    c4, c_chain = MockClient(), MockClient()
    direct = StageRunner(c4, cfg).run(CASE, 0, 4, "cascade", stage1_record=s1)
    rec = StageRunner(c_chain, cfg).run(CASE, 0, 4, mode, stage1_record=s1, via_record=via_rec)
    assert rec["parse_ok"] and rec["mode"] == mode and rec["stage"] == 4
    assert rec["fields_given"] == direct["fields_given"] == CASE.available_fields(list(cfg["context_fields"]))
    msgs = c_chain.calls[-1]["messages"]
    assert len(c_chain.calls) == 1                                  # only the final turn is queried
    assert msgs[:-1] == sent_via                                    # rebuilt 1->via conversation is exact
    assert msgs[-1] == c4.calls[-1]["messages"][-1]                 # same stage-4 turn as cascade 1->4
    assert c_chain.calls[-1]["seed"] == c4.calls[-1]["seed"]


def test_chain_skips_failed_intermediate_and_no_context():
    s1 = runner(MockClient()).run(CASE, 0, 1, SHARED)
    bad = {**s1, "stage": 2, "mode": "cascade", "parse_ok": False}
    c = MockClient()
    rec = runner(c).run(CASE, 0, 4, "chain_124", stage1_record=s1, via_record=bad)
    assert not rec["parse_ok"] and "no answer" in rec["error"] and not c.calls
    empty = Case(case_id="c2", images=["/x.png"], label="Eczema", label_confidence=4, context={})
    assert runner(c).run(empty, 0, 4, "chain_134")["error"] == "no context fields"


def test_plan_tasks_chains_are_stage4_only():
    cfg = {**CFG, "stages": [2, 3, 4], "modes": ["chain_124", "chain_134"], "n_samples": 2}
    tasks = plan_tasks([CASE], cfg)
    assert sorted((s, m) for _, _, s, m in tasks) == sorted([(4, "chain_124"), (4, "chain_134")] * 2)
