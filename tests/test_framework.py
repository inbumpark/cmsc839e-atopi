import json
from pathlib import Path

import pytest

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
