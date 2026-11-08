"""The four information stages, in independent and cascade modes.

Stage 1: image only
Stage 2: image + one context field picked at random (seeded) from the case's available fields
Stage 3: image + one context field the model picks from the same menu (two turns)
Stage 4: image + all available context fields

independent: every stage is a fresh conversation.
cascade:     stages 2-4 each branch from the stage-1 conversation (same sample/seed), and the
             model is asked to reconsider given the new information (1->2, 1->3, 1->4).
chain_124 / chain_134 (stage 4 only): continue the finished cascade 1->2 or 1->3 conversation
             (rebuilt exactly from its saved records) with the stage-4 turn giving all context,
             i.e. 1->2->4 and 1->3->4. The final turn is the same prompt as cascade 1->4.
"""
import random
import re

from . import prompts
from .client import GenerationAborted, parse_json
from .mcq import build_options, load_pool

SHARED = "shared"   # mode label for stage-1 records, which both modes use
CHAINS = {"chain_124": 2, "chain_134": 3}   # chain mode -> cascade stage it continues from


def _validate_diagnosis(obj):
    if not obj:
        return None
    diagnosis = str(obj.get("diagnosis") or "").strip()
    try:
        confidence = int(round(float(obj.get("confidence"))))
    except (TypeError, ValueError):
        return None
    if not diagnosis or not 1 <= confidence <= 5:
        return None
    return {"diagnosis": diagnosis, "reason": str(obj.get("reason") or "").strip(),
            "confidence": confidence}


def _validator_mcq(options):
    """Accepts "B", "B.", "(B)", "B. Psoriasis" or the option text; returns the chosen option."""
    letters = prompts.mcq_letters(len(options))
    by_name = {o.lower(): i for i, o in enumerate(options)}

    def validate(obj):
        if not obj:
            return None
        ans = str(obj.get("answer") or "").strip()
        m = re.fullmatch(r"\(?([A-Za-z])\)?(?:[.):]\s*.*)?", ans)
        if m and m.group(1).upper() in letters:
            idx = letters.index(m.group(1).upper())
        elif ans.lower() in by_name:
            idx = by_name[ans.lower()]
        else:
            return None
        try:
            confidence = int(round(float(obj.get("confidence"))))
        except (TypeError, ValueError):
            return None
        if not 1 <= confidence <= 5:
            return None
        return {"diagnosis": options[idx], "answer": letters[idx],
                "reason": str(obj.get("reason") or "").strip(), "confidence": confidence}
    return validate


def _validator_select(keys, field_names):
    lookup = {k.lower(): k for k in keys}
    lookup.update({field_names[k].lower(): k for k in keys})

    def validate(obj):
        if not obj:
            return None
        key = lookup.get(str(obj.get("requested_field") or "").strip().lower())
        return {"requested_field": key, "why": str(obj.get("why") or "").strip()} if key else None
    return validate


class StageRunner:
    def __init__(self, client, cfg):
        self.client = client
        self.field_names = cfg["context_fields"]
        self.field_order = list(self.field_names)
        self.base_seed = cfg["base_seed"]
        self.max_retries = cfg["max_retries"]
        self.images_per_case = cfg.get("images_per_case")
        self.mcq = cfg.get("answer_format", "free_text") == "mcq"
        if self.mcq:
            self.pool = load_pool(cfg)
            self.n_choices = cfg.get("mcq_choices", 4)
            self.mcq_seed = cfg.get("mcq_seed", 0)
            self.mcq_weighting = cfg.get("mcq_weighting", "uniform")
            self.mcq_distractors = cfg.get("mcq_distractors")
            self._options = {}

    # ---- helpers -------------------------------------------------------------------------
    def seed(self, k):
        return self.base_seed + k

    def random_field(self, case, k):
        """Stage-2 field: deterministic per (case, sample), shared by both modes."""
        fields = case.available_fields(self.field_order)
        return random.Random(f"{case.case_id}:{self.seed(k)}").choice(fields) if fields else None

    def options(self, case):
        """MCQ options for this case (same in every stage, mode and sample)."""
        if case.case_id not in self._options:
            self._options[case.case_id] = build_options(case.case_id, case.label, self.pool, self.n_choices,
                                                        self.mcq_seed, self.mcq_weighting,
                                                        fixed=self.mcq_distractors)
        return self._options[case.case_id]

    def _answer_spec(self, case):
        """(schema, validator) for the final answer of a stage."""
        if self.mcq:
            opts = self.options(case)
            return prompts.mcq_schema(opts), _validator_mcq(opts)
        return prompts.DIAGNOSIS_SCHEMA, _validate_diagnosis

    # Prompt builders that switch between free-text and MCQ wording.
    def _diagnosis_prompt(self, case, n, keys):
        if self.mcq:
            return prompts.mcq_diagnosis_prompt(n, case.context, keys, self.field_names, self.options(case))
        return prompts.diagnosis_prompt(n, case.context, keys, self.field_names)

    def _select_prompt(self, case, n, keys):
        if self.mcq:
            return prompts.mcq_select_prompt(n, keys, self.field_names, self.options(case))
        return prompts.select_prompt(n, keys, self.field_names)

    def _answer_to_request_prompt(self, case, key):
        if self.mcq:
            return prompts.mcq_answer_to_request_prompt(case.context, key, self.field_names, self.options(case))
        return prompts.answer_to_request_prompt(case.context, key, self.field_names)

    def _cascade_update_prompt(self, case, keys):
        if self.mcq:
            return prompts.mcq_cascade_update_prompt(case.context, keys, self.field_names, self.options(case))
        return prompts.cascade_update_prompt(case.context, keys, self.field_names)

    def _image_paths(self, case):
        return case.images[: self.images_per_case] if self.images_per_case else case.images

    def _first_user(self, case, text):
        return {"role": "user", "content": text, "images": self.client.images(self._image_paths(case))}

    def _ask(self, messages, schema, seed, validate):
        """Query with retries on unparseable output. Returns (parsed|None, raw, latency, attempts)."""
        latency, raw = 0.0, ""
        for attempt in range(self.max_retries + 1):
            try:
                raw, dt = self.client.chat(messages, schema, seed + 1000 * attempt)
            except GenerationAborted as e:          # retried with a new seed like a parse failure
                raw, dt = f"<generation aborted: {e}>", 0.0
            latency += dt
            parsed = validate(parse_json(raw))
            if parsed:
                return parsed, raw, latency, attempt + 1
        return None, raw, latency, self.max_retries + 1

    def _record(self, case, k, stage, mode, fields_given, parsed, raw, latency, attempts,
                select=None, error=None):
        rec = {
            "case_id": case.case_id, "mode": mode, "stage": stage, "sample": k, "seed": self.seed(k),
            "fields_given": fields_given,
            "requested_field": select["requested_field"] if select and select.get("parsed") else None,
            "requested_why": select["parsed"]["why"] if select and select.get("parsed") else None,
            "select_raw": select["raw"] if select else None,
            "diagnosis": parsed["diagnosis"] if parsed else None,
            "reason": parsed["reason"] if parsed else None,
            "confidence": parsed["confidence"] if parsed else None,
            "parse_ok": parsed is not None,
            "raw": raw, "latency_s": round(latency, 3), "attempts": attempts, "error": error,
        }
        if self.mcq:
            opts = self.options(case)
            rec.update(options=opts, answer=parsed["answer"] if parsed else None,
                       correct_answer=prompts.mcq_letters(len(opts))[opts.index(case.label)])
        return rec

    # ---- stages --------------------------------------------------------------------------
    def stage1_messages(self, case):
        n = len(self._image_paths(case))
        return [self._first_user(case, self._diagnosis_prompt(case, n, []))]

    def stage1(self, case, k):
        schema, validate = self._answer_spec(case)
        parsed, raw, lat, att = self._ask(self.stage1_messages(case), schema, self.seed(k), validate)
        return self._record(case, k, 1, SHARED, [], parsed, raw, lat, att)

    def cascade_messages(self, case, stage, stage1_record, record):
        """The full cascade 1->stage conversation that produced `record`, ending with its answer."""
        msgs = self.stage1_messages(case) + [{"role": "assistant", "content": stage1_record["raw"]}]
        if stage == 3:
            available = case.available_fields(self.field_order)
            msgs += [{"role": "user", "content": prompts.cascade_select_prompt(available, self.field_names)},
                     {"role": "assistant", "content": record["select_raw"]}]
        msgs += [{"role": "user", "content": self._cascade_update_prompt(case, record["fields_given"])},
                 {"role": "assistant", "content": record["raw"]}]
        return msgs

    def run(self, case, k, stage, mode, stage1_record=None, via_record=None):
        if stage == 1:
            return self.stage1(case, k)
        available = case.available_fields(self.field_order)
        if not available:
            return self._record(case, k, stage, mode, [], None, "", 0.0, 0, error="no context fields")
        if mode in ("cascade", *CHAINS) and stage1_record is None:
            raise ValueError(f"{mode} mode needs the stage-1 record for this sample")
        if mode in CHAINS:
            if stage != 4:
                raise ValueError(f"{mode} only has stage 4")
            if via_record is None:
                raise ValueError(f"{mode} needs the cascade stage-{CHAINS[mode]} record for this sample")
            if not (stage1_record["parse_ok"] and via_record["parse_ok"]):
                return self._record(case, k, stage, mode, [], None, "", 0.0, 0,
                                    error=f"cascade stage {CHAINS[mode]} has no answer")
            messages = self.cascade_messages(case, CHAINS[mode], stage1_record, via_record) + [
                {"role": "user", "content": self._cascade_update_prompt(case, available)}]
            schema, validate = self._answer_spec(case)
            parsed, raw, lat, att = self._ask(messages, schema, self.seed(k), validate)
            rec = self._record(case, k, stage, mode, available, parsed, raw, lat, att)
            rec["via_diagnosis"] = via_record["diagnosis"]
            return rec

        n = len(self._image_paths(case))
        seed = self.seed(k)
        if mode == "independent":
            history = []
        else:
            history = self.stage1_messages(case) + [{"role": "assistant", "content": stage1_record["raw"]}]

        def diagnose(messages, keys, select=None):
            schema, validate = self._answer_spec(case)
            parsed, raw, lat, att = self._ask(messages, schema, seed, validate)
            if select:
                lat += select["latency"]
                att += select["attempts"]
            return self._record(case, k, stage, mode, keys, parsed, raw, lat, att, select=select)

        if stage in (2, 4):
            keys = [self.random_field(case, k)] if stage == 2 else available
            if mode == "independent":
                messages = [self._first_user(case, self._diagnosis_prompt(case, n, keys))]
            else:
                messages = history + [{"role": "user", "content": self._cascade_update_prompt(case, keys)}]
            return diagnose(messages, keys)

        if stage == 3:
            if mode == "independent":
                ask = [self._first_user(case, self._select_prompt(case, n, available))]
            else:
                ask = history + [{"role": "user", "content": prompts.cascade_select_prompt(
                    available, self.field_names)}]
            parsed, raw, lat, att = self._ask(ask, prompts.select_schema(available), seed,
                                              _validator_select(available, self.field_names))
            select = {"parsed": parsed, "raw": raw, "latency": lat, "attempts": att,
                      "requested_field": parsed["requested_field"] if parsed else None}
            if not parsed:
                return self._record(case, k, stage, mode, [], None, "", lat, att, select=select,
                                    error="could not parse requested_field")
            key = parsed["requested_field"]
            if mode == "independent":
                follow = self._answer_to_request_prompt(case, key)
            else:
                follow = self._cascade_update_prompt(case, [key])
            messages = ask + [{"role": "assistant", "content": raw}, {"role": "user", "content": follow}]
            return diagnose(messages, [key], select=select)

        raise ValueError(f"unknown stage {stage}")
