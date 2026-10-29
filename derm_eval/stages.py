"""The four information stages, in independent and cascade modes.

Stage 1: image only
Stage 2: image + one context field picked at random (seeded) from the case's available fields
Stage 3: image + one context field the model picks from the same menu (two turns)
Stage 4: image + all available context fields

independent: every stage is a fresh conversation.
cascade:     stages 2-4 each branch from the stage-1 conversation (same sample/seed), and the
             model is asked to reconsider given the new information.
"""
import random

from . import prompts
from .client import parse_json

SHARED = "shared"   # mode label for stage-1 records, which both modes use


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

    # ---- helpers -------------------------------------------------------------------------
    def seed(self, k):
        return self.base_seed + k

    def random_field(self, case, k):
        """Stage-2 field: deterministic per (case, sample), shared by both modes."""
        fields = case.available_fields(self.field_order)
        return random.Random(f"{case.case_id}:{self.seed(k)}").choice(fields) if fields else None

    def _image_paths(self, case):
        return case.images[: self.images_per_case] if self.images_per_case else case.images

    def _first_user(self, case, text):
        return {"role": "user", "content": text, "images": self.client.images(self._image_paths(case))}

    def _ask(self, messages, schema, seed, validate):
        """Query with retries on unparseable output. Returns (parsed|None, raw, latency, attempts)."""
        latency, raw = 0.0, ""
        for attempt in range(self.max_retries + 1):
            raw, dt = self.client.chat(messages, schema, seed + 1000 * attempt)
            latency += dt
            parsed = validate(parse_json(raw))
            if parsed:
                return parsed, raw, latency, attempt + 1
        return None, raw, latency, self.max_retries + 1

    def _record(self, case, k, stage, mode, fields_given, parsed, raw, latency, attempts,
                select=None, error=None):
        return {
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

    # ---- stages --------------------------------------------------------------------------
    def stage1_messages(self, case):
        n = len(self._image_paths(case))
        return [self._first_user(case, prompts.diagnosis_prompt(n, case.context, [], self.field_names))]

    def stage1(self, case, k):
        parsed, raw, lat, att = self._ask(self.stage1_messages(case), prompts.DIAGNOSIS_SCHEMA,
                                          self.seed(k), _validate_diagnosis)
        return self._record(case, k, 1, SHARED, [], parsed, raw, lat, att)

    def run(self, case, k, stage, mode, stage1_record=None):
        if stage == 1:
            return self.stage1(case, k)
        available = case.available_fields(self.field_order)
        if not available:
            return self._record(case, k, stage, mode, [], None, "", 0.0, 0, error="no context fields")
        if mode == "cascade" and stage1_record is None:
            raise ValueError("cascade mode needs the stage-1 record for this sample")

        n = len(self._image_paths(case))
        seed = self.seed(k)
        if mode == "independent":
            history = []
        else:
            history = self.stage1_messages(case) + [{"role": "assistant", "content": stage1_record["raw"]}]

        def diagnose(messages, keys, select=None):
            parsed, raw, lat, att = self._ask(messages, prompts.DIAGNOSIS_SCHEMA, seed, _validate_diagnosis)
            if select:
                lat += select["latency"]
                att += select["attempts"]
            return self._record(case, k, stage, mode, keys, parsed, raw, lat, att, select=select)

        if stage in (2, 4):
            keys = [self.random_field(case, k)] if stage == 2 else available
            if mode == "independent":
                text = prompts.diagnosis_prompt(n, case.context, keys, self.field_names)
                messages = [self._first_user(case, text)]
            else:
                messages = history + [{"role": "user", "content": prompts.cascade_update_prompt(
                    case.context, keys, self.field_names)}]
            return diagnose(messages, keys)

        if stage == 3:
            if mode == "independent":
                ask = [self._first_user(case, prompts.select_prompt(n, available, self.field_names))]
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
                follow = prompts.answer_to_request_prompt(case.context, key, self.field_names)
            else:
                follow = prompts.cascade_update_prompt(case.context, [key], self.field_names)
            messages = ask + [{"role": "assistant", "content": raw}, {"role": "user", "content": follow}]
            return diagnose(messages, [key], select=select)

        raise ValueError(f"unknown stage {stage}")
