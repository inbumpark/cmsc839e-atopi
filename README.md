# atopi: context vs. MLLM dermatology classification (SCIN)

Runs the four-stage experiment from `CMSC839E_Proposal.pdf` on a local Qwen3-VL through Ollama, then computes the RQ1–RQ3 metrics.

| Stage | Model sees |
|---|---|
| 1 | image only |
| 2 | image + **one random** context field (seeded pick from the case's available fields) |
| 3 | image + **one field the model asks for** (it picks from a menu of the case's available fields, then gets the answer) |
| 4 | image + **all** available context fields |

Each stage runs in two modes:
- **independent**: every stage is a fresh conversation.
- **cascade**: stages 2–4 continue the stage-1 conversation (same seed), and the model is asked to reconsider.

Every (case, stage, mode) is sampled `n_samples` times with different seeds. Agreement across samples is the uncertainty metric.

## Setup (once)
```bash
brew install ollama && brew services start ollama
ollama pull qwen3-vl:8b-instruct        # or qwen3-vl:4b-instruct (smaller/faster)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Use an `-instruct` tag. The plain `qwen3-vl:8b` tag is the *thinking* model: it ignores `think: false` and spends about 100 s reasoning per answer.

## Run
```bash
source .venv/bin/activate
python -m derm_eval.schema data/cases.jsonl          # validate Su Hwan's data first
python -m derm_eval.run --config configs/default.yaml
python -m derm_eval.analyze results/default
```
- **Resumable:** rerun the same command after an interruption and finished queries are skipped.
- **Useful flags:** `--max-cases 20`, `--n-samples 1`, `--stages 1,4`, `--modes independent`, `--run-name pilot`.
- **Pipeline check without real data:** `python scripts/make_sample_data.py && python -m derm_eval.run --config configs/smoke.yaml`.
- **Query count per case:** `n_samples × (1 + 3 × #modes)` = 35 with the defaults. Stage 3 makes two calls.

## Data format (for the preprocessing step)
Put the data in `data/cases.jsonl`, one JSON object per line. Image paths are relative to the jsonl file.
```json
{"case_id": "-1234567890",
 "images": ["images/-1234567890_1.png"],
 "label": "Eczema",
 "label_confidence": 4,
 "differential": {"Eczema": 0.67, "Allergic Contact Dermatitis": 0.33},
 "context": {"duration": "1 to 4 weeks", "body_location": "arm, leg", "texture": "rough or flaky",
             "symptoms": "itching", "other_symptoms": "none", "age": "30-39", "sex": "female",
             "fitzpatrick": "FST3"}}
```
| key | required | meaning |
|---|---|---|
| `case_id` | yes | unique id (SCIN `case_id`) |
| `images` | yes | list of image paths; the config's `images_per_case` decides how many are sent |
| `label` | yes | reference diagnosis: the top dermatologist label (e.g. argmax of `weighted_skin_condition_label`) |
| `label_confidence` | no | dermatologist confidence (1–5) for `label` (`dermatologist_skin_condition_confidence`), used for RQ3 |
| `differential` | no | dermatologist differential (label → weight), used for the "label in differential" accuracy |
| `context` | no | field → **human-readable** value; leave a field out (or null) if it's missing |

- **Context fields:** the allowed keys, and the wording the model sees for each, are in `context_fields` in `configs/default.yaml`. To add or rename fields, edit that list; no code changes are needed.
- **Labels:** labels should use SCIN's names. `configs/labels.yaml` maps paraphrases ("atopic dermatitis" → Eczema) to those names.

## Outputs
- **`results/<run>/raw.jsonl`:** one record per answer: prompt fields given, requested field (stage 3), raw model text, parsed diagnosis, reason, confidence, latency.
- **`results/<run>/analysis/`** (all grading happens here, so it can be redone without re-querying):
  - `rq1_accuracy.csv`: per-sample accuracy by stage × mode (exact label with bootstrap CI, label-in-differential, binary eczema acc/precision/recall); `rq1_majority_vote.csv`
  - `rq2_stage2_vs_stage3.csv`: paired difference with bootstrap CI and exact McNemar test; `rq2_stage3_field_choice.csv`, `rq2_accuracy_by_field.csv`
  - `rq3_confidence.csv`: mean confidence, confidence when right/wrong, ECE (confidence/5), AUROC, Spearman vs dermatologist confidence; `rq3_reliability.csv`
  - `change_vs_stage1.csv` (flip rates), `uncertainty_seed_agreement.csv`
  - `accuracy_by_stage.png`, `reliability.png`, `stage3_field_choice.png`, `summary.txt`
  - `unmatched_diagnoses.csv`: model answers the synonym map couldn't grade. Add aliases to `configs/labels.yaml` and re-run analyze.

## Layout
`derm_eval/prompts.py` holds all prompt wording (the proposal prompt, verbatim). The other modules are `stages.py` (stage logic), `client.py` (Ollama), `run.py`, `normalize.py`, `analyze.py` and `schema.py`. Tests: `pytest`.
