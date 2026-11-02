# atopi: context vs. MLLM dermatology classification (SCIN)

Runs the four-stage experiment from `CMSC839E_Proposal.pdf` on a local Qwen3-VL through [Ollama](https://ollama.com), then computes the RQ1–RQ3 metrics.

| Stage | Model sees |
|---|---|
| 1 | image only |
| 2 | image + **one random** context field (seeded pick from the case's available fields) |
| 3 | image + **one field the model asks for** (it picks from a menu of the case's available fields, then gets the answer) |
| 4 | image + **all** available context fields |

Each stage runs in two modes:
- **independent**: every stage is a fresh conversation.
- **cascade**: stages 2–4 continue the stage-1 conversation (same seed), and the model is asked to reconsider.

Every (case, stage, mode) is sampled `n_samples` times (default 5) with different seeds. Agreement across samples is the uncertainty metric.

---

## 1. Setup

### Requirements
- **macOS on Apple Silicon** (tested on an M4 with 16 GB) or **Linux with an NVIDIA GPU**.
- **Python 3.10+**.
- **The [gcloud CLI](https://cloud.google.com/sdk/docs/install)**, only if you rebuild the data from SCIN with `data_parser.py` (see §2).
- **About 10 GB of free disk space:** roughly 6 GB for the model, plus headroom for swap. On a 16 GB Mac, close memory-heavy apps during a run. If RAM runs short, macOS swaps to disk, and with a nearly full disk that can fail.

### Steps
```bash
# 1. Get the code
git clone https://github.com/inbumpark/cmsc839e-atopi.git
cd cmsc839e-atopi

# 2. Install and start Ollama
brew install ollama && brew services start ollama          # macOS
# curl -fsSL https://ollama.com/install.sh | sh             # Linux (then: ollama serve &)

# 3. Download the model (~6 GB)
ollama pull qwen3-vl:8b-instruct

# 4. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Check the install
pytest                                    # 25 tests, no model needed
python scripts/make_sample_data.py        # tiny synthetic dataset in data/sample/
python -m derm_eval.run --config configs/smoke.yaml
python -m derm_eval.analyze results/smoke
```
The smoke run sends 14 queries on synthetic images; the answers themselves are meaningless. It's working if every line ends in `-> <diagnosis> (<confidence>)` and `results/smoke/analysis/` fills with CSVs and PNGs.

> **Use an `-instruct` model tag.** The plain `qwen3-vl:8b` tag is the *thinking* model: it ignores `think: false` and spends about 100 s reasoning per answer. If memory is tight, use `qwen3-vl:4b-instruct`: pull it and set `model:` in `configs/default.yaml`.

---

## 2. Add the preprocessed data

Put the files here:
```
data/
  cases.jsonl
  images/...
```
`data/` is git-ignored, so the data never goes into the repo; share it separately.

### Building the data from SCIN (`data_parser.py`)
`data_parser.py` builds `data/cases.jsonl` and `data/images/` straight from the public SCIN bucket on Google Cloud Storage:
1. `load_dataset()` downloads `scin_cases.csv` and `scin_labels.csv` and merges them on `case_id`.
2. `get_case_ids_with_condition(df, 'Eczema')` picks every case with Eczema anywhere in `weighted_skin_condition_label` (1056 cases).
3. `export_cases(df, case_ids)` downloads each case's 1–3 images to `data/images/{case_id}_{n}.png`. It then appends one line to `cases.jsonl` in the format below:
   - `label_confidence` is the highest confidence among the dermatologists who gave Eczema.
   - `differential` is the weighted label dict.
   - `context` holds the self-reported answers, made readable (e.g. `AGE_30_TO_39` → `30-39`). Unknown or unspecified answers are left out.

Note that `label` is always `Eczema`, even when Eczema isn't the top-weighted diagnosis; check `differential` for the weights.

The Python packages, including `google-cloud-storage`, are installed by `pip install -r requirements.txt` in setup step 4. You also need to log in to Google Cloud once per machine (needs the [gcloud CLI](https://cloud.google.com/sdk/docs/install)):
```bash
gcloud auth application-default login
```
Run:
```bash
python data_parser.py --limit 5    # small test run
python data_parser.py              # all 1056 cases
```
The export can be resumed. Cases already in `cases.jsonl` are skipped and images already on disk aren't downloaded again, so an interrupted run can simply be restarted. To start over, delete `data/cases.jsonl` and `data/images/`.

`cases.jsonl` holds one JSON object per line. Image paths are relative to the jsonl file:
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
| `images` | yes | list of image paths; the config's `images_per_case` (default 1) decides how many are sent |
| `label` | yes | reference diagnosis: the top dermatologist label (e.g. argmax of `weighted_skin_condition_label`) |
| `label_confidence` | no | dermatologist confidence (1–5) for `label` (`dermatologist_skin_condition_confidence`), used for RQ3 |
| `differential` | no | dermatologist differential (label → weight), used for the "label in differential" accuracy |
| `context` | no | field → **human-readable** value; leave a field out (or null) if it's missing |

- **Context fields:** the allowed keys, and the wording the model sees for each, are in `context_fields` in `configs/default.yaml`. To add or rename fields, edit that list; no code changes are needed.
- **Labels:** labels should use SCIN's names. `configs/labels.yaml` maps paraphrases ("atopic dermatitis" → Eczema) to those names.

Validate before running:
```bash
python -m derm_eval.schema data/cases.jsonl
```
This prints the label counts and context-field coverage. It exits with an error for missing images, and warns about unknown fields, cases with fewer than 2 context fields, and labels that have no aliases in `labels.yaml`.

---

## 3. Run the experiment

```bash
source .venv/bin/activate
brew services start ollama        # if not already running

# Pilot first: 20 cases. Check the timing and the unmatched diagnoses.
python -m derm_eval.run --config configs/default.yaml --max-cases 20 --run-name pilot
python -m derm_eval.analyze results/pilot

# Full run
python -m derm_eval.run --config configs/default.yaml
python -m derm_eval.analyze results/default
```

| flag | effect |
|---|---|
| `--run-name NAME` | output folder `results/NAME/` (use a new name when changing model/temperature/seeds) |
| `--max-cases N` | first N cases only |
| `--n-samples K` | samples per (case, stage, mode); default 5 |
| `--stages 1,2,3,4` | subset of stages |
| `--modes independent,cascade` | subset of modes |

- **Cost:** each case needs `n_samples × (1 + 3 × #modes)` = 35 answers with the defaults (stage 3 makes 2 calls). Each line of the progress output shows the latency and an ETA; multiply the pilot's per-query time to estimate the full run.
- **Resuming:** if a run is interrupted (Ctrl-C, sleep, crash), re-run the *same* command; finished queries are skipped. The runner refuses to mix results if you change the model/temperature/seed settings under an existing run name.
- **Keeping the Mac awake:** for long runs, wrap the command in `caffeinate -i` so the Mac doesn't sleep.

All settings (model, temperature, seeds, image size, number of samples, the eczema target label, context fields) are in `configs/default.yaml`. To make a variant, create a new YAML with `base: default.yaml` and only the keys you change (see `configs/smoke.yaml`).

---

## 4. Outputs

- **`results/<run>/raw.jsonl`:** one record per answer: fields given, requested field and its reason (stage 3), raw model text, parsed diagnosis, reason, confidence (1–5), latency.
- **`results/<run>/config.yaml`:** the exact config used.
- **`results/<run>/analysis/`:** all grading happens here, so it can be redone without re-querying the model.

| file | what |
|---|---|
| `summary.txt` | all tables below in one text file |
| `rq1_accuracy.csv` | per-sample accuracy by stage × mode: exact label (bootstrap 95% CI), label in differential, binary eczema acc/precision/recall, mean confidence |
| `rq1_majority_vote.csv` | the same, using the majority answer across samples |
| `rq2_stage2_vs_stage3.csv` | random vs model-chosen field: paired difference with CI and exact McNemar p |
| `rq2_stage3_field_choice.csv` / `rq2_accuracy_by_field.csv` | which field the model asks for; accuracy per field |
| `rq3_confidence.csv` | mean confidence, confidence when right/wrong, ECE (confidence/5), AUROC, Spearman vs dermatologist confidence |
| `rq3_reliability.csv` | accuracy at each confidence level |
| `change_vs_stage1.csv` | how often the diagnosis flips vs stage 1 (wrong→right, right→wrong), confidence change |
| `uncertainty_seed_agreement.csv` | agreement / entropy across seeds, split by correct vs wrong |
| `graded_answers.csv`, `majority_votes.csv` | row-level data for custom analysis |
| `accuracy_by_stage.png`, `reliability.png`, `stage3_field_choice.png` | figures |
| `unmatched_diagnoses.csv` | model answers that couldn't be mapped to a SCIN label |

**Grading loop:** after a run, open `unmatched_diagnoses.csv`, add the missing synonyms to `configs/labels.yaml`, and re-run `python -m derm_eval.analyze results/<run>`. Unparseable answers and unmatched diagnoses count as incorrect.

---

## 5. Troubleshooting

| symptom | fix |
|---|---|
| `Cannot reach Ollama` | `brew services start ollama` (macOS) or `ollama serve` (Linux) |
| `Model '...' is not pulled` | `ollama pull <model>` with the name in `configs/default.yaml` |
| ~100 s per query | you're on a thinking model; switch to an `-instruct` tag |
| `no space left on device` while pulling | free disk space; leftover blobs in `~/.ollama/models` are reused by the next pull |
| very slow queries, system lag | memory pressure: close other apps, or use `qwen3-vl:4b-instruct` / a lower `image_max_side` |
| many `FAILED` lines | check `raw` in `raw.jsonl`; raise `max_retries` or lower `temperature` |

## Code layout
```
configs/default.yaml   all experiment settings
configs/labels.yaml    SCIN label -> synonyms (grading)
derm_eval/prompts.py   all prompt wording (proposal prompt, verbatim)
derm_eval/stages.py    stage 1-4 logic, independent + cascade
derm_eval/client.py    Ollama calls, image encoding, JSON parsing
derm_eval/run.py       resumable runner
derm_eval/analyze.py   metrics + plots
derm_eval/normalize.py free-text diagnosis -> SCIN label
derm_eval/schema.py    data loading + validation
scripts/make_sample_data.py   synthetic smoke-test data
data_parser.py         SCIN (GCS) -> data/cases.jsonl + data/images/ (Eczema cases)
tests/                 pytest suite (mock model)
```
