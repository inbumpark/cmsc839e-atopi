# atopi: context vs. MLLM dermatology classification (SCIN)

Runs the four-stage experiment from `CMSC839E_Proposal.pdf` on local vision-language models (Qwen3-VL, MedGemma, Gemma 3) through [Ollama](https://ollama.com), then computes the RQ1–RQ3 metrics. **Results so far: [section 6](#6-results-nov-2026).**

| Stage | Model sees |
|---|---|
| 1 | image only |
| 2 | image + **one random** context field (seeded pick from the case's available fields) |
| 3 | image + **one field the model asks for** (it picks from a menu of the case's available fields, then gets the answer) |
| 4 | image + **all** available context fields |

Each stage runs in two modes:
- **independent**: every stage is a fresh conversation.
- **cascade**: stages 2–4 each continue the stage-1 conversation (same seed), and the model is asked to reconsider (1→2, 1→3, 1→4).
- **chains** (optional): 1→2→4 and 1→3→4 continue a finished cascade run; see [Cascade chains](#cascade-chains-124-134).

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
| `--modes independent,cascade` | subset of modes (also `chain_124`, `chain_134`; see "Cascade chains" below) |
| `--shard I/N` | run only shard I of N (0-based); see "Splitting a run across GPUs" below |
| `--ollama-host URL` | use this Ollama server instead of `ollama_host` from the config |
| `--history-from RUN_DIR` | finished run whose cascade conversations `chain_*` modes continue (not needed when the run itself has `cascade` first, as in `configs/conf.yaml`) |

- **Cost:** each case needs `n_samples × (1 + 3 × #modes)` = 35 answers with the defaults (stage 3 makes 2 calls). Each line of the progress output shows the latency and an ETA; multiply the pilot's per-query time to estimate the full run.
- **Resuming:** if a run is interrupted (Ctrl-C, sleep, crash), re-run the *same* command; finished queries are skipped. The runner refuses to mix results if you change the model/temperature/seed settings under an existing run name.
- **Keeping the Mac awake:** for long runs, wrap the command in `caffeinate -i` so the Mac doesn't sleep.

### Splitting a run across GPUs
With several GPUs, run one Ollama server per GPU and one shard per server. `--shard I/N` takes every N-th case starting at case I and writes to `results/<run>/shards/<I>of<N>/`. Each shard resumes on its own. When all shards are done, merge them and analyze as usual:
```bash
# e.g. two GPUs, each with its own Ollama server
python -m derm_eval.run --run-name big --shard 0/2 --ollama-host http://127.0.0.1:11434 &
python -m derm_eval.run --run-name big --shard 1/2 --ollama-host http://127.0.0.1:11435 &
wait
python -m derm_eval.merge results/big      # checks all shards match and are complete
python -m derm_eval.analyze results/big
```
`merge` refuses shards run with different settings, and it lists planned queries that are missing; finish those shards or pass `--allow-incomplete`. You can re-run it safely.

### Cascade chains (1→2→4, 1→3→4)
Cascade mode already branches 1→2, 1→3 and 1→4 from the same stage-1 chat. The `chain_124` and `chain_134` modes (stage 4 only) continue the finished 1→2 / 1→3 chats of an earlier run with the stage-4 turn (all context, same prompt as 1→4). The earlier chats are rebuilt exactly from that run's records, so only the final turn is queried (2 × n_samples answers per case):
```bash
python -m derm_eval.run --config configs/mcq.yaml --run-name mcq_chains --stages 4 \
    --modes chain_124,chain_134 --history-from results/mcq
python -m derm_eval.chains results/mcq results/mcq_chains --exclude-no-context
```
`chains` writes accuracy for 1, 1→2, 1→3, 1→4, 1→2→4, 1→3→4 (and fresh-chat stage 4), plus paired differences, to `results/mcq_chains/analysis_with_context/`. The history run must use the same model and settings (checked at start).

### Confidence-only experiment (`configs/conf.yaml`)
The model is **told the patient was diagnosed with eczema** and only rates its confidence in that diagnosis on a scale of 1 to 5 (`answer_format: confidence`). It names no condition itself. The prompt keeps the proposal's framing (`CONF_*` in `derm_eval/prompts.py`):

> You are a skilled dermatologist … Here is a photo from a patient who has been diagnosed with eczema. Rate your confidence in this diagnosis on a scale of 1 to 5, and give your reason for the rating in a few words.

Each sample is one growing conversation, asked in this order (6 answers, 7 model calls per sample):

| path | what the model has seen |
|---|---|
| 1 | image only |
| 1→2 | + one random context field |
| 1→3 | + one context field the model asks for |
| 1→4 | + all context |
| 1→2→4 | the 1→2 chat, then all context |
| 1→3→4 | the 1→3 chat, then all context |

1→2, 1→3 and 1→4 all branch from the same stage-1 answer, and the two chains continue the 1→2 / 1→3 chats exactly, so no `--history-from` is needed: `modes: [cascade, chain_124, chain_134]`. (`mode_stages: {cascade: [2, 3]}` would skip the direct 1→4.)
```bash
python -m derm_eval.run --config configs/conf.yaml --max-cases 20 --run-name conf_pilot   # pilot
python -m derm_eval.run --config configs/conf.yaml                                        # -> results/conf
python -m derm_eval.conf results/conf                                                     # -> results/conf/conf_analysis/
```
`--shard` / `merge` work as above. `configs/conf_medgemma4b.yaml` and `configs/conf_gemma3_4b.yaml` run the same experiment with the other two models.

`derm_eval.conf` (not `analyze`, which grades diagnoses) writes these tables. Every case is labelled Eczema; the references are the dermatologists' own 1–5 confidence in that label and their weight for Eczema in the case's differential (1.0 = all of them said only Eczema). Cases with no context are left out.

| file | what |
|---|---|
| `conf_by_path.csv` | per path: mean confidence (95% CI over cases), share of answers at each level 1–5, spread across seeds, and difference / mean absolute error / Spearman against the dermatologist confidence, Spearman against the dermatologist weight |
| `conf_step_change.csv` | what the last step did to the same sample's previous rating: mean change (CI, p), share raised / lowered / unchanged |
| `conf_paired_paths.csv` | paired differences between paths (1→3 vs 1→2, 1→3→4 vs 1→2→4, each chain vs 1 and vs 1→4) |
| `conf_by_field.csv` | stages 2 and 3 by the field given: mean confidence, change from stage 1, how often the model asks for it |
| `conf_by_derm_confidence.csv`, `conf_by_derm_weight.csv` | mean confidence per path, by how sure the dermatologists were |
| `answers.csv` | row-level data |

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

## 6. Results (Nov 2026)

Raw outputs are in `results/` (gitignored); every number below can be regenerated from them with the commands in [How these were run](#how-these-were-run).

### Setup
- **Data:** the 1056 SCIN cases labelled Eczema (one image each, 1024 px max side). 129 have no context fields at all, so stages 2–4 can't run for them; **all tables use the other 927 cases** (`--exclude-no-context`).
- **Sampling:** 5 samples per case × path, temperature 0.7, seeds 0–4. Accuracy is per answer, averaged per case; brackets are 95% bootstrap CIs over cases; p-values are paired sign-flip permutation tests over cases.
- **Models** (Ollama, Q4_K_M, thinking off): Qwen3-VL 8B Instruct, MedGemma 4B, Gemma 3 4B. ⚠️ Google's model card lists SCIN in MedGemma's training data, so MedGemma may have seen these cases; Gemma 3 4B is the clean comparison.
- **Free-text baseline (Qwen3-VL 8B, `results/default`):** asked for an open diagnosis, the model named eczema in only **1–2%** of answers at every stage and mode (it mostly said "contact dermatitis"), so the task was switched to multiple choice.
- **Multiple choice (`configs/mcq.yaml`):** 4 options per case: Eczema plus 3 distractors drawn from the other 355 SCIN labels, weighted by how often each label occurs (eczema synonyms and nonspecific "dermatitis" labels excluded). The options are shuffled per case with a fixed seed, so every stage, path, sample and model sees exactly the same 4 options. Chance = 25%. Correct = the model picked Eczema.
- **Paths:** "new chat" stages are independent conversations. Arrows are one conversation continued turn by turn: 1→2, 1→3 and 1→4 each continue the same stage-1 answer (cascade mode); 1→2→4 and 1→3→4 continue the saved 1→2 / 1→3 conversations with the all-context turn (chain modes). The final turn of 1→2→4 and 1→3→4 is the same prompt as 1→4.

### Accuracy (% of answers that picked Eczema)

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 38.9 [36.0, 42.0] | 58.9 [55.6, 62.1] | 58.5 [55.2, 61.6] |
| 2 random field (new chat) | 42.0 [39.3, 44.5] | 60.2 [57.4, 63.0] | 59.1 [56.4, 61.6] |
| 3 model-chosen field (new chat) | 44.8 [41.7, 48.0] | 42.2 [39.1, 45.1] | 35.3 [32.2, 38.3] |
| 4 all context (new chat) | 38.4 [35.3, 41.4] | 54.8 [51.6, 58.0] | 49.1 [45.6, 52.2] |
| 1→2 | 39.9 [37.4, 42.7] | 60.0 [57.0, 63.0] | 57.9 [54.8, 61.2] |
| 1→3 | 28.9 [26.3, 31.8] | 60.1 [56.9, 63.1] | 47.1 [44.1, 50.1] |
| 1→4 | 40.0 [37.0, 43.0] | 61.1 [57.9, 64.1] | 58.8 [55.7, 62.0] |
| 1→2→4 | 40.8 [38.1, 43.7] | 62.6 [59.6, 65.6] | 59.7 [56.9, 62.7] |
| 1→3→4 | 35.9 [33.1, 38.7] | 60.4 [57.3, 63.2] | 49.5 [46.4, 52.5] |

Paired differences in accuracy (percentage points):

| comparison | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 2 vs 1 (new chat) | +3.1 (p<0.001) | +1.3 (p=0.14) | +0.6 (p=0.55) |
| 3 vs 2 (new chat) | +2.8 (p=0.09) | −18.0 (p<0.001) | −23.8 (p<0.001) |
| 4 vs 1 (new chat) | −0.5 (p=0.77) | −4.1 (p=0.003) | −9.4 (p<0.001) |
| 1→3 vs 1→2 | −11.0 (p<0.001) | +0.1 (p=0.80) | −10.9 (p<0.001) |
| 1→4 vs 1 | +1.1 (p=0.42) | +2.2 (p<0.001) | +0.3 (p=0.31) |
| 1→2→4 vs 1→4 | +0.9 (p=0.17) | +1.5 (p<0.001) | +0.9 (p=0.19) |
| 1→3→4 vs 1→4 | −4.0 (p<0.001) | −0.7 (p=0.04) | −9.3 (p<0.001) |

### Which context helps, and which the model asks for

Change in accuracy when a field is given at random (stage 2, vs the same sample's image-only answer), and how often the model asks for that field in stage 3 (new chat):

| field | Qwen Δ | MedGemma Δ | Gemma Δ | Qwen asks | MedGemma asks | Gemma asks |
|---|---|---|---|---|---|---|
| symptoms | +24.5 | +11.2 | +3.5 | 44% | 0% | 1% |
| texture | +7.9 | +10.5 | +11.5 | 2% | 3% | 2% |
| body_location | +5.6 | −2.3 | −0.3 | 5% | 2% | 1% |
| sex | −1.0 | +6.9 | +7.6 | 0% | 0% | 0% |
| other_symptoms | −3.9 | +2.3 | +1.7 | 1% | 0% | 0% |
| age | 0.0 | 0.0 | −0.6 | 1% | 0% | 0% |
| fitzpatrick | −1.1 | −0.8 | −2.5 | 0% | 1% | 0% |
| **duration** | **−11.2** | **−15.0** | **−14.5** | 48% | **94%** | **95%** |

### Confidence (self-reported, 1–5)

Mean confidence (and % of answers rated 5):

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 4.06 (8%) | 5.00 (100%) | 4.06 (6%) |
| 2 random field (new chat) | 4.05 (7%) | 4.97 (98%) | 4.04 (5%) |
| 3 model-chosen field (new chat) | 4.56 (56%) | 5.00 (100%) | 4.00 (0%) |
| 4 all context (new chat) | 4.03 (4%) | 5.00 (100%) | 4.01 (1%) |
| 1→2 | 4.49 (51%) | 4.99 (99%) | 4.96 (96%) |
| 1→3 | 4.84 (85%) | 4.98 (98%) | 4.55 (56%) |
| 1→4 | 4.64 (66%) | 4.99 (99%) | 4.97 (97%) |
| 1→2→4 | 4.45 (48%) | 5.00 (100%) | 4.94 (95%) |
| 1→3→4 | 4.84 (85%) | 4.99 (99%) | 4.97 (97%) |

Does confidence separate right from wrong answers? AUROC of confidence vs correctness (0.5 = no better than chance), with mean confidence when right / wrong:

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 0.45 (4.00 / 4.10) | 0.50 (5.00 / 5.00) | 0.43 (4.00 / 4.14) |
| 2 random field (new chat) | 0.46 (4.00 / 4.08) | 0.49 (4.96 / 4.98) | 0.45 (4.00 / 4.10) |
| 3 model-chosen field (new chat) | 0.48 (4.54 / 4.58) | 0.50 (5.00 / 4.99) | 0.50 (4.00 / 4.00) |
| 4 all context (new chat) | 0.49 (4.01 / 4.04) | 0.50 (5.00 / 4.99) | 0.49 (4.00 / 4.02) |
| 1→2 | 0.49 (4.48 / 4.49) | 0.50 (4.99 / 4.99) | 0.53 (4.99 / 4.92) |
| 1→3 | 0.52 (4.88 / 4.82) | 0.49 (4.97 / 4.99) | 0.50 (4.55 / 4.54) |
| 1→4 | 0.53 (4.68 / 4.61) | 0.50 (4.99 / 4.99) | 0.53 (4.99 / 4.94) |
| 1→2→4 | 0.49 (4.45 / 4.45) | 0.50 (4.99 / 5.00) | 0.52 (4.97 / 4.91) |
| 1→3→4 | 0.48 (4.82 / 4.86) | 0.50 (4.99 / 4.99) | 0.50 (4.97 / 4.97) |

How often the last turn of a conversation changed the answer (same sample):

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1→2 (vs 1) | 10% | 2% | 2% |
| 1→3 (vs 1) | 24% | 2% | 17% |
| 1→4 (vs 1) | 24% | 3% | 2% |
| 1→2→4 (vs 1→2) | 21% | 4% | 9% |
| 1→3→4 (vs 1→3) | 15% | 1% | 6% |

### Findings
1. **Smaller models win on the image alone.** Both 4B models beat Qwen3-VL 8B by about 20 points (58.5–58.9% vs 38.9%). Gemma 3 4B matches MedGemma, so MedGemma's possible exposure to SCIN gives it no visible edge.
2. **Context barely helps.** A random field adds at most +3 points. All context in a new chat hurts the 4B models (−4.1 MedGemma, −9.4 Gemma). Given to a model that has already answered (1→4), all context changes accuracy by at most +2 points.
3. **Letting the model choose a field hurts, because it chooses badly.** The 4B models ask for **duration** 94–95% of the time, and duration is the most harmful field (−11 to −15 points when given at random). The useful fields (symptoms, texture) are almost never asked for. Qwen asks for symptoms 44% of the time (+24.5 points when given) and is the only model not hurt in a new chat. In a continued chat, 1→3 still costs Qwen 11 points.
4. **Order matters a little; a bad stage-3 turn persists.** Seeing a random field before all context (1→2→4) is within 1.5 points of going straight to 1→4. But 1→3→4 stays below 1→4 (−4.0 Qwen, −9.3 Gemma): all context afterwards doesn't undo the damage of the self-chosen field.
5. **The models mostly stick with their first answer.** MedGemma changes its answer in 1–4% of follow-up turns, so its continued-chat rows mostly repeat its stage-1 answer. Gemma changes 2% at 1→2 and 1→4 but 17% at 1→3, right after its own field choice. Qwen changes most (10–24%).
6. **Confidence carries no information.** AUROC is 0.43–0.53 for every model and path, and with the image only, Qwen and Gemma are slightly *more* confident when wrong. MedGemma answers 5/5 almost always. Follow-up turns raise confidence without raising accuracy: Gemma goes from 4.06 to about 4.96 once it is asked to reconsider, while its answer changes 2% of the time. Qwen's highest confidence (4.84) is at 1→3, its least accurate path (28.9%).
7. **Errors and biases.** The most common wrong picks for all three models are Allergic Contact Dermatitis, Insect Bite and Urticaria (acute look-alikes, consistent with duration pushing answers away from eczema). MedGemma rarely picks option D (10% of answers) and gets 20% right when Eczema is D vs 67–76% otherwise; Qwen and Gemma show no strong position bias.

### Caveats
- **MedGemma and SCIN:** see above; treat MedGemma numbers as possibly contaminated.
- **Label noise:** SCIN labels come from dermatologists looking at images. Accuracy stays around 40% for Qwen even on cases where the dermatologists put full weight on Eczema, so label noise doesn't explain the low scores.
- **GPU nondeterminism:** runs were split over RTX 2080 Ti and GTX 1080 Ti cards. Same-GPU-type shards reproduce answers exactly, but about 3% of answers differ between GPU types, and Ollama's prompt cache makes the final chain turn not bit-reproducible. This is noise, too small to move any number above.
- **Larger models not run:** Qwen3-VL 30B-A3B and MedGemma 27B don't fit in the 11 GB 2080 Ti GPUs available (30B-A3B ran at about 0.5 tokens/s with experts spilled to CPU).

### Rerun: far distractors, certain-Eczema cases (`configs/mcq_far*.yaml`)

The random distractors above are often conditions dermatologists confuse with eczema, and many "Eczema" cases are only partly Eczema in SCIN's differential (one has weight 0.18). This rerun changes two things and repeats every test:
- **Fixed far distractors:** every case gets the same 3 wrong options, **Acne, Rosacea and Herpes Zoster** (shuffled with Eczema per case). They are the SCIN conditions that dermatologists least often list in the same differential as Eczema: only 4.1%, 5.8% and 8.5% of the cases with each condition also list Eczema (among labels with ≥30 cases; `python scripts/rank_mcq_distractors.py`). Correlation of the differential weights was not used because it mostly reflects how common a label is: it ranks Insect Bite and Urticaria as least related although they share a differential with Eczema in about 21% of cases.
- **Certain cases only:** `min_label_weight: 1.0` keeps the cases whose differential is 100% Eczema, i.e. every dermatologist said only Eczema: **127 of the 1056 cases, 104 with context**. Tables use those 104. With 104 cases the 95% CIs are about ±8 points (vs ±3 above).

Config keys (any MCQ config): `mcq_distractors: [A, B, C]` fixes the distractors (must be SCIN labels, not the target); `min_label_weight: W` keeps only cases whose label weight in the differential is ≥ W (e.g. 0.67 keeps 186 cases with context).

Accuracy (% of answers that picked Eczema) [95% CI]:

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 46.0 [36.7, 55.4] | 86.2 [79.4, 92.3] | 81.7 [74.0, 88.5] |
| 2 random field (new chat) | 56.9 [48.7, 65.2] | 87.3 [81.9, 91.9] | 81.7 [75.4, 87.3] |
| 3 model-chosen field (new chat) | 74.6 [66.9, 82.5] | 64.0 [54.8, 72.5] | 48.7 [39.6, 57.7] |
| 4 all context (new chat) | 73.5 [64.8, 81.2] | 87.5 [80.8, 93.3] | 76.9 [69.2, 84.6] |
| 1→2 | 55.0 [46.9, 63.3] | 87.3 [81.2, 92.9] | 81.5 [74.0, 88.5] |
| 1→3 | 37.5 [29.0, 46.2] | 85.4 [78.7, 91.5] | 77.7 [69.8, 85.2] |
| 1→4 | 71.5 [63.5, 79.6] | 88.5 [82.1, 94.0] | 82.7 [76.0, 89.4] |
| 1→2→4 | 72.7 [65.2, 80.0] | 88.5 [82.9, 93.7] | 84.0 [77.3, 90.2] |
| 1→3→4 | 65.8 [56.9, 74.4] | 85.4 [78.3, 91.9] | 81.0 [73.8, 88.3] |

Which change made the difference? Accuracy per path for the random-distractor runs on all 927 cases → the same runs restricted to the 104 certain cases → the far-distractor runs on those 104 cases:

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 38.9 → 37.9 → 46.0 | 58.9 → 70.6 → 86.2 | 58.5 → 62.5 → 81.7 |
| 2 random field (new chat) | 42.0 → 42.9 → 56.9 | 60.2 → 73.7 → 87.3 | 59.1 → 65.0 → 81.7 |
| 3 model-chosen field (new chat) | 44.8 → 44.8 → 74.6 | 42.2 → 43.3 → 64.0 | 35.3 → 38.5 → 48.7 |
| 4 all context (new chat) | 38.4 → 40.8 → 73.5 | 54.8 → 68.3 → 87.5 | 49.1 → 58.7 → 76.9 |
| 1→2 | 39.9 → 38.8 → 55.0 | 60.0 → 71.0 → 87.3 | 57.9 → 61.9 → 81.5 |
| 1→3 | 28.9 → 23.7 → 37.5 | 60.1 → 71.3 → 85.4 | 47.1 → 51.5 → 77.7 |
| 1→4 | 40.0 → 41.2 → 71.5 | 61.1 → 72.3 → 88.5 | 58.8 → 62.7 → 82.7 |
| 1→2→4 | 40.8 → 41.2 → 72.7 | 62.6 → 72.7 → 88.5 | 59.7 → 63.8 → 84.0 |
| 1→3→4 | 35.9 → 33.5 → 65.8 | 60.4 → 71.3 → 85.4 | 49.5 → 53.8 → 81.0 |

Paired differences in accuracy (percentage points):

| comparison | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 2 vs 1 (new chat) | +11.0 (p<0.001) | +1.2 (p=0.68) | +0.0 (p=1.00) |
| 3 vs 2 (new chat) | +17.7 (p<0.001) | −23.3 (p<0.001) | −33.1 (p<0.001) |
| 4 vs 1 (new chat) | +27.5 (p<0.001) | +1.3 (p=0.79) | −4.8 (p=0.42) |
| 1→3 vs 1→2 | −17.5 (p<0.001) | −1.9 (p=0.08) | −3.8 (p=0.02) |
| 1→4 vs 1 | +25.6 (p<0.001) | +2.3 (p=0.25) | +1.0 (p=1.00) |
| 1→2→4 vs 1→4 | +1.2 (p=0.66) | +0.0 (p=1.00) | +1.3 (p=0.33) |
| 1→3→4 vs 1→4 | −5.8 (p=0.03) | −3.1 (p=0.06) | −1.7 (p=0.46) |

Change in accuracy when a field is given at random (stage 2 vs the same sample's stage 1; n = answers) and how often the model asks for it in stage 3 (new chat):

| field | Qwen Δ | MedGemma Δ | Gemma Δ | Qwen asks | MedGemma asks | Gemma asks |
|---|---|---|---|---|---|---|
| symptoms (n=73) | **+42.5** | +2.7 | +8.2 | 66% | 0% | 2% |
| texture (n=107) | +9.3 | +8.4 | +5.6 | 0% | 2% | 2% |
| body_location (n=104) | +19.2 | +3.8 | −2.9 | 10% | 4% | 3% |
| other_symptoms (n=64) | 0.0 | +1.6 | +7.8 | 1% | 0% | 0% |
| sex (n=37) | +2.7 | −5.4 | −2.7 | 0% | 0% | 1% |
| age (n=29) | −6.9 | +6.9 | −3.4 | 2% | 0% | 0% |
| fitzpatrick (n=25) | −8.0 | 0.0 | −16.0 | 1% | 3% | 0% |
| **duration** (n=81) | −1.2 | **−12.3** | **−9.9** | 21% | **91%** | **92%** |

In a continued chat (1→3) the choice changes for Qwen: it asks for duration 83% of the time (symptoms 5%). MedGemma and Gemma ask for duration about 90% of the time in both modes.

Mean confidence (% rated 5):

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 4.06 (7%) | 5.00 (100%) | 4.05 (5%) |
| 2 random field (new chat) | 4.02 (4%) | 4.98 (99%) | 4.04 (4%) |
| 3 model-chosen field (new chat) | 4.65 (65%) | 4.99 (100%) | 4.03 (3%) |
| 4 all context (new chat) | 4.02 (2%) | 4.99 (99%) | 4.03 (3%) |
| 1→2 | 4.51 (54%) | 4.99 (99%) | 4.99 (99%) |
| 1→3 | 4.68 (72%) | 5.00 (100%) | 4.58 (59%) |
| 1→4 | 4.59 (62%) | 4.99 (99%) | 5.00 (100%) |
| 1→2→4 | 4.44 (46%) | 5.00 (100%) | 4.98 (98%) |
| 1→3→4 | 4.76 (76%) | 5.00 (100%) | 5.00 (100%) |

AUROC of confidence vs correctness (mean confidence right / wrong):

| path | Qwen3-VL 8B | MedGemma 4B | Gemma 3 4B |
|---|---|---|---|
| 1 image only | 0.43 (3.99 / 4.12) | 0.51 (5.00 / 4.99) | 0.36 (4.00 / 4.27) |
| 2 random field (new chat) | 0.48 (4.00 / 4.04) | 0.49 (4.98 / 5.00) | 0.42 (4.01 / 4.17) |
| 3 model-chosen field (new chat) | 0.58 (4.70 / 4.53) | 0.50 (5.00 / 4.99) | 0.48 (4.00 / 4.05) |
| 4 all context (new chat) | 0.50 (4.01 / 4.02) | 0.51 (5.00 / 4.97) | 0.44 (4.00 / 4.12) |
| 1→2 | 0.56 (4.58 / 4.44) | 0.51 (4.99 / 4.97) | 0.53 (5.00 / 4.94) |
| 1→3 | 0.59 (4.83 / 4.60) | 0.50 (5.00 / 5.00) | 0.40 (4.54 / 4.73) |
| 1→4 | 0.60 (4.67 / 4.39) | 0.50 (4.99 / 5.00) | 0.51 (5.00 / 4.99) |
| 1→2→4 | 0.56 (4.48 / 4.34) | 0.50 (5.00 / 5.00) | 0.52 (4.99 / 4.94) |
| 1→3→4 | 0.48 (4.75 / 4.78) | 0.50 (5.00 / 5.00) | 0.50 (5.00 / 5.00) |

How often the last turn changed the answer: Qwen 13% (1→2), 32% (1→3), 32% (1→4), 27% (1→2→4), 34% (1→3→4); MedGemma 0–4%; Gemma 1–5%.

Findings of the rerun:
1. **With easy distractors, context clearly helps Qwen.** All context adds +27.5 (new chat) and +25.6 (1→4) points; above it added about 0. Symptoms alone add +42.5 and body location +19.2. Against look-alike distractors these fields can't separate eczema; against Acne, Rosacea and Herpes Zoster they can.
2. **The 4B models are near their ceiling from the image alone** (MedGemma 86%, Gemma 82%), and context adds nothing significant for them. Most of their gain over the first runs comes from the distractors (Gemma 62.5 → 81.7 on the same cases), part from the certain-case filter (MedGemma 58.9 → 70.6). For Qwen the filter alone changes nothing (38.9 → 37.9).
3. **Choosing a field in a new chat still costs the 4B models 23–33 points.** They still ask for duration about 91% of the time, which still hurts them (−12 and −10 when given at random). Qwen benefits in a new chat (+17.7 vs a random field) because it asks for symptoms 66% of the time; after its own first answer it asks for duration instead, and 1→3 drops below image-only (37.5% vs 46.0%).
4. **Order:** 1→2→4 is again within about 1 point of 1→4; 1→3→4 is again lower (−5.8 for Qwen, p=0.03; −3.1 MedGemma and −1.7 Gemma, not significant).
5. **Confidence is still uninformative** (AUROC 0.36–0.60). MedGemma rates nearly every answer 5; Gemma's image-only confidence is higher when it is wrong (AUROC 0.36). Follow-up turns again push Gemma to 5/5 (5% → 99–100%) while it changes its answer in 1–5% of turns.
6. **Errors:** the most common wrong picks are Acne (Qwen 19% of answers, Gemma 12%) and Herpes Zoster (10–15%); Rosacea is almost never picked. MedGemma still avoids option D (69% right when Eczema is D vs about 90% otherwise).

### How these were run
```bash
# per model: full MCQ run (stages 1-4, new chat + cascade), then the two chains
python -m derm_eval.run --config configs/mcq_gemma3_4b.yaml                       # -> results/mcq_gemma3_4b
python -m derm_eval.run --config configs/mcq_gemma3_4b.yaml --run-name mcq_gemma3_4b_chains \
    --stages 4 --modes chain_124,chain_134 --history-from results/mcq_gemma3_4b
python -m derm_eval.analyze results/mcq_gemma3_4b --exclude-no-context
python -m derm_eval.chains results/mcq_gemma3_4b results/mcq_gemma3_4b_chains --exclude-no-context
```
Configs: `configs/mcq.yaml` (Qwen3-VL 8B → `results/mcq`), `configs/mcq_medgemma4b.yaml`, `configs/mcq_gemma3_4b.yaml`. The rerun uses `configs/mcq_far.yaml`, `configs/mcq_far_medgemma4b.yaml` and `configs/mcq_far_gemma3_4b.yaml` the same way (→ `results/mcq_far*`).

The markdown tables in this section are printed by `python scripts/results_tables.py` (random distractors) and `python scripts/results_tables.py --set far`. On the cluster each run was split into 4–6 shards (`--shard I/N`, one Ollama server per GPU) and merged with `python -m derm_eval.merge`.

---

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
derm_eval/merge.py     combine --shard outputs into one run
derm_eval/chains.py    compare cascade paths 1->2->4 / 1->3->4 against 1->2, 1->3, 1->4
derm_eval/mcq.py       multiple-choice options (Eczema + sampled or fixed distractors)
derm_eval/conf.py      analysis of confidence-only runs (configs/conf.yaml)
scripts/make_sample_data.py   synthetic smoke-test data
scripts/rank_mcq_distractors.py  rank SCIN labels by how rarely they co-occur with Eczema
scripts/results_tables.py     print the README results tables
data_parser.py         SCIN (GCS) -> data/cases.jsonl + data/images/ (Eczema cases)
tests/                 pytest suite (mock model)
```
