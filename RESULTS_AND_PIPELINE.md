# Business Entity Resolution: Pipeline, Results and Analysis

Submitted pipeline: **P5-Bwide**. Package: `output/EpochAlypse_submission.zip`, built 2026-09-27 05:54.

This document covers how the pipeline works end to end, what it scored, and an audit of the submitted outputs:
consistency checks, anomalies, and concrete ways to improve the model.

---

## 1. Problem

- **Task.** For every Source 1 (S1) record, find all Source 2 and Source 3 (S2/S3) records that are the same
  business. The only fields are `business_name`, `business_address` and `country`.
- **Metric.** Macro F0.5, computed per S1 and then averaged. Singletons count: an empty prediction for an S1
  with no true matches scores 1.0, and any prediction for it scores 0.
- **Data.**

  | split | S1 | S2 + S3 pool | pool / S1 | countries |
  |---|---|---|---|---|
  | train | 2,206,821 | 10,320,219 | 4.67 | US, India |
  | test | 1,732,544 | 9,969,589 | **5.76** | US, India, **France (unseen)** |

- **Train ground truth.** Singleton rate is 5.58% (US) and 5.59% (India). There are 3.46 links per S1 and
  48% of them go to S2. No link crosses countries, and every S2/S3 record belongs to at most one S1.

---

## 2. Pipeline, step by step

Code: `code/business_entity_resolution/src/`. There is one entry point, `run.py`, and every stage can be
resumed.

```
data ─▶ clean ─▶ key cache ─▶ wide blocking on a labelled train sample ─▶ pair features
     ─▶ Optuna: blocking (multi-objective) + LightGBM / XGBoost / CatBoost
     ─▶ 5-fold out-of-fold (OOF) evaluation of every candidate pipeline ─▶ comparison report
     ─▶ best pipeline: refit ─▶ holdout score ─▶ test retrieval + scoring ─▶ output/*.tsv ─▶ validator ─▶ zip
```

### 2.1 Cleaning (`cleaning.py`)

- **Text repair.** ftfy fixes broken encoding, then NFKC normalisation, then `anyascii` transliteration. The
  last step converts Devanagari, Telugu, Bengali and other native scripts to Latin characters.
- **Names.**
  - Lower-case, and turn `name.com` into `name`.
  - Remove phone numbers and `(india)`-style tags.
  - Turn `&` into "and" and remove punctuation.
  - Normalise transliterated "private limited" spellings (`praivet limited`, `pra li`).
  - Extract the legal form into a canonical set: llc, inc, private limited, sarl, sas, sa, sci and others.
  - Produce `name_norm` (legal forms kept as tokens) and `name_core` (legal forms and stopwords removed).
- **Addresses.** Abbreviations are expanded per country, with separate maps for US, India (English) and
  France. Region, postcode, house number and street are extracted. French departments are mapped to their
  regions.
- **Country** is kept as an open set of labels, so France flows through like any other country.

### 2.2 Blocking (`blocking.py`, `data.py`)

- **Method.** Multi-pass sparse TF-IDF top-K retrieval, with country as a hard partition (100% of true
  pairs share a country).
- **Keys.** Each record emits hashed keys from eight families:
  - `name` (name tokens)
  - `phon` (consonant skeleton, robust to typos and transliteration)
  - `join` (the joined name)
  - `nreg` (name tokens scoped by region)
  - `init` (initials)
  - `addr` (address tokens)
  - `anum` (address numbers)
  - `house` (house number + street)
- **Weighting.** IDF is computed per country over the S2+S3 pool. Keys shared by more than `cap` records are
  dropped from retrieval.
- **Passes.** There are three: `all`, `name` and `addr`. Their results are unioned.
- **Wide set.** Training samples are retrieved wide (all@50, name@30, addr@30) against the **full train
  pool**. Every blocking configuration is a truncation of this set.
- **Tuning.** Optuna NSGA-II trades the blocking ceiling against the number of candidates per S1. The chosen
  configuration, **Bwide**, is:
  - `all`: k=34, min_score 0.141
  - `name`: k=7, min_score 0.268
  - `addr`: k=24, min_score 0.135

### 2.3 Samples (`prepare.py`)

Disjoint samples of train S1 records:

- **dev**: 100k S1, split into 5 stratified folds. Every pipeline is scored on these.
- **hpo**: 50k S1, seen only by Optuna.
- **holdout**: 50k S1, scored once for the final pipeline.

### 2.4 Pair features (`features.py`, `pipelines.add_ret_ctx`)

| family | content | gain share |
|---|---|---|
| RET | retrieval scores and ranks, reciprocal-rank fusion (`rrf`) | **0.651** |
| ADDR | address ratio, token-set, char-3 cosine, IDF-weighted Jaccard, street, region, postcode | 0.150 |
| STR | name Levenshtein, Jaro-Winkler, token-set, partial, prefix, char-3 cosine | 0.088 |
| RARE | IDF-weighted name overlap, rare missing tokens, name frequency | 0.046 |
| NUM | house number and number equality or conflict | 0.034 |
| CTX | candidate-set context: rank, gap to best, counts | 0.019 |
| LEGAL | legal-form equality or conflict | 0.012 |
| TXT | phone / domain equality or conflict | **0.000** |

### 2.5 Matcher, calibration and decision (`models.py`, `pipelines.py`)

- **Leakage control.** Inside each training fold, entities are split 70 / 15 / 15 into fit / calibrate /
  tune. No stage sees scores produced from its own labels.
- **Matcher.** LightGBM with Optuna-tuned parameters: num_leaves 215, learning rate 0.0162,
  min_data_in_leaf 237, feature_fraction 0.66, bagging_fraction 0.92, pos_weight 4.68.
- **Negatives (E2).** All candidate negatives are used, and negatives of singletons get 3× weight.
- **Calibration.** Isotonic.
- **Decision (policy B).** A pair is predicted when p ≥ t **and** a singleton gate says the S1 has any match.
  The gate is a per-entity model that uses candidate-set context.
- **Test only.** Each S2/S3 record goes to at most one S1, its highest-scoring one ("unique assignment").

### 2.6 Submission (`submit.py`)

1. Choose the best pipeline by OOF macro F0.5.
2. Refit it on the full dev sample and score the holdout once.
3. Retrieve test candidates with Bwide.
4. Score them in batches of 100k S1, apply the policy, then apply unique assignment.
5. Write both TSVs, run `utils/validate_submission.py`, and zip.

---

## 3. Results

### 3.1 Leaderboard (dev, 100k S1, 5-fold OOF)

| pipeline | change | macro F0.5 | Δ vs parent (95% CI) | admitted |
|---|---|---|---|---|
| P-empty | predict nothing | 0.0562 | | |
| P-rule | blocking score threshold | 0.6811 | | |
| P0 | LightGBM, name + retrieval | 0.8941 | +0.2130 | ✔ |
| P0-lr | logistic regression | 0.8060 | −0.0881 | ✘ |
| P1 | + rarity, numeric conflicts | 0.9481 | +0.0540 | ✔ |
| P2 | full feature set | 0.9669 | +0.0188 | ✔ |
| P3-xgb / P3-cat | XGBoost / CatBoost | 0.9667 / 0.9660 | −0.0002 / −0.0008 | ✘ |
| P2-hpo-lgbm / xgb | Optuna-tuned | 0.9673 / 0.9673 | +0.0004 / +0.0007 | ✘ |
| P4 | singleton weighting + gate (B) | 0.9673 | −0.0000 | ✘ |
| P4-C | ambiguity-adjusted threshold | 0.9673 | +0.0000 | ✘ |
| P4-E1 | capped hard negatives | 0.9665 | −0.0008 | ✘ |
| P6 | + frozen multilingual-E5 cosine | 0.9677 | +0.0004 | ✘ |
| P8 | + unique assignment | 0.9674 | +0.0001 | ✘ |
| **P5-Bwide** | **P4 on Bwide blocking** | **0.9699** | **+0.0025 [0.0022, 0.0029]** | ✔ |
| P-oracle | perfect matcher (blocking ceiling, B0) | 0.9862 | | |

P5-Bwide ranks first in all 5 folds, with fold scores from 0.9692 to 0.9713.

### 3.2 Holdout (50k S1, scored once)

| metric | value |
|---|---|
| macro F0.5 | **0.9699** (CI 0.9691–0.9708) |
| blocking ceiling F0.5 | 0.9887 |
| pair precision / recall | 0.9926 / 0.9319 |
| singleton accuracy | 0.972 (29 false positives per 1000 singletons) |
| candidate pair recall | 0.968; 42 candidates per S1 (p95 57) |

### 3.3 Slices (dev OOF)

| slice | F0.5 | ceiling |
|---|---|---|
| US | 0.9783 | 0.9951 |
| India | **0.9572** | 0.9797 |
| multiplicity 1 | **0.918** | 0.972 |
| multiplicity 2 / 3–5 / 6+ | 0.962 / 0.975 / 0.979 | |
| most common names (q5) | **0.950** | 0.971 |
| train US → eval India | **0.841** | proxy for unseen France |
| train India → eval US | 0.936 | |

### 3.4 Where the remaining F0.5 is lost (P5-Bwide)

| loss | ΔF0.5 |
|---|---|
| matcher false negatives | **−0.0123** |
| blocking misses | **−0.0111** |
| matcher false positives | −0.0050 |
| singleton false positives | −0.0018 |
| **achieved** | **0.9699** |

Why blocking misses matches:

| reason | share of misses |
|---|---|
| no shared name token | 43% |
| candidate has no address | 30% |
| shares tokens but ranked out of the candidate list | 25% |

### 3.5 Test submission

| | France | India | US |
|---|---|---|---|
| S1 | 259,452 | 809,986 | 663,106 |
| empty predictions | 5.60% | 6.50% | 5.49% |
| links per S1 | 3.25 | 3.20 | 3.38 |
| S2 share of links | 0.485 | 0.486 | 0.483 |
| candidates per S1 | 49.3 | 43.7 | 38.5 |
| mean rank of a predicted match in its candidate list | **5.92** | 4.01 | 2.80 |
| collision rate before unique assignment | **4.35%** | 0.71% | 0.33% |
| predicted links with zero name-token overlap | 7.2% | **20.8%** | 7.4% |

The pipeline scored 73.7M candidate pairs and predicted 5.68M links. Unique assignment dropped 35,136 of them.

---

## 4. Output audit

Everything below was checked directly against the test files.

- ✔ `matching_results.tsv` and `candidate_pairs.tsv` in the zip are byte-identical (same CRC) to
  `output/output/`.
- ✔ The code in the zip is identical to the working tree.
- ✔ All 1,732,544 S1 rows are present. There are no duplicate IDs, no IDs missing from the test set, and no
  links across countries.
- ✔ Every predicted ID also appears in `candidate_pairs.tsv`, and the validator passes.
- ✔ Each S2/S3 record is claimed by at most one S1, which is consistent with the training labels.
- ✔ Our submission is better than `shreyas_subs/matching_results.tsv`. That file:
  - assigns 199,803 records to more than one S1, up to 133 each (the training labels never do this);
  - leaves 11.8% of France rows and 11.6% of India rows empty, about twice the true singleton rate;
  - agrees exactly with ours on only 14–29% of S1 rows (mean Jaccard 0.53–0.69).

---

## 5. Anomalies

1. **The methodology document in the zip is the blank template.** `Documentation_template.md` at the repo
   root is 0 bytes, so `make_zip` fell back to the organizers' unfilled 2,249-byte template. **Fix this
   before the final package goes in.**
2. **The holdout score does not measure the submitted decision rule.** The test run applies unique
   assignment, but the holdout was scored without it. The holdout also contains no France records.
3. **The test set has more decoys than training.** Pool per S1 is 5.76 on test against 4.67 on train, about
   23% more records that match no S1. Expect lower precision on test than dev or holdout.
4. **France predictions rest on weaker evidence.** Predicted matches rank lower in their candidate lists
   (5.9 against 2.8 for US), and collisions are 10× the US rate, because French names are generic (SARL,
   "& Fils", city names). The closest proxy, US → India transfer, scores only 0.84.
5. **The most confident false positives are near-copies with a changed house number.** Examples: 4806→806,
   841→84, 1470→1477, 48-30→48-31, 2721→272, and zero-padding such as 3905→003905. Name and street are the
   same, so p reaches 1.0.
6. **Many missed matches sit at exactly p = 0.704.** This looks like a flat step in the isotonic calibration
   just below the threshold. Beta calibration scored better: NLL 0.00942 (beta) against 0.00944 (isotonic).
7. **India native-script names mislead the model.** 21% of India links share no name token. Example of a
   false positive: "Sai Trading" matched to "सुपर ट्रेडिंग" (Super Trading) because the address is the same.
8. **Dead or nearly dead features.**
   - The `TXT` family (phone and domain) has exactly zero gain. The domain regex strips `.com` to give
     `bronaughseries`, but nothing compares that to the S1 name "bronaugh series".
   - `region_*` and `postcode_*` gains are around 1e-5, which suggests the extraction is mostly empty.
9. **Report bug.** `errors_gallery.html` shows blank name and address for some candidates that do exist in
   the training files (e.g. `S3-250475004`, `S2-328225771`).
10. **Uncommitted change on the remote machine.** `final_bundle.pkl` was modified at 11:09, after the zip was
    built at 05:54. Re-running `submit` loads that file if it exists, so check which model it holds.

---

## 6. Improvements, by expected gain

| # | target | what to do | why |
|---|---|---|---|
| 1 | package | Fill in `Documentation_template.md` at the repo root and rebuild the zip | The rules require it |
| 2 | matcher FN (−0.0123) | Switch to beta calibration or threshold the raw score. Tune the threshold together with unique assignment, which removes false positives and so permits a lower threshold. Try per-country and per-multiplicity thresholds | Plateau at p=0.704; multiplicity-1 F0.5 is only 0.918 |
| 3 | matcher FP (−0.0068 incl. singletons) | Add features that compare candidates for the same S1: house number exact vs the best competitor, address similarity minus the best other candidate's, name-similarity rank among candidates. Strip leading zeros from house numbers. Try a LambdaRank (ranking) objective grouped by S1 | Decoys differ only by house number |
| 4 | matcher / India | Add a matcher feature on the transliterated or phonetic name (reuse `blocking.skeleton`), a squashed-name equality feature (name with spaces removed vs the domain stem), and an acronym feature (initials of one name equal the other). These replace the dead `TXT` family | 21% of India links share no name token; domain-form and acronym names are common |
| 5 | blocking miss (−0.0111) | Raise `k` or lower `min_score` on the `phon`, `join` and `init` passes for India. Add a pass on house number + street only, for candidates with junk names | India blocking recall 93% vs US 98%; 43% of misses share no name token |
| 6 | France | Add `ei`, `cie`, `ets`/`établissements` and `societe` to `LEGAL_FORMS`. Down-weight generic French tokens. Score unique assignment on the holdout. Check French blocking recall with a hand-labelled sample of about 200 S1 | France is unseen and transfer is weak |
| 7 | validation | Score the holdout with the exact test decision rule (including unique assignment), and report the collision rate at test-scale competition | The reported number currently differs from what was submitted |
| 8 | cleanup | Fix the gallery record lookup. Fix region and postcode extraction, or drop those features | Report bug; dead features |

---

## 7. How to reproduce

```bash
cd code/business_entity_resolution/src
uv run python run.py all --data-dir <student_resource/dataset> --work-dir <work> --team EpochAlypse
# or run stage by stage: prepare → tune → (neural) → evaluate → compare → submit
```

Outputs:

- `<work>/reports/index.html`: every figure and CSV.
- `<work>/output/*.tsv`: the two submission files.
- `<work>/EpochAlypse_submission.zip`: the final package.
