# Pipelines P5-Bwide and P9, in depth

This document describes the two final entity-resolution pipelines stage by stage:

- **P5-Bwide**: the best pipeline from the original search. Its best dev score came from 5-fold OOF, and it
  produced `EpochAlypse_submission.zip`.
- **P9**: P5-Bwide hardened with the fixes from the output audit in `RESULTS_AND_PIPELINE.md`. **P9 is the
  preferred model for the final private test run**, because it targets failure modes that the private split
  and the unseen France data are likely to expose.

| | P5-Bwide | P9 |
|---|---|---|
| Blocking | Bwide (Optuna Pareto) | same |
| Features | 8 families, 77 columns | + XTRA family: 11 pair + 5 candidate-set columns (93 total) |
| Matcher | LightGBM, Optuna HPO params | same |
| Negatives | E2 (all, singleton negatives ×3) | same |
| Calibration | isotonic | **beta** |
| Decision | policy B (singleton gate + threshold) | same |
| Unique assignment | test only | test **and holdout** (holdout scored with the exact test rule) |
| Code | `src/` (original) | `src/xfeatures.py`, `src/run_v2.py`, hooks in `pipelines.py` and `submit.py` |
| Reported score | 0.948 | 0.941 |

The reported scores were given by the team. P9 is still preferred for the private run because of its
robustness (§5).

---

## 1. Problem recap

- **Task.** For every Source 1 (S1) record, return every S2/S3 record that describes the same business.
  The only fields are `business_name`, `business_address` and `country`.
- **Metric.** Macro F0.5 over S1 records. Singletons score 1.0 for an empty prediction and 0 for any
  prediction.
- **Train.** 2.21M S1 and 10.3M S2/S3 records (US and India).
  - The singleton rate is 5.6%, with 3.46 links per S1.
  - The labels are strictly one-to-one: each S2/S3 record belongs to at most one S1.
- **Test.** 1.73M S1 and 9.97M S2/S3 records.
  - It adds **France**, which is unseen in training.
  - It has 5.76 pool records per S1, against 4.67 in train, so about 23% more decoys.

---

## 2. Shared stages (identical in both pipelines)

Entry point: `code/business_entity_resolution/src/run.py`. Every stage is resumable, and its outputs live
under `--work-dir`.

### 2.1 Cleaning (`cleaning.py`)

- **Text repair.** Mojibake repair, then ftfy, then NFKC, then `anyascii` transliteration of native scripts
  (Devanagari, Telugu, Bengali, …). Native-script region names are mapped first.
- **Names:**
  1. Lower-case.
  2. Turn `domain.com` into `domain`, and drop phone numbers and `(india)`-style tags.
  3. Turn `&` into "and", `+` into "plus", and possessives like `'s` into `s`; collapse `l.l.c` into `llc`.
  4. Strip punctuation.
  5. Normalise transliterated "private limited" shapes (`praivet limited`, `pra li`).
  6. Extract the legal form into a canonical set (`LEGAL_FORMS`: llc, inc, corp, private limited, limited,
     llp, sarl, sas, sa, sci, …).
  7. Emit two versions: `name_norm` (legal forms kept as tokens) and `name_core` (legal forms and stopwords
     removed).
- **Addresses.**
  - Abbreviations are expanded per country: US (compass points), India (English) and France
    (`st` = saint, `bd` = boulevard, `r` = rue, …).
  - Region, postcode, house number and street are extracted. French departments are mapped to their
    regions, and Indian states split in 2000/2014 are grouped with the state they split from.
- **Country** is an open set of labels, so France flows through like any other country.

### 2.2 Blocking (`blocking.py`, `data.py`)

- **Method.** Multi-pass sparse TF-IDF top-K retrieval, hard-partitioned by country (100% of true pairs
  share a country).
- **Key families.** Each record emits hashed keys:
  - `name`: name tokens, plus joined bigrams and trigrams.
  - `phon`: consonant skeleton (robust to typos and transliteration).
  - `join`: the whole name joined into one string.
  - `nreg`: name tokens scoped to the region.
  - `init`: initials.
  - `addr`: address tokens scoped to the region.
  - `anum`: numeric address tokens, country-wide.
  - `house`: house number + first street word.
- **Weighting.** IDF is computed per country over the S2+S3 pool. Keys held by more than `cap` records are
  dropped from retrieval, but still count in the vector norm.
- **Passes.**
  - `all`: every family.
  - `name`: name-side families.
  - `addr`: address-side families.

  Each pass streams the pool in chunks and keeps a running top-K per S1. The passes are then unioned.
- **Wide set.** Sampled train S1 records are retrieved wide (all@50, name@30, addr@30) against the **full
  train pool**. Every blocking configuration is an exact truncation of this set (`data.truncate`), so
  configurations can be compared without re-retrieving.
- **Bwide** (Optuna NSGA-II, blocking ceiling F0.5 against candidates per S1):

  | pass | k | min_score |
  |---|---|---|
  | all | 34 | 0.1412 |
  | name | 7 | 0.2681 |
  | addr | 24 | 0.1348 |

  On dev this gives 41.9 candidates per S1 (p95 57), a candidate pair recall of 0.968, and a ceiling F0.5 of
  0.9889.

### 2.3 Samples (`prepare.py`)

Disjoint train samples, stratified by gold multiplicity:

- **dev**: 100k S1 in 5 folds. Every pipeline is scored on these OOF, with the same entities and folds.
- **hpo**: 50k S1, seen only by Optuna.
- **holdout**: 50k S1, scored once for the final pipeline.

### 2.4 Base pair features (`features.py`, `pipelines.add_ret_ctx`)

| family | columns (examples) | P5 gain share |
|---|---|---|
| RET | blocking `score`, `score_<pass>`, `rr_<pass>` (reciprocal rank), `rrf` (reciprocal-rank fusion), `n_passes` | 0.651 |
| ADDR | `addr_ratio`, `addr_tset`, `addr_char3_cos`, `addr_wjacc`, `street_ratio`, `region_eq`/`conflict`, `postcode_eq`/`conflict` | 0.150 |
| STR | `name_ratio`, Levenshtein, Damerau-Levenshtein, Jaro(-Winkler), token-set/sort, partial, prefix, `name_char3_cos`, `is_s3` | 0.088 |
| RARE | IDF-weighted Jaccard, shared/missing-token IDF, exact-name pool frequency | 0.046 |
| NUM | number sets in name and address: equality, conflict, house number, `conflict_count` | 0.034 |
| CTX | per-S1 candidate set: rank, gap to best score, top-2 gap, z-score, name/address gaps | 0.019 |
| LEGAL | legal-form state, equality, conflict | 0.012 |
| TXT | phone and domain equality/conflict | 0.000 |

### 2.5 Leakage-safe fitting (`pipelines.fit_bundle`)

Inside each training fold, entities are split **70 / 15 / 15** into fit / calibrate / tune:

- **fit**: trains the matcher. 10% of the fit entities are held out for early stopping.
- **calibrate**: fits the calibrator and the singleton model on scores from a model that never saw those
  labels.
- **tune**: fits the decision policy (thresholds) on calibrated scores.

### 2.6 Matcher and decision (`models.py`)

- **LightGBM**, with the Optuna HPO parameters:

  | parameter | value |
  |---|---|
  | num_leaves | 215 |
  | learning_rate | 0.01617 |
  | min_data_in_leaf | 237 |
  | feature_fraction | 0.661 |
  | bagging_fraction | 0.923 |
  | bagging_freq | 1 |
  | lambda_l1 / lambda_l2 | 0.0496 / 0.383 |
  | max_bin | 127 |
  | pos_weight | 4.675 |

- **Negatives (E2).** All candidate negatives are used, and negatives belonging to a singleton S1 get
  weight 3.
- **Singleton model.** A small LightGBM gives q = P(the S1 has any match). Its inputs summarise the S1's
  calibrated candidate scores (max, second, gap, sum, counts above 0.5/0.8, name frequency).
- **Policy B.** A pair is predicted when `p ≥ t` **and** `q ≥ τ`. For each τ on a quantile grid, the exact
  macro-F0.5-optimal t is found on the tune split, and the best (τ, t) pair is kept.
- **Unique assignment (graph H1).** An S2/S3 record predicted for several S1 records is kept only for its
  highest-p S1. This matches the one-to-one training labels, and it matters at test scale, where 1.73M S1
  records compete for the same pool.

### 2.7 Submission (`submit.py`)

1. Refit the chosen pipeline on the whole dev sample. The bundle is cached as `final_bundle.pkl`.
2. Score the holdout once.
3. Retrieve test candidates with the pipeline's blocking configuration. They are cached in
   `artifacts/candidates_test_<cfg>/`.
4. Score the test set in batches of 100k S1 records. Each batch gets its own record stores and features,
   with IDF fitted once on the whole test pool.
5. Apply the policy, then unique assignment across all S1.
6. Write `matching_results.tsv` and `candidate_pairs.tsv`, run the official validator, and build the zip.

---

## 3. P5-Bwide

`experiments/runs/P5-Bwide/config.json`: `blocking=Bwide`, `families=FULL` (STR, RARE, NUM, ADDR, LEGAL, TXT,
RET, CTX), `matcher=lgbm`, `params=hpo`, `negatives=E2`, `singleton_weight=3`, `calibration=isotonic`,
`policy=B`, `graph=false`.

### Results

| measure | value |
|---|---|
| dev OOF macro F0.5 (100k S1) | **0.9699** [0.9692, 0.9705]; best of 16 pipelines in all 5 folds |
| holdout (50k S1, no unique assignment) | 0.9699 [0.9691, 0.9708] |
| pair precision / recall (holdout) | 0.9926 / 0.9319 |
| singleton accuracy | 0.972 |
| US / India (dev) | 0.978 / 0.957 |
| US → India transfer (France proxy) | 0.841 |
| reported score | **0.948** |
| test run | 73.7M candidate pairs, 5.68M links; unique assignment removed 35,136 |

### Known weaknesses (from the audit)

1. The top false positives are decoys that differ only in the house number (4806→806, 1470→1477,
   48-30→48-31), with the same name and street, so p is about 1.0.
2. Isotonic calibration creates a plateau: many missed matches sit at exactly p = 0.704.
3. India native-script names: 21% of India links share no name token, and the address alone drives some
   false positives ("Sai Trading" vs "सुपर ट्रेडिंग").
4. Domain-style names ("bronaughseries.com") and acronyms ("HGR") get no matcher feature, and the TXT family
   has zero gain.
5. The holdout was scored without the unique assignment used on test, so it didn't measure the submitted
   rule.

---

## 4. P9

`run_v2.configs()` builds P9 from P5-Bwide's own `config.json` and changes only:
`families = FULL + ("XTRA",)` and `calibration = "beta"`. Blocking, parameters, negatives, policy and
samples are unchanged, so any difference comes from the changes below.

### 4.1 XTRA pair features (`src/xfeatures.py`, `XTRA_COLS`)

Record-level strings are built once per record (`XStore`), then compared per pair (vectorised with
rapidfuzz `cpdist`; about 28k pairs/s on the team laptop).

| column | definition | fixes |
|---|---|---|
| `x_join_eq` | Name tokens (after `blocking.name_token` undoes leetspeak) joined into one string; 1 if equal | domain-style names: `bronaughseries.com` = "Bronaugh Series" |
| `x_join_ratio` | `fuzz.ratio` of the joined names | same, with typos |
| `x_join_partial` | `fuzz.partial_ratio` of the joined names | one name contained in the other |
| `x_acr` | 1 if one side's joined name equals the other side's initials (at least 2 tokens) | "HGR" = "Holloman, Groseclose and Ramirez" |
| `x_skel_tset` | token-set ratio of consonant skeletons (`blocking.skeleton`) | transliterated names: vowel and spelling variation |
| `x_house_eq` | house number with leading zeros stripped (`^0*(\d{1,7})`); 1 if equal | 003905 = 3905 |
| `x_house_trunc` | 1 if the house numbers differ but one is a prefix or suffix of the other | decoys: 4806 vs 806, 841 vs 84, 2721 vs 272 |
| `x_addrnum_eq` | canonical numeric address token sets (`blocking.addr_tokens`: 0067→67, 12th→12, ranges split); 1 if equal | 48-30 vs 48-31 (the house regex keeps only 48) |
| `x_addrnum_jacc` | Jaccard of those sets | partial number agreement |
| `x_native_l` / `x_native_r` | 1 if the raw name contains non-ASCII characters | tells the model that the name similarity is a transliteration |

Features are NaN when they can't be compared (either side empty), as in `features.py`. Coverage on dev:

- `x_house_eq` is set for 37% of pairs.
- `x_addrnum_eq` is set for 63%.
- `x_native_r` = 1 for 11% of candidates. S1 names are never in native script.

### 4.2 Candidate-set context (`xfeatures.add_xctx`, `XCTX_COLS`)

These are computed per S1 over its candidate list, after `add_ret_ctx` sorts the rows by entity.

| column | definition | purpose |
|---|---|---|
| `x_house_eq_other` | 1 if **another** candidate of this S1 matches the house number exactly | a truncated-number decoy loses to the exact-number record |
| `x_addr_c3_gap_best` | `addr_char3_cos` − its maximum within the S1 | relative address evidence |
| `x_street_gap_best` | `street_ratio` − its maximum within the S1 | same, for the street |
| `x_join_ratio_gap_best` | `x_join_ratio` − its maximum within the S1 | relative joined-name evidence |
| `x_skel_gap_best` | `x_skel_tset` − its maximum within the S1 | relative phonetic-name evidence |

The matcher previously scored each pair on its own. These features let it compare a candidate with its
competitors, which is how the near-copy decoys can be told apart.

### 4.3 Beta calibration

`models.Calibrator("beta")` fits a logistic regression on `[log p, −log(1−p)]` (Kull et al. 2017):

- It is smooth and monotone, with no step plateaus, so the policy threshold is not stuck on a 0.704 block of
  tied pairs.
- On P5-Bwide's own OOF scores, beta already had the best NLL: 0.00942, against 0.00944 for isotonic.

### 4.4 Holdout scored with the test rule (`submit.holdout(..., unique=True)`)

The holdout is now scored **with** unique assignment, exactly as test predictions are made.
`holdout.json` also records `macro_f05_without_unique`, so the effect of the one-to-one step is visible.

### 4.5 Hooks in the original code

- **`pipelines.Workspace.wide_pairs`** merges `artifacts/xtra_<role>.parquet` when it exists. It asserts
  that rows are aligned with `pairs_<role>.parquet`, so the base pair files are never rewritten.
- **`pipelines.add_ret_ctx`** calls `X.add_xctx` when XTRA columns are present.
- **`pipelines.feature_columns`** resolves the family `"XTRA"` to `XTRA_COLS + XCTX_COLS`.
- **`submit.predict_test`** builds `XStore(s1)` and `XStore(pool)` once, then appends XTRA features to each
  scored batch. This happens only when `"XTRA"` is in the pipeline's families, so P5-Bwide's behaviour is
  unchanged.

### 4.6 Running P9

It reuses P5-Bwide's prepared artifacts: `samples`, `pairs_dev`, `pairs_holdout`, the HPO params, the
blocking configs, `candidates_test_Bwide`, and the `cleaned/` and `blocking_cache/` directories.

```bash
cd code/business_entity_resolution/src
python run_v2.py xtra     --data-dir <dataset> --work-dir <work> --clean-dir <cleaned> --cache-dir <blocking_cache>
python run_v2.py evaluate ...   # P9a (beta only, ablation) and P9, 5-fold OOF on the same dev folds
python run_v2.py compare  ...   # paired bootstrap vs P5-Bwide → reports/v2_compare.json
python run_v2.py submit --run P9 ...   # refit, holdout (unique assignment), test, validator, zip
```

Without `--run`, `submit` only proceeds if a P9 variant beats P5-Bwide with a paired 95% CI lower bound
above 0. Pass `--run P9` to submit it by choice.

### 4.7 Runtime on the team laptop

28 threads and 15 GB RAM:

| step | time / memory |
|---|---|
| XTRA features | 7.5M dev pairs in 206 s; 3.8M holdout pairs in 121 s |
| OOF fold | about 18 min (the first fold is about 48 min, including loading), peak about 8.4 GB |
| test run | about 74M pairs; XTRA adds roughly 45 min at 28k pairs/s |

### 4.8 Results

| | P5-Bwide | P9a (beta only) | P9 |
|---|---|---|---|
| dev fold 0 | 0.9713 | 0.9708 | stopped before completion |
| dev fold 1 | 0.9695 | 0.9692 | |
| reported score | **0.948** | | **0.941** |

---

## 5. Why P9 is preferred for the private test run

Despite the lower reported score, the team chose P9 for robustness:

- **It relies less on retrieval scores.** P5-Bwide takes 65% of its gain from blocking scores and ranks. Those
  distributions shift when the pool density changes (5.8 pool records per S1 on test against 4.7 on train)
  and for France. P9 adds evidence the matcher computes itself (joined names, skeletons, house numbers), which
  depends less on the pool.
- **It targets decoys directly.** Candidate-set context and house-number truncation go after the most
  confident false-positive pattern. That pattern matters more as decoys increase, and F0.5 weights precision.
- **Its calibration is smooth.** A monotone parametric calibrator extrapolates more gracefully than isotonic
  steps when the score distribution shifts to an unseen country.
- **Its validation matches the submission.** The holdout now measures the exact test rule.

### Risks to watch

- **French generic names.** They can collide on joined and skeleton strings ("Foot & Fils SARL" against
  "Aide & Fils SARL"). France has no labels, so these features were never trained on French data.
- **Native-script flag.** `x_native_*` is always 0 for France and US S1 records, so it only helps India
  candidates.
- **Beta calibration on dev.** On the two finished folds it was neutral to slightly negative (−0.0005,
  −0.0003).
- **Next step.** A full P9 OOF run, plus a holdout run with unique assignment, would quantify the trade-off on
  labelled data before the private leaderboard is revealed.
