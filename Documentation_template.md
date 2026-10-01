# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EpochAlypse  
**Team Members:** [List all team members]  
**Submission Date:** 2026-10-02

---

## 1. Executive Summary

We clean all three sources into several normalised views of each field: transliterated, legal-form-aware and
abbreviation-expanded. Candidates come from a **country-partitioned, multi-pass sparse TF-IDF retrieval**
over eight hashed key families: name, phonetic skeleton, joined name, region-scoped name, initials, address,
address number and house number.

Each candidate pair is scored by a **LightGBM matcher** on 77 engineered features, then calibrated with
isotonic regression. A pair is predicted only if it clears a threshold chosen with an **exact macro-F0.5
optimiser**, *and* a **per-entity singleton gate** says the S1 has any match at all. Finally, each S2/S3
record is assigned to at most one S1 (**unique assignment**), because the training labels are one-to-one.

The submitted pipeline, **P5-Bwide**, won a 16-pipeline search scored out-of-fold. Its dev OOF macro F0.5 is
**0.9699**, and the holdout (scored once) gives the same **0.9699**.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the full train set (2.21M S1; 10.3M S2/S3):

- **Singletons.** 5.58% of S1 records have no match. Predicting nothing scores only 0.056, but one wrong
  link on a singleton costs a full 1.0, so precision on singletons needs its own model.
- **One-to-one labels.** Each S1 has 3.46 matches on average (maximum 11). **No** S2/S3 record is linked to
  more than one S1, and about 26% of S2/S3 records are decoys that match nothing.
- **Country.** 0% of links cross countries, so country is a safe hard partition. Region agrees on 94.7% of
  true pairs, which is too lossy for a partition, so it only scopes keys. Postcode agrees on only 5%, so it
  is not used as a key.
- **Native scripts.** S1 names are always Latin. India S2 names are 23% non-Latin (Devanagari, Telugu,
  Tamil, Kannada and others) and India S3 names 13%.
- **Name noise.** Double spaces, trailing brackets, accents, legal forms moved to the front, domain-style
  names (`qureontinto.com`), leading junk (`--`, `>>`), phone numbers, DBA/aka prefixes, and corrupted legal
  forms (`praivet limited`, `pra li`).
- **Address noise.** Upper-casing (64% of S2), reordered components, landmarks (`near`, `opp`), a literal
  `null` component, and missing addresses (about 3.3% of S2/S3).
- **True pairs are rarely near-copies.** Only 10.75% have the exact same name and 7.24% the exact same
  address.
- **Cleaning helps measurably.** Name-similarity AUC against hard negatives (same first name token) rises
  from 0.76 on raw text to 0.91 after cleaning. The cleaned address reaches 0.955.
- **The hardest true pairs** are initials (`MS` vs Mallick & Sons), renamed or junk names (`Novinovi`) and
  transliterations. Only address or initials keys can recover them.
- **Train/test drift.** Test adds **France** (15% of test S1, no labels) and has 5.76 pool records per S1,
  against 4.67 in train, so about 23% more decoys.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (GBDT) + calibrated set-decision layer + one-to-one assignment.

**Core Innovation:**

1. **Multi-pass blocking tuned on the competition metric.** Blocking is multi-pass sparse retrieval whose
   configuration was chosen by multi-objective Optuna (NSGA-II). The objective trades the *blocking ceiling
   F0.5*, which is the score a perfect matcher would get, against the number of candidates per S1.
2. **A leakage-safe decision layer.** The fit / calibrate / tune data are separated (70 / 15 / 15 by entity
   inside every fold). The decision combines an exact macro-F0.5 threshold with a separate singleton gate.
3. **A staged pipeline search with significance gates.** Each change was compared against its parent with a
   paired entity bootstrap, and was admitted only when the 95% CI lower bound of Δ F0.5 was above 0.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used.**
  - **Hard partition:** `country`. It is an open set, so France is handled like any other country.
  - **Eight hashed key families**, built from the cleaned record:

    | family | content |
    |---|---|
    | `name` | cleaned name tokens, plus joined bigrams and trigrams |
    | `phon` | consonant skeleton, robust to typos and transliteration |
    | `join` | the whole name as one string, for domain-style names |
    | `nreg` | name tokens scoped to the region |
    | `init` | initials |
    | `addr` | address tokens scoped to the region |
    | `anum` | canonical address numbers: `0067`→`67`, `12th`→`12`, ranges split |
    | `house` | house number + street |

  - **Weighting.** Keys get **TF-IDF** weights computed per country over the S2+S3 pool. Very frequent
    keys, held by more than `cap` records, are dropped from retrieval.
  - **Retrieval.** Three cosine top-K passes (`all`, `name`, `addr`) are computed as streamed sparse matrix
    products and unioned. The chosen configuration, **Bwide**:

    | pass | k | min_score |
    |---|---|---|
    | `all` | 34 | 0.141 |
    | `name` | 7 | 0.268 |
    | `addr` | 24 | 0.135 |

- **Candidate pairs generated.** 73,715,243 test pairs, about 42.5 per S1 (p95 57). Only 2 of 1,732,544 S1
  records have no candidate.
- **How we ensured true matches were not lost.**
  - **The configuration was chosen on recall.** Every configuration was scored against the ground truth on
    a labelled train sample, queried against the **full** 10.3M-record train pool so the decoy load was
    realistic. The metrics were pair recall and ceiling F0.5. Bwide reaches **96.8% pair recall** and a
    **0.9889 ceiling**, against 95.9% and 0.986 for the narrower default (27 candidates per S1).
  - **Unioning passes recovers what the combined score buries.** Leave-one-pass-out costs −0.009 of
    ceiling for `all` and −0.007 for `addr`, so the address pass is essential for junk or initials-only
    names.
  - **Remaining misses were analysed.** 43% share no name token, 30% are candidates with no address, and
    25% are ranked out of the list. Recall is 98% for the US and 93% for India.

---

## 4. Matching Model

**Features used** (77 columns in 8 families; gain share in brackets):

- **Name features (STR, 0.088).** Levenshtein, Damerau-Levenshtein, Jaro and Jaro-Winkler ratios;
  token-set, token-sort, partial and prefix ratios; character-3-gram cosine on `name_norm` / `name_core`.
- **Rarity (RARE, 0.046).** IDF-weighted Jaccard, IDF of shared and missing tokens (rare missing tokens
  strongly signal a different business), and how often the exact name appears in the pool.
- **Address features (ADDR, 0.150).** Ratio, token-set, character-3-gram cosine and IDF-weighted Jaccard on
  `addr_norm`; street ratio; region and postcode equality or conflict.
- **Numbers (NUM, 0.034).** Number sets in the name and address: equality, conflict and house number. These
  separate branches such as `Hotel21` vs `Hotel12`.
- **Legal form (LEGAL, 0.012).** Canonical legal-form state, equality and conflict (e.g. `llc` vs
  `private limited`).
- **Retrieval (RET, 0.651).** The blocking score per pass, reciprocal rank per pass, reciprocal-rank fusion,
  and the number of passes that retrieved the pair.
- **Candidate-set context (CTX, 0.019).** The candidate's rank within its S1, gap to the best score, top-2
  gap, z-score, and the name and address gap to the best candidate.
- **Other (TXT, 0.000).** Phone and domain equality or conflict extracted from the text. This family ended
  up with zero gain.

**Model type:** LightGBM binary classifier with Optuna-tuned parameters (TPE sampler, median pruner, top-5
trials re-scored on fresh splits):

| parameter | value |
|---|---|
| num_leaves | 215 |
| learning_rate | 0.0162 |
| min_data_in_leaf | 237 |
| feature_fraction | 0.661 |
| bagging_fraction | 0.923 |
| lambda_l1 / lambda_l2 | 0.050 / 0.383 |
| max_bin | 127 |
| pos_weight | 4.68 |

The matcher trains on every retrieved negative, and negatives of singleton S1 records get 3× weight. Scores
are then calibrated with isotonic regression.

**Threshold selection method:** an exact macro-F0.5 optimiser over calibrated scores (O(M log M),
unit-tested against brute force), run on a separate tune split. **Policy B** predicts a pair only if
`p ≥ t` **and** `q ≥ τ`, where `q` = P(the S1 has any match) comes from a small per-entity LightGBM over
its candidate-score summary (max, second, gap, sum, counts above 0.5/0.8, name frequency). `(τ, t)` is
chosen jointly to maximise macro F0.5. On test, **unique assignment** then keeps each S2/S3 record only for
its highest-probability S1.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Dev, 100k S1, 5-fold OOF: **0.9699** [95% CI 0.9692–0.9705]. It ranked first of 16 pipelines in every
    fold (0.9692–0.9713).
  - Holdout, 50k S1, scored once: **0.9699** [0.9691–0.9708].
  - Holdout details: pair precision 0.9926, pair recall 0.9319, singleton accuracy 0.972. The blocking
    ceiling is 0.9887.
  - Reported leaderboard score: **0.948**.
  - By slice: US 0.978, India 0.957. Entities with exactly one match score 0.918.
  - Cross-country transfer, as a proxy for unseen France: train US → eval India 0.841; India → US 0.936.

- **How we got there** (dev OOF, each step against its parent):

  | step | macro F0.5 |
  |---|---|
  | threshold on retrieval score | 0.681 |
  | + LightGBM on names and retrieval | 0.894 |
  | + rarity and number conflicts | 0.948 |
  | + address, context, legal form | 0.967 |
  | + tuning, singleton gate | 0.967 |
  | + Bwide blocking | **0.970** |

  Rejected because the change was not significant or was negative:
  - XGBoost / CatBoost: no gain over LightGBM.
  - Capped hard negatives: −0.0008.
  - Ambiguity-adjusted threshold: +0.0000.
  - Frozen multilingual-E5 cosine feature: +0.0004, at a large compute cost.
  - Unique assignment on dev: +0.0001. It is still used on test, where competition is far higher.

- **Loss decomposition** (OOF):

  | loss | ΔF0.5 |
  |---|---|
  | matcher false negatives | −0.0123 |
  | blocking misses | −0.0111 |
  | matcher false positives | −0.0050 |
  | singleton false positives | −0.0018 |

- **Common false positives (wrong merges).**
  - **Near-copy decoys that differ only in the house number** (4806 vs 806, 1470 vs 1477, 48-30 vs 48-31,
    3905 vs 003905). Name and street are the same, so the model is very confident.
  - **Generic names** (most-common-name quintile: 0.950) and **chain branches**.
  - **India records where only the address agrees** and the name is a different business written in native
    script.
  - **France** has a 4.35% collision rate before unique assignment (US 0.33%), because French names are
    generic: SARL, "& Fils", city names.

- **Common false negatives (missed matches).**
  - **Blocking misses:** no shared name token (initials, renamed or junk names, heavy transliteration such
    as `phorcyun imphotek` vs *fortune infotech*), or a candidate with no address whose name is shared by
    many businesses.
  - **Matcher misses:** many sit on an isotonic calibration plateau just below the threshold. India
    multiplicity-1 entities are the weakest slice.

---

## 6. Conclusion

A careful cleaning layer plus multi-pass sparse retrieval, tuned directly on the blocking ceiling F0.5, gets
about 97% of true pairs into the candidate set at about 42 candidates per S1. A gradient-boosted matcher on
engineered rarity, number, address and context features, followed by a leakage-safe threshold, a singleton
gate and one-to-one assignment, converts that into **0.970 macro F0.5** on held-out data.

The largest gains came from rarity and number-conflict features and from wider, metric-tuned blocking. The
library choice, dense embeddings and fancier thresholds added little.

The main remaining risks are near-copy decoys, transliterated India names and transfer to the unseen French
data. A follow-up variant targeting these (P9: joined-name, acronym, skeleton and house-number-truncation
features, candidate-set context, beta calibration) is included in `src/run_v2.py`. It was not the submitted
pipeline.

---

## Appendix

### A. Code Artefacts

Everything is under `code/business_entity_resolution/`. `README.md` has exact reproduction steps, and
`requirements.txt` pins every dependency (Python 3.12).

```
src/
  run.py        single entry point: prepare → tune → evaluate → compare → submit (every stage resumable)
  cleaning.py   text repair, transliteration, name/address normalisation, legal forms
  blocking.py   key families, per-country TF-IDF, multi-pass streamed top-K retrieval
  data.py       paths, samples/folds, wide candidates, blocking-config truncation
  prepare.py    cleaning + key cache + labelled dev/hpo/holdout pairs
  features.py   pair features (STR, RARE, NUM, ADDR, LEGAL, TXT)
  models.py     matchers, calibrators, singleton model, set policies, unique assignment
  pipelines.py  pipeline configs, leakage-safe fitting, OOF evaluation
  submit.py     refit, holdout, batched test scoring, TSV writing, validator, zip
  er_eval/ er_tune/ er_viz/   metric + exact threshold optimiser, Optuna studies, report figures
  neural.py, run_v2.py, xfeatures.py   optional experiments (E5 feature; P9 follow-up), not used by P5-Bwide
tests/test_metrics.py
```

Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`:

```bash
cd code/business_entity_resolution/src
python run.py all --data-dir <student_resource/dataset> --work-dir <work> --team EpochAlypse --run P5-Bwide
```

### B. Additional Results

Full dev leaderboard (100k S1, 5-fold OOF):

| pipeline | macro F0.5 | admitted |
|---|---|---|
| P-empty (predict nothing) | 0.0562 | |
| P-rule (retrieval-score threshold) | 0.6811 | |
| P0-lr (logistic regression) | 0.8060 | ✘ |
| P0 (LightGBM, names + retrieval) | 0.8941 | ✔ |
| P1 (+ rarity, numbers) | 0.9481 | ✔ |
| P2 (full features) | 0.9669 | ✔ |
| P3-xgb / P3-cat | 0.9667 / 0.9660 | ✘ |
| P2-hpo-lgbm / P2-hpo-xgb | 0.9673 / 0.9673 | ✘ |
| P4 (singleton weighting + gate) | 0.9673 | ✘ |
| P4-C / P4-E1 | 0.9673 / 0.9665 | ✘ |
| P6 (+ multilingual-E5) | 0.9677 | ✘ |
| P8 (+ unique assignment) | 0.9674 | ✘ |
| **P5-Bwide** | **0.9699** | ✔ |
| P-oracle (blocking ceiling of the default blocking) | 0.9862 | |

The figures behind every number here are produced by `run.py compare` in `<work>/reports/index.html`:
blocking Pareto front, PR curves, SHAP, calibration, threshold curves, error gallery and cross-country
transfer.
