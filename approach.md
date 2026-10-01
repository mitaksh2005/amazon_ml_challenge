# Approach: Business Entity Resolution (Amazon ML Challenge 2026)

This document gives the conclusions from every EDA step and every candidate pipeline that was run, and how
each one led to the next. The **final pipeline is P9, run through `code/business_entity_resolution/src/run_v2.py`**.

Sources:

- EDA: `notebooks/01_nlp_cleaning_eda.ipynb`, `code/business_entity_resolution/src/EDA.ipynb`
- Blocking: `notebooks/02_blocking.ipynb`, `notebooks/03_run_full_pipeline.ipynb`
- Research and search plan: `entity_resolution_research/`
- Pipeline outputs: `output/experiments/runs/*` and `output/reports/*`
- Write-ups: `RESULTS_AND_PIPELINE.md`, `PIPELINES_P5_P9.md`

---

## 1. Problem

- **Task.** For every Source 1 (S1) record, find all Source 2 and Source 3 (S2/S3) records that describe the
  same business. The only fields are `business_name`, `business_address` and `country`.
- **Metric.** Macro F0.5 per S1, which weights precision twice as much as recall. A singleton (an S1 with no
  true matches) scores 1.0 for an empty prediction and 0 for any prediction.
- **Scale.**

  | split | S1 | S2 | S3 | countries |
  |---|---|---|---|---|
  | train | 2,206,821 | 5,034,616 | 5,285,603 | US, India |
  | test | 1,732,544 | 4,887,273 | 5,082,316 | US, India, **France (unseen)** |

---

## 2. EDA conclusions

### 2.1 Structure and ground truth

| finding | number | consequence |
|---|---|---|
| Singleton rate | **5.58%** (US 5.58%, India 5.59%) | Predicting nothing scores 0.056. Every false positive on a singleton costs a full 1.0, so a dedicated singleton gate is needed. |
| Matches per S1 | mean 3.46, max 11. Mode is 3–4 (25% / 23%). Only 5.7% of non-singletons have exactly 1. | Multi-match output; the threshold has to be tuned on macro F0.5, not on pair accuracy. |
| S1 with an S2 / S3 match | 87.0% / 87.9%; 48% of links go to S2 | The two sources contribute equally, so both are treated as one pool. |
| S2/S3 records linked to more than one S1 | **0** of 7,638,365 | The labels are strictly one-to-one. This motivated **unique assignment** (one S1 per S2/S3 record). |
| Distractors | 26.6% of S2 and 25.4% of S3 match no S1 | The pool is full of decoys, which puts more weight on precision. |
| Cross-country links | **0%** | Country is a safe **hard partition** for blocking. |
| Missing S2/S3 addresses | ~3.3% (S1 has none missing). A literal `null` appears in 1.5% of addresses. | Address features must be NaN-aware. Records with no address are a blocking blind spot. |
| Test pool per S1 | 5.76 against 4.67 in train | About 23% more decoys on test, so test precision will be lower than validation precision. |

### 2.2 Raw text and noise (catalogued per source; S1 is the clean reference)

- **Native scripts.** S1 names are always Latin. India S2 names are 23% non-Latin (Devanagari 13%, plus
  Telugu, Tamil, Kannada, Bengali and others) and India S3 names are 13% non-Latin. Transliteration
  (`anyascii`) is therefore mandatory. Native-script **region** names have to be mapped first, so that
  `ಕರ್ನಾಟಕ` becomes `karnataka` and not `krnatk`.
- **Casing.** 64% of S2 addresses are UPPER-case, and some names are all lower-case, so everything is
  lower-cased.
- **Name noise in S2/S3** (train rates): double spaces 11%, trailing brackets 7%, accented letters 6%,
  parenthesised parts 5%, legal form placed at the start 3%, domain-style names (`qureontinto.com`) 3%,
  leading junk (`--`, `>>`, `#`) 2%, phone numbers 0.3%, and DBA/aka prefixes.
- **Address noise.** Components are reordered (region first in about 5%), there are landmark references
  (`near`/`opp`/`behind`, 11% of India S1 addresses) and floor/building cues (25% of India). ZIP/PIN codes
  appear in only 7% and 1.5% of addresses.
- **Legal forms differ by source.** India S1 records are 62.5% `private limited`, but S2/S3 only 38%. In the
  US, 43% of S1 and 50% of S2/S3 have no legal form. Some generated spellings are corrupted (`praivet
  limited`, `pra li`, `piraivet limitet`) and appear among the most common India bigrams. Legal form becomes
  its own feature: a conflict is strong negative evidence, and agreement is weak positive evidence.
- **True pairs are not near-copies.** Only 10.75% of true pairs have the exact same name and 7.24% the exact
  same address. 9.85% contain the same words in a different order. Mean character similarity is 0.78
  (S2 0.774, S3 0.790).

### 2.3 Cleaning and its measured effect

The cleaning pipeline (`cleaning.py`) runs in this order:

1. ftfy
2. NFKC normalisation
3. native-script region mapping
4. `anyascii` transliteration
5. lower-casing
6. name rules: strip domains, phones and tags; `&` becomes "and"; canonicalise legal forms
7. per-country address abbreviation expansion. For example, `St` is "street" in the US and India but
   "saint" in France, and `OR` is Oregon in the US but Odisha in India.

It produces several views of each field: `name_norm`, `name_core`, `legal_form`, `addr_norm`, `region`,
`postcode` and `house_no`.

- **After cleaning there is no non-ASCII text left**, and `name_core` is empty in at most 0.1% of records.
- **Vocabulary overlap with S1 goes up.** For names it rises from 24–30% to 36–41%. For India S2 addresses
  it rises from 8% to 40%.
- **Matches separate better from hard negatives.** Hard negatives share the first name token.

  | similarity | mean on matches | mean on hard non-matches | AUC vs random | AUC vs hard |
  |---|---|---|---|---|
  | name raw (token_sort) | 0.695 | 0.447 | 0.851 | 0.764 |
  | name clean (token_sort) | 0.852 | 0.535 | 0.969 | **0.908** |
  | name_core (token_set) | 0.914 | 0.640 | 0.982 | 0.865 |
  | addr raw (token_sort) | 0.596 | 0.278 | 0.920 | 0.914 |
  | addr clean (token_set) | 0.926 | 0.373 | 0.957 | 0.955 |
  | legal-form conflict | 0.037 | 0.259 | | |

  Cleaning adds about 0.14 AUC to names against hard negatives. The address is the strongest single
  discriminator against hard negatives.
- **India is harder.** India true pairs average 0.82 name similarity, against 0.88 for the US.
- **The lowest-similarity true pairs** are initials (`MS` vs Mallick & Sons, `LA` vs Laxmi Agro), junk or
  renamed names (`Novinovi`, `Deltaquo`) and `s.com` stubs. Only the address can recover these, so blocking
  needs address and initials keys.

### 2.4 Blocking-key signals (on true pairs)

| key shared | match | hard non-match | random non-match |
|---|---|---|---|
| country | 100% | 100% | 100% |
| region | 94.7% | 10.2% | 7.0% |
| any `name_core` token | 86.7% | 100% | 0.9% |
| house number | 36.4% | 0% | 0% |
| postcode | **5.0%** | 0% | 0% |

Conclusions:

- **Country** is the partition.
- **Region** is too lossy to be a hard partition (it would cost 5% of recall). It is used only to scope
  address and name keys.
- **Postcode** is useless as a channel.
- **Name tokens** are the backbone but miss 13% of pairs, so address, number and phonetic keys are needed.
- **House number** is a precise precision signal.

### 2.5 Train vs test drift (France)

- France makes up 15% of test S1 records (259k). It has no labels.
- 62% of French records carry a legal form (SARL, SAS, SCI, EURL). French names are generic (`Club`,
  `Centre`, `& Fils`, city names).
- **Region extraction fails for French S2/S3**: 65–68% have no region, against 3% for India and US.
- **Postcode** is missing from about 99% of records in every country.
- **Proxy for France:** the model is also trained on one country and scored on the other (US → India and
  India → US) to measure how well it transfers.

---

## 3. Blocking conclusions

Method: **multi-pass sparse TF-IDF top-K retrieval**, hard-partitioned by country.

- **Key families.** There are eight: `name`, `phon` (consonant skeleton), `join`, `nreg`, `init`, `addr`,
  `anum` and `house`.
- **Weighting.** IDF is computed per country over the pool, and keys held by more than `cap` records are
  dropped.
- **Passes.** The `all`, `name` and `addr` passes are unioned.

Results:

- **Unioning passes helps.** On 10k S1, `all@20` gives 94.5% pair recall and a 0.980 ceiling. The B0 default,
  `all@20 + name@10 + addr@10`, gives 95.9% recall and a 0.985 ceiling with 27 candidates per S1. Adding the
  name-only and address-only passes recovers pairs that the combined score buries.
- **Dropping one pass (leave-one-out):**

  | pass dropped | Δ ceiling | Δ candidates |
  |---|---|---|
  | `all` | −0.0091 | −9.1 |
  | `addr` | −0.0069 | −4.9 |
  | `name` | −0.0002 | −1.8 |

  The address pass is essential; the name pass is nearly redundant given `all`.
- **The blocking ceiling is the hard limit.** Optuna NSGA-II traded ceiling F0.5 against candidates per S1
  over truncations of a wide retrieval (all@50, name@30, addr@30). The chosen configuration is **Bwide**:
  `all` k=34, `name` k=7, `addr` k=24. It has 41.9 candidates per S1 (p95 57), 96.8% pair recall and a
  ceiling of **0.9889**, against 0.9862 for B0.
- **Recall by slice:** US 97.9–98.0%, **India 93.1–93.5%**.
- **Why true pairs are missed:**

  | reason | share of misses |
  |---|---|
  | no shared name token | 43% |
  | candidate has no address | 30% |
  | shares tokens but ranked out of the candidate list | 25% |
  | region differs | 2% |

---

## 4. Evaluation harness

- **Samples.** Three disjoint S1 samples, stratified by gold multiplicity:
  - dev: 100k S1, 5 folds grouped by S1
  - hpo: 50k S1, seen only by Optuna
  - holdout: 50k S1, scored once
- **Leakage control.** Inside each training fold, entities are split 70 / 15 / 15 into fit / calibrate /
  tune.
- **Pipeline comparison.** Paired entity bootstrap of Δ macro F0.5 against the parent pipeline. A change is
  **admitted** only if the 95% CI lower bound is above 0.
- **Reports.** Every run also reports loss decomposition, slices (country, multiplicity, name frequency) and
  cross-country transfer.

---

## 5. Pipelines and their conclusions

Staged coordinate descent: blocking → features → matcher → negatives → calibration → decision → graph.

All scores are dev macro F0.5 (100k S1, 5-fold OOF).

| pipeline | change | macro F0.5 | Δ vs parent | verdict / conclusion |
|---|---|---|---|---|
| P-empty | predict nothing | 0.0562 | | Equals the singleton rate; the lower bound. |
| P-rule | threshold on blocking score | 0.6811 | | Retrieval score alone is far from enough; 1,509 false positives per 1000 singletons. |
| P0-lr | logistic regression, STR + RET | 0.8060 | −0.088 vs P0 | Rejected. The decision surface is non-linear, so use a GBDT. |
| P0 | LightGBM, name similarity + retrieval | 0.8941 | **+0.213** | Admitted. A learned matcher is the biggest single jump. |
| P1 | + RARE (IDF overlap, name frequency) + NUM (number conflicts) | 0.9481 | **+0.054** | Admitted. Rarity and number conflicts handle chains and branches; singleton false positives drop from 209 to 86 per 1000. |
| P2 | + ADDR, CTX, LEGAL, TXT (full set) | 0.9669 | **+0.019** | Admitted. Address and candidate-set context add the most. |
| P3-xgb / P3-cat | XGBoost / CatBoost | 0.9667 / 0.9660 | −0.0002 / −0.0008 | Rejected. The library makes no difference; LightGBM is fastest. |
| P2-hpo-lgbm / -xgb | Optuna-tuned GBDT | 0.9673 / 0.9673 | +0.0004 / +0.0007 | Not significant, but the tuned LightGBM parameters were kept. Re-evaluated robust score 0.9651. |
| P4 | + E2 negatives (singleton negatives ×3) + singleton gate (policy B) | 0.9673 | ±0.0000 | Neutral on dev, but structurally right for singletons, so it is kept as the base. |
| P4-C | ambiguity/conflict-adjusted threshold | 0.9673 | +0.0000 | Rejected. No gain over policy B. |
| P4-E1 | capped hard negatives | 0.9665 | −0.0008 | Rejected. The matcher needs all retrieved negatives. |
| P6 | + frozen multilingual-E5 cosine feature | 0.9677 | +0.0004 | Rejected. The dense embedding adds almost nothing over the engineered features, at a large compute cost. |
| P8 | + unique assignment (graph H1) | 0.9674 | +0.0001 | Neutral on the small dev sample, where there is little competition. It is still applied on test, where 1.73M S1 compete for the same pool. |
| **P5-Bwide** | P4 on **Bwide** blocking | **0.9699** | **+0.0025 [0.0022, 0.0029]** | **Admitted. Best of 16 and ranked first in all 5 folds** (0.9692–0.9713). More recall pays off once the precision layer is strong. |
| P-oracle | perfect matcher on B0 | 0.9862 | | The blocking ceiling. |

Ablation waterfall: rule 0.681 → P0 +0.213 → P1 +0.054 → P2 +0.019 → HPO +0.0004 → P4 ±0 → Bwide +0.0025 =
**0.9699**.

### 5.1 P5-Bwide in detail

- **Holdout** (50k S1, scored once): **0.9699** [0.9691, 0.9708].
  - Pair precision 0.9926, recall 0.9319.
  - Singleton accuracy 0.972 (29 false positives per 1000 singletons).
- **Slices:**
  - US 0.978, **India 0.957**.
  - Multiplicity-1 entities **0.918**.
  - Most common names 0.950.
- **Transfer:** US → India **0.841**, India → US 0.936. France is likely to score well below dev.
- **Where the remaining F0.5 is lost:**

  | loss | ΔF0.5 |
  |---|---|
  | matcher false negatives | −0.0123 |
  | blocking misses | −0.0111 |
  | matcher false positives | −0.0050 |
  | singleton false positives | −0.0018 |

- **Feature gain share:**

  | family | gain share |
  |---|---|
  | RET | **0.651** |
  | ADDR | 0.150 |
  | STR | 0.088 |
  | RARE | 0.046 |
  | NUM | 0.034 |
  | CTX | 0.019 |
  | LEGAL | 0.012 |
  | TXT | **0.000** (dead) |

- **Singleton gate.** The τ sweep is flat (0.9699 at τ 0–0.15) and falls slowly as τ grows. The gate removes
  few false positives beyond what the threshold already removes.
- **Feature-space / sparse-autoencoder (SAE) analysis.** A sparse autoencoder was trained on the matcher's
  feature space to see what it learns.
  - With k=32 the reconstruction keeps a pair AUC of 0.978, against 0.988 on the original features.
  - Zeroing the top 50 latents only moves the AUC to 0.977, so the signal is spread out rather than held by
    a few shortcut latents.
  - False positives are driven by missing rare name tokens, address-number Jaccard and `score_all`
    (SHAP: FP minus TP).
- **Test run.** 73.7M candidate pairs produced 5.68M links. Unique assignment dropped 35,136 of them. The
  collision rate was **France 4.35%**, India 0.71% and US 0.33%. The validator passed.
- **Reported leaderboard score: 0.948.**

### 5.2 Audit findings on P5-Bwide (these motivated P9)

1. **The most confident false positives are near-copies with a changed house number.** Examples: 4806→806,
   841→84, 1470→1477, 48-30→48-31, and zero-padding such as 3905→003905. Name and street are the same, so p is
   about 1.0. The matcher scored each pair in isolation and had no way to see the exact-number competitor.
2. **Isotonic calibration creates a plateau.** Many missed matches sit at exactly p = 0.704, a flat step
   just below the threshold. Beta calibration had a better NLL (0.00942 against 0.00944).
3. **21% of India links share no name token.** These are native-script names transliterated into a
   different spelling. In the false positives, the address alone carries the decision.
4. **Domain-style names and acronyms have no matcher feature.** For example, `bronaughseries.com` vs
   "Bronaugh Series" and `HGR`. The TXT family is dead.
5. **The holdout did not measure the submitted rule.** It was scored without the unique assignment that was
   applied on test.
6. **The retrieval features dominate (65% of gain).** Their distributions shift with pool density, which is
   5.76 records per S1 on test against 4.67 on train, and with France.

### 5.3 Other approaches explored in the repo (not used)

- **`shreyas_subs/`**: inverted-index blocking + a 12-feature LightGBM + a global threshold. It reported
  0.9665 on its own validation. On test it assigns 199,803 records to more than one S1 (up to 133 each) and
  leaves about 12% of rows empty, twice the true singleton rate. It agrees exactly with P5-Bwide on only
  14–29% of S1 rows. It was dropped.
- **`anshp/`**: a Qwen3-Reranker-0.6B + LoRA cross-encoder over BM25 (SQLite FTS5) candidates. It was built
  and smoke-tested, but trained-model accuracy was never measured. It was not pursued, because of GPU cost
  at 74M test pairs and because P6 showed that dense semantics added little over the engineered features.
- **P7** (GBDT + cross-encoder on the uncertain band) was planned but not run, for the same reason.

---

## 6. Final pipeline: P9 (`run_v2.py`)

P9 is **P5-Bwide with only two changes**:

```python
families = FULL + ("XTRA",)
calibration = "beta"
```

`run_v2.configs()` builds it from P5-Bwide's own `config.json`. Blocking (Bwide), the LightGBM HPO
parameters, E2 negatives, policy B and the samples are all unchanged, so any difference in score comes only
from these two changes. P9a, which has beta calibration only, is the ablation.

### 6.1 End to end

```
raw TSV ─▶ clean (ftfy → NFKC → native-region map → anyascii → name/address normalisation, legal forms)
        ─▶ key cache (8 hashed key families) ─▶ Bwide multi-pass TF-IDF top-K, country-partitioned
        ─▶ pair features: STR, RARE, NUM, ADDR, LEGAL, TXT, RET, CTX  + XTRA (11 pair + 5 candidate-set)
        ─▶ LightGBM (HPO params, E2 negatives) ─▶ beta calibration
        ─▶ policy B: p ≥ t AND singleton gate q ≥ τ ─▶ unique assignment (each S2/S3 → its best S1)
        ─▶ matching_results.tsv + candidate_pairs.tsv ─▶ validator ─▶ zip
```

### 6.2 What P9 adds, and the audit finding each change fixes

**XTRA pair features** (`xfeatures.py`):

| feature | definition | fixes |
|---|---|---|
| `x_join_eq` / `x_join_ratio` / `x_join_partial` | Name tokens joined into one string; equal / ratio / partial ratio | Domain-style names (`bronaughseries.com` = "Bronaugh Series") |
| `x_acr` | One side's joined name equals the other side's initials | Acronyms (`HGR`) |
| `x_skel_tset` | Token-set ratio of consonant skeletons | Transliteration and vowel variation in India names |
| `x_house_eq` | House number with leading zeros stripped; 1 if equal | `003905` = `3905` |
| `x_house_trunc` | 1 if the numbers differ but one is a prefix or suffix of the other | Truncated-number decoys: 4806 vs 806, 2721 vs 272 |
| `x_addrnum_eq` / `x_addrnum_jacc` | Canonical numeric address token sets; equal / Jaccard | 48-30 vs 48-31 |
| `x_native_l` / `x_native_r` | 1 if the raw name contains non-ASCII characters | Tells the model the name similarity comes from a transliteration |

**Candidate-set context** (`add_xctx`). These features let the matcher compare a candidate with the other
candidates of the same S1, instead of scoring each pair alone:

- `x_house_eq_other`: 1 if another candidate of this S1 matches the house number exactly
- `x_addr_c3_gap_best`, `x_street_gap_best`, `x_join_ratio_gap_best`, `x_skel_gap_best`: the gap between
  this candidate's value and the best value among the S1's candidates

**Beta calibration** (Kull et al. 2017). A logistic regression on `[log p, −log(1−p)]`. It is smooth and
monotone, which removes the isotonic plateau at 0.704, and it extrapolates more gracefully when the score
distribution shifts.

**Holdout scored with the test rule.** `submit.holdout(..., unique=True)` scores the holdout with unique
assignment, exactly as on test. `macro_f05_without_unique` is also recorded.

### 6.3 Running it

It reuses P5-Bwide's artifacts: samples, `pairs_dev`/`pairs_holdout`, the HPO parameters, `candidates_test_Bwide`,
`cleaned/` and `blocking_cache/`.

```bash
cd code/business_entity_resolution/src
python run_v2.py xtra     --data-dir <dataset> --work-dir <work> --clean-dir <cleaned> --cache-dir <blocking_cache>
python run_v2.py evaluate ...         # P9a and P9, 5-fold OOF on the same dev folds
python run_v2.py compare  ...         # paired bootstrap vs P5-Bwide → reports/v2_compare.json
python run_v2.py submit --run P9 ...  # refit, holdout with unique assignment, test, validator, zip
```

Without `--run`, `submit` proceeds only if a P9 variant beats P5-Bwide with a paired CI lower bound above 0.
P9 is submitted by choice with `--run P9`.

Runtime on a 28-thread, 15 GB machine:

| step | time / memory |
|---|---|
| XTRA features, dev (7.5M pairs) | 206 s |
| XTRA features, holdout (3.8M pairs) | 121 s |
| OOF fold | about 18 min, peak about 8.4 GB |
| XTRA on test (about 74M pairs) | about +45 min |

### 6.4 Results

| | P5-Bwide | P9a (beta only) | P9 |
|---|---|---|---|
| dev fold 0 | 0.9713 | 0.9708 | stopped before completion |
| dev fold 1 | 0.9695 | 0.9692 | |
| reported leaderboard score | **0.948** | | **0.941** |

### 6.5 Why P9 is the final pipeline despite the lower reported score

- **It depends less on retrieval scores.** P5-Bwide took 65% of its gain from blocking scores, which shift
  with pool density (+23% decoys on test) and with France. P9 adds evidence the matcher computes itself:
  joined names, skeletons and house numbers.
- **It targets the decoy failure mode directly.** House-number canonicalisation and truncation, together
  with candidate-set context, go after the most confident false-positive pattern. That pattern matters more
  as decoys increase, and F0.5 rewards precision.
- **Its calibration is smooth.** There is no step plateau, and it behaves better under distribution shift
  to an unseen country.
- **Its validation matches the submission.** The holdout is scored with the exact test decision rule.

### 6.6 Risks and next steps

- **Beta calibration was slightly negative on dev.** It changed the two finished folds by −0.0005 and
  −0.0003. A full 5-fold P9 OOF run, plus a holdout with unique assignment, is still needed to quantify the
  trade-off on labelled data.
- **French generic names** ("Foot & Fils SARL" vs "Aide & Fils SARL") can collide on the joined and skeleton
  strings, and the XTRA features were never trained on French data. `x_native_*` only helps India.
- **Remaining improvements, by expected gain:**
  1. Tune the threshold jointly with unique assignment, and per country or multiplicity, since
     multiplicity-1 entities score only 0.918.
  2. Add a LambdaRank objective grouped by S1.
  3. Use wider `phon`/`join`/`init` passes for India, and add a house + street pass for candidates with junk
     names (blocking misses cost −0.0111).
  4. Add French legal forms (`ei`, `cie`, `ets`, `societe`) and down-weight generic French tokens.
  5. Fix region and postcode extraction, or drop those features (their gain is about 1e-5).
