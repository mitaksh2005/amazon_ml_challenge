# Candidate-pipeline search and analysis plan

Companion to `research_report.md`. That report proposes an architecture and a validation protocol;
this document turns it into (1) a concrete, staged search over candidate pipelines that fits *this*
dataset and repo, and (2) the analysis/plotting work needed to understand and compare what each
pipeline learns — including sparse-autoencoder (SAE) views of the neural models' feature spaces.

Status: plan only. Nothing below has been run; every number quoted from the repo is marked with its
source.

---

## 1. What the report says, and what changes for this dataset

### 1.1 Report take-aways we keep

| Report recommendation (section) | Keep as | Why it still holds here |
|---|---|---|
| Sparse-first retrieval, dense only as a measured rescue channel (§1, §4.2, gate 3) | Blocking search axis B | `blocking.py` already does multi-pass sparse TF-IDF; dense must *prove* marginal recall |
| Numeric / conflict features (house, name-number, sector) (§4.1, features 17–34) | Feature family F-NUM | Branch disambiguation (`Hotel21` vs `Hotel12`) is where F0.5 precision is won |
| Token rarity (IDF, weighted Jaccard, SoftTF-IDF) (§5.1, §6.1) | Feature family F-RARE | Common names (`n|services`) are the main false-merge source |
| Train on *retrieved* negatives + singleton hard cases (§4.4) | Negative-curriculum axis E | Candidate parquet already carries labels for every retrieved pair |
| CPU GBDT primary matcher; XGBoost/CatBoost as direct competitors (§5.2) | Matcher axis D | Cheap, strong, explainable |
| Calibrate OOF, then choose a set policy A/B/C incl. singleton gate (§6.2–6.3) | Decision axes F, G | Singletons score 1.0/0.0 — the single biggest lever on macro F0.5 |
| Exact global-threshold optimiser (§6.4 pseudocode) | `metrics.exact_macro_threshold` | Needed by every pipeline; implement once |
| Graph refinement only after a held-out gain (§6.5, gate 5) | Axis H, last | Default off |
| 5 outer folds grouped by S1 id, paired entity bootstrap (§7.1–7.2) | Evaluation harness (§3) | Labels are complete → protocol usable as written |
| Ablation ledger with ΔF0.5 per parent (§7.3) | Experiment registry (§3.4) | Drives all comparison plots |

### 1.2 Report assumptions that do **not** match this data

| Report assumes | Reality in this repo | Consequence for the plan |
|---|---|---|
| Rich schema: phone, URL/domain, lat/lon, category, alt names (§4.1) | Only `business_name`, `business_address`, `country` | Features 30, 35–46 (phone, domain, geo, category) are **dropped** unless extracted from free text. `cleaning.py` already strips phones/domains from names (`PHONE_RE`, `DOMAIN_RE`) — add a *text-extracted* phone/domain column and test it as a small feature family, not as a retrieval channel |
| libpostal/libphonenumber parsing | Not installed; rule-based `clean_address` gives `addr_norm`, `region`, `postcode`, `house_no` | Parser choice is an ablation (rule-based vs libpostal), not a prerequisite |
| SageMaker jobs, $200 ledger (§9–11) | Local runs (blocking is designed for an 8 GB laptop) | Replace dollar ledger with **wall-time + peak RAM** ledger; keep the report's gates but measured in minutes |
| Postcode is a useful anchor (channel table §4.2) | Same postcode on only **5%** of true pairs (notebook 02, from notebook 01 full run) | Postcode is a weak feature only; never a channel |
| Country is the only safe hard partition | Confirmed: 100% of true pairs share country; region 94.7% | Matches current design |
| Test distribution ≈ validation distribution | Test adds **France**, absent from train | Add a **country-held-out** evaluation (train US → eval India and vice-versa) as a proxy for France drift; penalise pipelines that collapse there |
| Unknown label completeness | `train_ground_truth.tsv` has one row per S1 (2,206,821 rows), empty = singleton | Unlisted pairs are true negatives → safe negative mining; still audit a sample |
| Candidate cap K≈40 | Current final config: `all@20 + name@10 + addr@10`, ~27 cand/S1, **95.9%** pair recall, **0.985** ceiling F0.5 (10k-entity dev sample, notebook 02 §8) | Blocking is already near the report's gate-1 bar (95%); gate 2 (98–99%) is the target for the blocking search |

### 1.3 Where the remaining F0.5 is likely to be lost (hypotheses to measure, not results)

1. **Blocking misses** (ceiling 0.985 → at most ~1.5 points): no-address S2/S3 records with common names,
   heavy back-transliteration (`phorcyun imphotek` ↔ *fortune infotech*) — notebook 02 §6.1.
2. **Singleton false positives**: one wrong match zeroes an entity (report correction 6).
3. **Chain / same-name branches**: correct name, wrong branch → numeric/address conflicts must veto.
4. **France drift**: abbreviation maps and legal forms tuned only on US/India.

The search below is ordered so the cheapest axes addressing these four go first.

---

## 2. Prerequisites (do these before any search)

1. **Run `notebooks/03_run_full_pipeline.ipynb`** to produce `artifacts/candidates_train/` (labelled,
   with per-pass `score_*`/`rank_*`) and `artifacts/candidates_test/`. Every stage after blocking reads
   these, so blocking variants only need to re-run when axis B changes.
2. **Fix the fold assignment once**: `artifacts/folds.parquet` = (`s1_entity_id`, `fold` 0–4,
   `country`, `multiplicity`, `is_singleton`), grouped by S1 id, stratified on country × singleton ×
   multiplicity bucket {0,1,2,3–5,6+}. All pipelines use the same folds → paired comparisons.
3. **Carve a final untouched holdout** (5% of S1, stratified the same way) that is scored only for the
   top 2–3 pipelines at the very end (report §7.2: repeated policy comparison needs one).
4. **Development subsample** for fast iteration: 100k S1 entities (all their candidates) from folds
   0–4 — the "dev" tier. The "full" tier is all 2.2M.
5. **Label audit**: 200 random singletons + 200 random multi-match groups, eyeball against top
   candidates. Records whether "entity" = branch (report assumption, §1).

---

## 3. Evaluation harness (shared by every pipeline)

New package `code/business_entity_resolution/src/er_eval/`:

```
er_eval/
  metrics.py        # entity_f05, macro F0.5, exact_macro_threshold, candidate recall variants,
                    # singleton accuracy, FP/1000 and Affected/1000, pair P/R
  bootstrap.py      # paired entity bootstrap (B=1000), CI for a metric and for Δ between pipelines
  registry.py       # run config → run_id (hash); writes results; loads any set of runs for plots
  slices.py         # slice definitions (country, source mix, multiplicity, name frequency, ...)
  error_types.py    # per-entity loss decomposition (see §6.4)
```

### 3.1 Metrics (all computed from one per-entity table)

Per S1 entity: `g` (gold size), `c` (# gold in candidates), `k` (predicted size), `tp`.
From those alone: entity F0.5 (count form, report §6.4), ceiling F0.5 `1.25c/(0.25g+c)`,
candidate macro recall, pair recall, any-match / all-match coverage, singleton accuracy,
FP/1000, Affected/1000. Pair-level P/R and PR curves use the pair table.

### 3.2 Validation tiers

| Tier | Refs | Pool | Used for |
|---|---|---|---|
| `dev` | 100k S1, one fold as val | full train pool (10.3M) | Every search stage |
| `cv5` | all 2.2M, 5 grouped folds | full train pool | Finalists only |
| `xcountry` | train US → val India; train India → val US | full pool | Drift robustness (France proxy) |
| `holdout` | 5% untouched | full pool | Final 2–3 pipelines, once |

### 3.3 Results schema (so every plot reads the same files)

```
experiments/<run_id>/
  config.yaml            # full pipeline spec (axes A–H below), git sha, data hash
  entity.parquet         # s1_entity_id, fold, g, c, k, tp, f05, slice columns, error_type
  pairs.parquet          # s1, cand, label, oof_raw, oof_cal, selected, channel_mask, rank_*
  thresholds.json        # chosen policy per fold
  metrics.json           # headline metrics + bootstrap CIs
  ledger.json            # wall time and peak RSS per stage
  models/                # fitted per-fold models (optional, for SHAP / SAE)
  embeddings/            # optional: sampled record/pair embeddings for feature-space plots
```

### 3.4 Registry / tracking

A flat `experiments/index.parquet` (run_id, parent_id, stage, axes values, headline metrics, CI,
wall time) is enough; MLflow is optional. Every run records its **parent** so ablation deltas and
waterfall plots are automatic.

---

## 4. Search space: pipeline axes

A pipeline = one choice on each axis. Values marked **(base)** form the baseline pipeline P0.

| Axis | Options to search | Report ref |
|---|---|---|
| **A. Normalisation** | A0 current `cleaning.py` **(base)**; A1 + text-extracted phone/domain columns; A2 + French abbreviation/legal-form expansion review; A3 libpostal parse (sample-only first) | §4.1 |
| **B. Blocking** | B0 `FINAL_PASSES` **(base)**; B1 K sweep per pass {10,20,30,50}; B2 cap sweep {500,1000,2000,5000}; B3 + char 3/4-gram name pass; B4 + dense rescue (frozen mE5-small / MiniLM, FAISS IVF or HNSW, top 20); B5 learned compressor (cheap GBDT over union, keep top K_final) | §4.2–4.3, gates 1–3 |
| **C. Features** | families: F-STR (lev, DL, jaro, JW, token/char-gram cosine), F-RARE (IDF, weighted Jaccard, SoftTF-IDF, rare-token overlap), F-NUM (house/name-number/sector conflicts), F-ADDR (street/locality/region/postcode), F-RET (per-pass score & rank, RRF, channel count), F-CTX (candidate-set context: rank, gap to best, z-score, count, name frequency), F-LEGAL (legal-form agreement/conflict), F-TXT (extracted phone/domain, needs A1). Base = F-STR + F-RET | §5.1 |
| **D. Matcher** | D0 logistic regression on base features (sanity); D1 LightGBM **(base)**; D2 XGBoost; D3 CatBoost; D4 bi-encoder cosine only; D5 GBDT + bi-encoder cosine as feature; D6 GBDT + targeted cross-encoder on uncertain pairs (stacked) | §5.2–5.3 |
| **E. Negatives** | E0 all retrieved candidates **(base)**; E1 capped 12/anchor (report curriculum); E2 + singleton hard cases up to 8; E3 + one refresh with model false positives | §4.4 |
| **F. Calibration** | F0 none; F1 Platt; F2 isotonic **(base)**; F3 beta | §6.2 |
| **G. Set policy** | G-A global threshold **(base)**; G-B singleton gate `q_i` + threshold; G-C ambiguity/conflict-adjusted threshold `T_i`; G-D GFM-style set utility (research comparator) | §6.3–6.4 |
| **H. Graph** | H0 none **(base)**; H1 conservative small-component refinement | §6.5 |

Full cross-product is ~10⁵ configs — not searchable. The search is **staged coordinate descent with
gates**: optimise the axis with the biggest expected headroom on frozen upstream artifacts, lock it,
move on. Each stage keeps the **top 2** variants (not just 1) as its output, so interactions get one
chance to show up in the next stage.

---

## 5. Staged search procedure

Each stage lists: what varies, what is frozen, the selection metric, and the gate to advance.
Selection metric is always **dev macro F0.5 (with the tuned policy)** unless stated; ties within the
bootstrap CI are broken by lower wall time, then simpler config.

### Stage 0 — Baselines and harness sanity (dev tier)

| Pipeline | Definition | Purpose |
|---|---|---|
| P-empty | predict ∅ for everyone | = singleton rate; the floor any pipeline must clearly beat |
| P-oracle | predict exactly the gold ∩ candidates | = ceiling F0.5 (should reproduce ≈0.985) — validates metric code |
| P-top1 | top-1 `all` pass if score > t | simplest non-trivial baseline |
| P-rule | score > t on the `all` pass, t tuned by exact optimiser | "blocking score is the matcher" |
| P0 | A0·B0·(F-STR+F-RET)·D1·E0·F2·G-A·H0 | the reference pipeline every later Δ is measured against |

Gate: P-oracle within ±0.001 of notebook 02's ceiling on the same sample; P0 > P-rule outside CI.

### Stage 1 — Blocking frontier (axis B, with A1/A2 if cheap)

Frozen: nothing downstream; measure *ceiling* F0.5 and candidate recall only (no matcher needed).

1. K × cap grid for each pass (B1, B2) → recall@K curves and the recall-vs-mean-K Pareto front;
   then a multi-objective Optuna study over K, cap and family weights (§5.2) to refine that front.
2. Incremental-addition and leave-one-channel-out for the 8 key families and 3 passes (report §7.3
   requires both).
3. B3 char-gram pass on the misses only first (cheap), then full.
4. B4 dense rescue **only if** the miss taxonomy (§6.1 plots) shows ≥30% of misses are
   lexical-divergence type; admission gate: ≥+0.5 pp macro candidate recall after compression at
   ≤+20% mean K (report gate 3).
5. B5 compressor: does a cheap scorer on a wider union (K≈100) beat the fixed union at equal final K?

Output: two blocking configs on the Pareto front (e.g. "~27 cand" and "~45 cand"). Stage 2 runs
on both, because a stronger matcher can afford more candidates.

### Stage 2 — Features and matcher (axes C, D, E) on frozen candidates

1. **Feature-family forward selection** with D1: base → +F-RARE → +F-NUM → +F-ADDR → +F-CTX
   → +F-LEGAL → +F-TXT; then leave-one-family-out from the full set (report E10–E12).
   F-CTX must be built from OOF first-pass scores (report §5.1 note).
2. **Matcher family** on the best feature set: D0, D1, D2, D3, each tuned with an
   equal-budget Optuna study (§5.2); compare the re-evaluated best configs, not the defaults.
3. **Negatives** E0–E3 on the winner.

Gate to Stage 3: best config beats P0 with paired-bootstrap 95% lower bound > 0.

### Stage 3 — Decision layer (axes F, G) on frozen OOF scores

Pure post-processing of `pairs.parquet` → minutes per config.

1. Calibrators F0–F3 compared by NLL, Brier, and **tail** reliability near the chosen threshold.
2. Policies G-A…G-D; singleton model `q_i` trained on one row per S1 incl. zero-candidate rows.
3. Threshold granularity: global vs per-country vs per-multiplicity-bucket (with shrinkage).
4. Gate τ_q, policy coefficients and shrinkage tuned jointly by a GP-based Optuna study (§5.2).

Gate: G-B/G-C adopted only if singleton FP/1000 does not rise and macro F0.5 CI lower bound > 0.

### Stage 4 — Neural add-ons (axis D4–D6, B4) — gated, optional

Only if Stage 2 error analysis shows a large residual class the GBDT cannot express (e.g.
transliteration FNs, semantic aliases). Licence constraint: MIT/Apache, ≤8B params —
`intfloat/multilingual-e5-small` (MIT), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0), `BAAI/bge-m3` (MIT) all qualify.

1. Frozen bi-encoder cosine as an extra feature (D5) — cheapest neural test.
2. Small contrastive fine-tune (multi-positive SupCon, same-country hard negatives, report §5.2) —
   fold-safe: one encoder per outer fold if promoted to `cv5`.
3. Cross-encoder on the uncertain band only (e.g. calibrated p ∈ [0.2, 0.8] or top-2 gap < δ),
   stacked with the GBDT score (D6).

Admission (report gate 4): paired lower CI > 0, point gain ≥ 0.001, singleton FP/1000 not worse.
Stage 4 is also where the **SAE analysis** (§6.6) happens, since it needs neural activations.

### Stage 5 — Graph refinement (axis H) — optional

H1 on the best locked pipeline; adopted only under report gate 5.

### Stage 6 — Confirmation

Top 3 pipelines → `cv5` (all folds) + `xcountry` → pick one → `holdout` once → fit on all train →
test inference → `utils/validate_submission.py`.

### 5.1 Named candidate pipelines to carry through the search

Expected to emerge from the stages above; listed up front so compute can be planned:

| ID | Blocking | Features | Matcher | Decision | Hypothesis it tests |
|---|---|---|---|---|---|
| P0 | B0 | STR+RET | LightGBM | iso + global t | Reference |
| P1 | B0 | +RARE+NUM | LightGBM | iso + global t | Rarity and numeric conflicts fix chains/branches |
| P2 | B0 | +RARE+NUM+ADDR+CTX+LEGAL | LightGBM | iso + global t | Full engineered set |
| P3 | B0 | P2 | best of XGB/CatBoost | iso + global t | Library choice matters? |
| P4 | B0 | P2 | P2/P3 winner + E2 negatives | iso + singleton gate | Singleton modelling is the big lever |
| P5 | B-wide (~45 cand) | P2 | P4 matcher | P4 policy | More recall affordable with strong precision layer |
| P6 | B0 + dense rescue | P2 + dense cos | P4 matcher | P4 policy | Dense recovers transliteration misses |
| P7 | P4/P5 best | + dense cos | GBDT + cross-encoder (uncertain band) | stacked, P4 policy | Neural reranking on the hard band |
| P8 | best | best | best | + graph H1 | Conservative graph gain |

### 5.2 Automated hyperparameter optimisation (Optuna / Bayesian optimisation)

The staged search in §5 picks **structural** choices (which channels, feature families, library,
policy). Inside each stage, the **numeric** knobs of the chosen structure are tuned automatically
with [Optuna](https://optuna.org) rather than by hand. HPO is a sub-step of Stages 1, 2, 3 and 4,
never a replacement for the gated structural comparisons.

#### Tooling

| Choice | Setting | Why |
|---|---|---|
| Sampler (default) | `TPESampler(multivariate=True, group=True, n_startup_trials=10, seed=…)` | Tree-structured Parzen estimator = sequential model-based (Bayesian) optimisation; handles mixed int/float/categorical and conditional spaces |
| Sampler (continuous, ≤10 params) | `GPSampler` (Gaussian-process BO) or `CmaEsSampler` | Stronger than TPE on small smooth continuous spaces, e.g. decision-layer coefficients |
| Multi-objective | `NSGAIISampler` / `TPESampler` with two directions | Blocking: max ceiling F0.5 vs min mean K; matcher: max F0.5 vs min wall time |
| Baseline sampler | `RandomSampler` with the same trial budget, once per stage | Proves BO actually beats random search here (report §3: matched-budget comparisons) |
| Pruner | `HyperbandPruner` (GBDT, via `optuna-integration` LightGBM/XGBoost callbacks reporting val loss per boosting round); `MedianPruner` across inner folds | Kills bad trials early; most of the HPO speed-up |
| Storage | `sqlite:///experiments/optuna.db`, one study per (stage, pipeline, tier) | Resumable after crashes/laptop sleep; lets two people run workers in parallel |
| Reproducibility | fixed sampler seed; each trial's full config written to the registry (§3.4) with `parent_id` = stage config and `trial_number` | Every trial is a normal run; all §6 plots work on it |

New module `code/business_entity_resolution/src/er_tune/` holding `spaces.py` (one search-space
function per stage), `objectives.py`, and `run_study.py` (CLI: `--stage --pipeline --tier
--n-trials --timeout --sampler`).

#### Objective and leakage rules

* **Metric**: dev-tier macro F0.5 **after** the exact global-threshold optimiser (report §6.4), not
  AUC or log-loss — the thing we are scored on. GBDT pruning uses val log-loss per round only as a
  cheap intermediate signal.
* **Nested splits**: within the dev tier, each trial trains on inner-train, tunes threshold/
  calibration on inner-tune, reports F0.5 on inner-val (grouped by S1 id). Outer folds of `cv5`
  and the `holdout` are **never** seen by any study.
* **Variance control**: objective = mean over 2–3 inner folds (pruned after the first if below
  the running median). Report the inner-fold std as a trial attribute.
* **Winner's curse**: the best trial's score is optimistically biased. Re-evaluate the top 5
  trials on fresh inner folds with 3 seeds and choose by **mean − 1·std**; that config (not the
  raw best trial) moves to the next stage.
* **Equal budgets**: when comparing matcher libraries (D1/D2/D3) or negative strategies, each gets
  the same number of trials and the same timeout, so "better library" ≠ "tuned longer".

#### Search spaces per stage

| Stage | Parameters (range, scale) | Objective | Budget (dev tier) |
|---|---|---|---|
| 1 Blocking | per pass `k` {5…60, int}; `cap` {300…10000, log}; family weights in `W_ALL`/`W_NAME`/`W_ADDR` [0, 2]; `min_score` [0, 0.3] | 2-objective: max ceiling F0.5, min mean candidates/S1 (Pareto front → pick 2 configs) | 60–100 trials on a ~10k S1 sample; top 5 re-scored on 100k |
| 2 LightGBM | `num_leaves` 15–255 log; `learning_rate` 0.01–0.2 log; `min_data_in_leaf` 20–2000 log; `feature_fraction` 0.4–1; `bagging_fraction` 0.5–1 (+`bagging_freq`); `lambda_l1`, `lambda_l2` 1e-3–10 log; `max_bin` {63,127,255}; positive weight 1–20 log; `n_estimators` via early stopping (≤3000) | macro F0.5 (post-threshold) | 50–80 trials, Hyperband pruning |
| 2 XGBoost | `max_depth` 4–12; `eta` log; `min_child_weight` 1–200 log; `subsample`, `colsample_bytree`; `gamma` 0–5; `alpha`/`lambda` log; `tree_method=hist` | same | same budget as LightGBM |
| 2 CatBoost | `depth` 4–10; `learning_rate` log; `l2_leaf_reg` 1–30 log; `border_count`; `random_strength`; `bagging_temperature` | same | same budget |
| 2 Negatives | total negatives per anchor 4–24; singleton hard negatives 0–8; easy-negative share 0–0.4 | same | 20–30 trials on the locked matcher params |
| 3 Decision | calibrator {platt, isotonic, beta}; singleton gate τ_q [0, 1]; policy coefficients α, β (G-C) [−0.5, 0.5]; group-threshold shrinkage 0–1 | macro F0.5, singleton FP/1000 as a constraint (`constraints_func`) | 100–200 trials (minutes: pure post-processing of OOF scores); `GPSampler` |
| 4 Bi-encoder fine-tune | lr 1e-6–1e-4 log; temperature 0.03–0.2 log; batch {64,128,256}; epochs 1–4; hard-negative ratio 0–0.75 | retrieval macro recall@20 on inner-val | 10–20 trials, `MedianPruner` per epoch |
| 4 Cross-encoder | lr log; max tokens {96,128,192}; uncertain-band bounds [p_lo, p_hi]; stacker regularisation | macro F0.5 of the stacked pipeline | 10–15 trials |

The global threshold itself is **not** an HPO parameter: the exact optimiser finds it inside every
trial. Letting BO search it as well would only add noise.

#### HPO diagnostics plots (`er_viz/hpo_plots.py`, rendered per study)

| Plot (`optuna.visualization.matplotlib` unless noted) | Shows | Decision |
|---|---|---|
| Optimisation history with best-so-far line, TPE vs random overlaid | Convergence; whether BO beats random | Stop early / raise the budget |
| Empirical distribution (EDF) of trial scores per sampler | Same, robust to lucky trials | Sampler choice |
| Hyperparameter importance (fANOVA and PED-ANOVA) | Which knobs matter | Shrink the space for the next study; fix unimportant ones |
| Slice plots per parameter | Where good values lie; whether the best sits on a range edge | Widen ranges that hit a boundary |
| Contour plot for the top-2 important parameters | Interactions | Joint vs separate tuning |
| Parallel-coordinates plot of the top 20% of trials | Shape of the good region | Sanity check |
| Pareto front (multi-objective studies), knee point marked | Recall vs candidates / F0.5 vs time | Pick the 2 blocking configs |
| Intermediate-value (pruning) plot | How early bad trials are cut | Tune pruner aggressiveness |
| Top-5 re-evaluation: mean ± std bars (custom) | Winner's-curse check | Final config choice |
| Tuned vs default-params Δ macro F0.5 with bootstrap CI (custom, feeds the §6.5 forest plot) | Value of HPO itself | Whether HPO was worth its wall time |

---

## 6. Analysis and plotting plan

New module `code/business_entity_resolution/src/er_viz/` (matplotlib, already a dependency; add
`seaborn`, `shap`, `umap-learn`, `scikit-learn`, `upsetplot`), plus notebook
`notebooks/04_pipeline_analysis.ipynb` that loads any list of `run_id`s from the registry and
renders every figure below to `reports/figures/<run_id or comparison_id>/`.

Conventions: one fixed colour per pipeline ID across all figures; every metric plot shows the
bootstrap 95% CI; every figure has a CSV of the plotted numbers saved next to it; slices always in
the same order.

```
er_viz/
  blocking_plots.py     # §6.1
  matcher_plots.py      # §6.2
  decision_plots.py     # §6.3
  error_plots.py        # §6.4
  compare_plots.py      # §6.5
  feature_space.py      # §6.6 (UMAP/t-SNE/PCA, probes, CKA, alignment/uniformity)
  sae.py                # §6.6 sparse autoencoder training + dashboards
  report.py             # stitches figures into one HTML report per comparison
```

### 6.1 Blocking analysis (Stage 1)

| Plot | Shows | Decision it supports |
|---|---|---|
| Recall@K curves per pass and per union (pair recall, macro candidate recall, all-match coverage) | How fast each pass saturates | K per pass |
| Recall vs mean candidates Pareto scatter (one point per B-config, front highlighted) | Cost/recall trade-off | Which 2 configs go to Stage 2 |
| Ceiling F0.5 vs K | Directly the metric blocking bounds | Same |
| UpSet plot of which passes/families recover each true pair | Redundancy vs unique contribution | Dropping redundant families |
| Leave-one-family-out Δrecall bar chart | Marginal value per family | Same |
| Rank-of-true-match histogram per pass (log y) | Whether misses are "just past K" or absent | K vs new channel |
| Candidate-count distribution (hist + p95/p99/max markers) per country | Tail blow-ups from common keys | Cap value |
| Miss taxonomy stacked bar (no address / junk name / transliteration / initials / region mismatch / numeric-only …) by country and source | What kind of recall is left | Whether dense rescue (B4) is justified |
| Recall by slice heatmap (country × source × name-frequency decile) | Where blocking is weak | Slice-specific fixes |
| Pool-size sweep (0.1M / 1M / 10M distractors) recall curve | Distractor sensitivity (report §7.1) | Confidence the dev tier is representative |

### 6.2 Matcher analysis (Stage 2)

| Plot | Shows |
|---|---|
| PR curve and ROC per pipeline (pair level), with operating point marked | Ranking quality |
| Score density by label, split into easy / hard (same-name) / singleton-candidate negatives | Where negatives overlap positives |
| Feature-family forward-selection line (macro F0.5 ± CI vs families added) and leave-one-out bars | Value of each family |
| GBDT gain importance + permutation importance (grouped by family) | What the model relies on |
| SHAP beeswarm (top 30), SHAP dependence plots for conflict features (house conflict, name-number conflict) coloured by name frequency | Whether conflicts actually veto matches |
| SHAP interaction heatmap (top 15 × 15) | Learned interactions, e.g. name-sim × address-conflict |
| Learning curve: dev F0.5 vs # training S1 (10k → 2M, log x) | Whether more data or more capacity helps |
| Train vs val metric per boosting round | Over/under-fitting |
| Per-country and per-source performance bars (+ `xcountry` transfer bars) | France-proxy robustness |

### 6.3 Calibration and decision analysis (Stage 3)

| Plot | Shows |
|---|---|
| Reliability diagrams per calibrator (overall + zoom on [0.7, 1.0]) with counts histogram | Tail calibration where the threshold lives |
| NLL / Brier table + bar chart per calibrator | Calibrator choice |
| **Macro F0.5 vs threshold** curve (exact optimiser output) per pipeline, with singleton-accuracy and mean-k curves on a shared x-axis | Sensitivity of the chosen threshold; flatness = robustness |
| Singleton gate sweep: macro F0.5, singleton accuracy, Affected/1000 vs τ_q | Singleton model value |
| Per-fold chosen threshold scatter | Threshold stability |
| Per-entity F0.5 histogram (mass at 0 and 1 highlighted) | Where the macro score is made/lost |
| Macro F0.5 by true multiplicity m (0,1,2,…,6+) with ceiling overlay | Singleton vs multi-match behaviour |

### 6.4 Error decomposition (every pipeline)

Every entity's shortfall `1 − F0.5_i` is attributed to exactly one primary cause, in order:
singleton FP → blocking miss (gold not in candidates) → matcher FN → matcher FP (non-singleton).

| Plot | Shows |
|---|---|
| Loss waterfall: 1.0 → −singleton FP → −blocking miss → −matcher FN → −matcher FP → achieved F0.5 | Which stage to invest in next |
| Same waterfall side by side for 2–3 pipelines | What a change actually fixed |
| Error-type × slice heatmap | Where each error concentrates |
| Error gallery (HTML table): top-N most confident FPs and FNs with raw + cleaned fields, SHAP top-5 reasons | Qualitative failure modes feeding back into cleaning/features |

### 6.5 Pipeline comparison (Stages 2–6)

| Plot | Shows |
|---|---|
| Leaderboard table: macro F0.5 [CI], ceiling, singleton acc, FP/1000, pair P/R, mean K, wall time, peak RAM | Headline |
| **Forest plot of paired Δ macro F0.5 vs parent** (bootstrap CI; vertical line at 0 and at +0.001 admission) | Which changes are real |
| Ablation waterfall P0 → P1 → … → final (cumulative Δ per step) | Story of the final pipeline |
| Pareto scatter: macro F0.5 vs wall time (and vs mean K), front highlighted | Cost-efficiency (report §7.3 CostEfficiency, in minutes) |
| Pipelines × slices heatmap of Δ vs P0 (diverging colormap centred at 0) | Slice-specific wins/regressions |
| Per-fold macro F0.5 lines (one line per pipeline across 5 folds) + rank-stability plot | Fold variance; whether rankings are stable |
| Critical-difference diagram (Friedman/Nemenyi over folds × slices) | Statistical ranking of ≥3 pipelines |
| Per-entity scatter F0.5(A) vs F0.5(B) with jitter, counts in each quadrant; + list of flipped entities | Where two pipelines disagree |
| `xcountry` transfer matrix per pipeline (train country × eval country) | Robustness to unseen country |

### 6.6 Feature-space visualisation (what the models learn)

Two kinds of models → two toolkits.

**(a) GBDT / engineered-feature pipelines**

1. **SHAP-space embedding**: UMAP of per-pair SHAP vectors (200k stratified pairs), coloured by
   TP/FP/FN/TN, by country, by error type. Clusters of FPs in SHAP space = a shared failure reason.
2. **Raw-feature PCA/UMAP** of the same pairs, coloured by label — does the input space already
   separate, or does the model create the separation?
3. **Leaf-embedding**: one-hot of leaf indices → TruncatedSVD → UMAP; compare with (1).
4. **Partial-dependence / ICE** curves for top name-similarity features, split by name-frequency tercile.

**(b) Neural pipelines (bi-encoder, cross-encoder; Stage 4)**

Sampling: 5k entities with ≥2 matches + 5k singletons + their top-10 candidates, stratified by country.
Store pooled embeddings (bi-encoder) and per-layer [CLS]/mean-pooled hidden states (cross-encoder).

| Analysis | Plot |
|---|---|
| Record embedding UMAP/t-SNE coloured by entity (only 30 sampled clusters, S1 ★, S2 ●, S3 ▲) | Do true matches cluster; do chain branches overlap? |
| Same UMAP coloured by country / source / has-native-script / has-house-number | Nuisance factors dominating geometry |
| Cosine-similarity histograms: positives vs hard negatives vs random, before/after fine-tuning | Separation gained by training |
| Alignment and uniformity (Wang & Isola) per checkpoint | Collapse vs spread over training |
| Retrieval recall@K vs embedding dimension/quantisation (FP32, FP16, PQ) | Compression retention (report §4.3) |
| Pair-representation UMAP (u, v, abs(u−v), u⊙v) or cross-encoder [CLS] coloured by TP/FP/FN/TN | Where the model is confused |
| Linear probes per layer: country, source, legal form, "contains number", "match" | What information each layer carries |
| CKA heatmap between layers and between models (frozen vs fine-tuned, mE5 vs MiniLM) | How much fine-tuning changes representations |

**(c) Sparse autoencoders on neural activations**

Goal: decompose dense activations into sparse, (hopefully) interpretable features — e.g. a
"house-number mismatch" feature, a "Pvt Ltd transliteration" feature, a "French street type"
feature — and check whether the matcher's decisions rely on them.

Steps:

1. **Collect activations**: bi-encoder pooled embeddings (d=384/768) for ~2–5M records; for the
   cross-encoder, residual stream / [CLS] hidden state at 2–3 layers (early, middle, final) for
   ~1–2M candidate pairs, with token-level activations kept for a 50k-pair subsample (needed for
   token attribution). Store FP16 memmaps under `experiments/<run_id>/activations/`.
2. **Train a TopK SAE** per (model, layer): dictionary size 8×–16×d, k ∈ {16, 32, 64}, MSE loss +
   auxiliary loss for dead latents, decoder columns unit-norm, activations mean-centred and
   scaled; ~10–20 epochs on the collected set (fits on CPU for d=384; one GPU hour otherwise).
   Implement in ~150 lines of PyTorch in `er_viz/sae.py` (no extra dependency).
3. **Quality plots**: reconstruction FVU vs L0 frontier (one point per k / dictionary size); dead
   and dense latent fractions; latent firing-frequency histogram (log10 density); **downstream
   fidelity** — replace activations with SAE reconstructions and re-measure retrieval recall@K
   (bi-encoder) or pair AUC / macro F0.5 (cross-encoder). An SAE that loses >1 pp downstream is not
   trusted for interpretation.
4. **Interpretation dashboards** (HTML, one card per latent): top-20 activating records/pairs with
   token highlights, activation histogram, and automatic concept tags from simple detectors
   (digit present, legal form, native script, country, source, address component type).
   Report the share of latents with a clean tag.
5. **Latent–label association**: for pair activations, mean activation on TP vs FP vs FN vs TN;
   volcano plot (log-odds of firing in FP vs TP against frequency) to find "false-merge" latents.
6. **Causal checks**: ablate (zero) or clamp the top FP-associated latents and measure the change
   in predicted probability and in macro F0.5 on the dev tier; plot Δp distributions.
7. **Cross-model comparison**: match latents between SAEs of frozen vs fine-tuned models (and
   mE5 vs MiniLM) by decoder cosine; histogram of max-cosine matches and a list of latents that
   appear only after fine-tuning — what the task training taught the encoder.
8. **Latent map**: UMAP of decoder directions coloured by concept tag.
9. **Feedback loop**: any clean, high-value SAE concept that the GBDT lacks (e.g. a specific
   transliteration pattern) becomes a candidate engineered feature for axis C — tested like any
   other family.

SAE work is **diagnostic**, never part of the submitted pipeline, and is only done on the neural
models that survive Stage 4 (or on the frozen encoder if Stage 4 is skipped, to decide whether it
is worth running).

---

## 7. Order of work and deliverables

| # | Work item | Output | Depends on |
|---|---|---|---|
| 1 | Run notebook 03 (full candidates) | `artifacts/candidates_{train,test}/` | — |
| 2 | Folds, holdout, dev sample, label audit | `artifacts/folds.parquet`, audit notes | 1 |
| 3 | `er_eval` metrics + bootstrap + registry; unit-test F0.5 and exact threshold on toy cases | package + tests | — |
| 4 | Stage 0 baselines + §6.4 waterfall + §6.5 leaderboard skeleton | first comparison report | 2, 3 |
| 4b | `er_tune` (Optuna studies, SQLite storage, registry hook) + `er_viz/hpo_plots.py`; smoke-test a 10-trial LightGBM study on P0 against the random sampler | HPO harness | 3, 4 |
| 5 | Stage 1 blocking search + §6.1 plots + blocking HPO (§5.2) | 2 blocking configs | 4, 4b |
| 6 | Pair-feature module (families C) with vectorised/rapidfuzz kernels, cached per candidate set | `features/` parquet | 5 |
| 7 | Stage 2 + §6.2 plots, §6.6(a) SHAP-space views | best engineered pipeline | 6 |
| 8 | Stage 3 + §6.3 plots | locked decision layer | 7 |
| 9 | First leaderboard submission from best of P1–P4 | test `matching_results.tsv` | 8 |
| 10 | Stage 4 neural (gated) + §6.6(b) + SAE §6.6(c) | P6/P7 results, SAE dashboards | 8 |
| 11 | Stage 5 graph (optional) | P8 | 8 |
| 12 | Stage 6 confirmation (`cv5`, `xcountry`, holdout) + final comparison report | final pipeline, `reports/` | all |
| 13 | Fill `Documentation_template.md` with the waterfall, ablation forest plot and key SHAP/SAE findings | submission docs | 12 |

Dependencies to add in `pyproject.toml` when the relevant step starts: `scikit-learn`, `lightgbm`,
`xgboost`, `catboost`, `optuna`, `optuna-integration` (pruning callbacks), `shap`, `seaborn`, `umap-learn`, `upsetplot` (steps 3–8);
`torch`, `sentence-transformers`, `faiss-cpu` (step 10 only).

### Stop / skip rules

* If Stage 3 already puts dev macro F0.5 within 0.005 of the blocking ceiling, skip Stage 4 and
  spend the time on blocking (the ceiling becomes the binding constraint).
* If a neural stage's projected wall time exceeds the whole Stages 0–3 time, run it on the dev tier
  only and report it as exploratory.
* Never tune on `holdout` or on test-leaderboard feedback beyond picking between ≤3 finalists.
