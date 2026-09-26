# Business entity resolution: candidate-pipeline search

This runs the plan in `entity_resolution_research/pipeline_search_plan.md` end to end:

- it builds a set of candidate pipelines;
- scores each one out-of-fold on the same labelled sample;
- tunes them with Optuna;
- writes a comparison report with every analysis figure;
- then refits the best pipeline and writes the submission files.

```
data ─▶ clean ─▶ key cache ─▶ wide blocking (labelled train sample) ─▶ pair features
     ─▶ Optuna: blocking (multi-objective) + LightGBM / XGBoost / CatBoost
     ─▶ 5-fold OOF evaluation of every pipeline ─▶ comparison report
     ─▶ best pipeline: refit, holdout score, test retrieval + scoring ─▶ output/*.tsv ─▶ validator ─▶ zip
```

Everything goes through one entry point, `src/run.py`.

## Quick start (SageMaker)

**Instance:**
- Use a CPU instance with plenty of RAM: **ml.r5.4xlarge** (16 vCPU, 128 GiB) is the recommended default, and ml.m5.8xlarge also works.
- Attach **≥ 150 GB** of storage. The data is 2.5 GB raw, and cleaned files, key caches, candidates and features add roughly 30–60 GB.
- A GPU instance (ml.g5.xlarge) is only needed for the optional `neural` stage and the SAE analysis.

This works on a SageMaker Studio JupyterLab space or a notebook instance, from a terminal:

```bash
# 1. code + data (copy the repo and student_resource/ onto the instance, e.g. from S3)
aws s3 sync s3://<bucket>/amazon_ml_challenge ~/amazon_ml_challenge
cd ~/amazon_ml_challenge

# 2. environment: Python 3.12 via uv (pandas 3 needs Python >= 3.11; the default conda env is older)
pip install -q uv
uv sync --no-dev                    # core stack from pyproject.toml / uv.lock
# uv sync --no-dev --group neural   # + torch / sentence-transformers (GPU instance only)
# uv sync --no-dev --group viz      # + umap-learn (otherwise PCA→t-SNE is used)
# (without uv: python3.12 -m pip install -r code/business_entity_resolution/requirements.txt)

# 3. run everything in the background; it is resumable, so just rerun the same command after an interruption
cd code/business_entity_resolution/src
nohup uv run python run.py all \
    --data-dir ~/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/dataset \
    --work-dir ~/work --team <team_name> > ~/work.log 2>&1 &
tail -f ~/work.log

# 4. results
#    ~/work/reports/index.html       comparison report (all figures + CSVs)
#    ~/work/reports/leaderboard.csv
#    ~/work/output/matching_results.tsv, candidate_pairs.tsv (validated)
#    ~/work/<team_name>_submission.zip
aws s3 sync ~/work/reports s3://<bucket>/reports && aws s3 sync ~/work/output s3://<bucket>/output
```

If `cleaned/` and `blocking_cache/` already exist from the notebooks, pass `--clean-dir` and `--cache-dir` to reuse them. The key cache must be complete for both train and test.

## Stages

| stage | what it does | main outputs (under `--work-dir`) |
|---|---|---|
| `prepare` | Cleans all six files and builds blocking keys. Draws disjoint train samples: **dev** (5 stratified folds, used to score every pipeline), **hpo** (only seen by Optuna) and **holdout** (scored once, for the final pipeline). Retrieves *wide* candidates for them (all@50, name@30, addr@30), labels them and computes pair features. | `artifacts/samples.parquet`, `pairs_{dev,hpo,holdout}.parquet`, `blocking_misses.parquet` |
| `tune` | Optuna runs on the **hpo** sample only. **Blocking:** NSGA-II trades the blocking ceiling F0.5 against candidates per S1, and picks `Btuned` / `Bwide` / `Blean` from the Pareto front. **Matchers:** TPE (Bayesian) on LightGBM / XGBoost / CatBoost with equal budgets, a median pruner across inner folds, and an equal-budget random-search baseline for LightGBM. The top 5 trials are re-scored on fresh splits and the best mean − std wins. | `experiments/hpo/optuna.db`, `experiments/hpo/<lib>/best_params.json`, `experiments/blocking/blocking_configs.json` |
| `neural` *(optional, `--neural`)* | Frozen `intfloat/multilingual-e5-small` (MIT) embeddings produce a cosine-similarity feature. | `artifacts/dense_*.parquet`, `emb_train.npy` |
| `evaluate` | Scores every candidate pipeline out-of-fold on the dev sample. Inside each training fold the entities are split into **fit 70 / calibrate 15 / tune 15**, so no stage sees scores that were produced from its own labels. | `experiments/runs/<pipeline>/` (config, per-entity and per-pair OOF results, metrics with bootstrap CIs, importances, TreeSHAP sample) |
| `compare` | Builds the leaderboard, paired-bootstrap Δ vs parent, the ablation path, the cross-country transfer check (proxy for the unseen France) and every figure in plan §6. | `reports/index.html` + one PNG/CSV per figure |
| `submit` | Picks the best pipeline by OOF macro F0.5, refits it on the whole dev sample, scores the **holdout** once, retrieves test candidates with the pipeline's blocking config, scores them and applies its policy. Then it writes both TSVs, runs `utils/validate_submission.py` and builds the zip. | `output/`, `experiments/runs/<best>/holdout.json`, `<team>_submission.zip` |

Run stages one at a time with `python run.py <stage> ...` and the same `--data-dir` / `--work-dir`. Finished outputs are skipped unless you pass `--force`.

## Candidate pipelines

| name | change vs parent |
|---|---|
| `P-empty` | predict no match for anyone (the singleton rate) |
| `P-oracle` | a perfect matcher on B0 candidates (the blocking ceiling; excluded from selection) |
| `P-rule` | blocking score above a tuned global threshold |
| `P0` | LightGBM on name similarity + retrieval scores |
| `P0-lr` | P0 with logistic regression (sanity check) |
| `P1` | adds rarity (IDF) and numeric-conflict features |
| `P2` | the full engineered set: address, candidate-set context, legal form, phone/domain found in the text |
| `P3-xgb`, `P3-cat` | P2 with XGBoost / CatBoost |
| `P2-hpo-<lib>` | P2 with Optuna-tuned hyperparameters |
| `P4` | the best tuned library, extra weight on singletons' negatives, and a **singleton gate** (policy B) |
| `P4-E1` | P4 with capped hard negatives |
| `P4-C` | P4 with an ambiguity/conflict-adjusted threshold; α, β, τ tuned by Bayesian optimisation |
| `P5-<Bcfg>` | P4 on the Optuna-selected blocking configs |
| `P8` | P4 plus unique assignment of external records (graph refinement) |
| `P6` *(with `--neural`)* | P4 plus frozen multilingual-E5 similarity |

**Thresholds:** always chosen with the exact macro-F0.5 optimiser, which is O(M log M) and unit-tested against brute force.

**Selection:** the highest OOF macro F0.5, with ties broken by wall time. The forest plot shows which changes pass the admission gate: paired 95% CI lower bound > 0 and Δ ≥ 0.001. To submit a different pipeline, pass `--run <name>`.

## Knobs for time and memory

| flag | default | effect |
|---|---|---|
| `--n-dev / --n-hpo / --n-holdout` | 100k / 50k / 50k | sample sizes (train has 2.2M S1) |
| `--trials`, `--blocking-trials` | 30 / 80 | Optuna budgets; `--tune blocking,lgbm` limits which studies run |
| `--no-random-baseline` | off | skips the equal-budget random-search study |
| `--folds N` | 5 | scores only the first N outer folds (quick passes) |
| `--pipelines P0,P2,P4` | all | evaluates a subset |
| `--feature-chunk` | 500k | pairs per feature batch; lower it if memory is tight |
| `--batch-chunks` | 2 | S1 key chunks (250k each) per test retrieval batch |
| `ER_THREADS` env | all cores | threads for LightGBM / XGBoost / CatBoost |

**Runtime expectations.** These are measured only on a small smoke sample on a laptop; full-scale numbers are **not measured yet**:
- Pair features ran at about 20k pairs/s on 8 laptop threads. The full run needs about 12M train pairs and about 47M test pairs.
- On a small sample, one LightGBM pipeline takes about 30 s for 5 folds. At 100k dev entities, expect minutes per pipeline.
- Tuning dominates. Start with `--trials 15 --tune blocking,lgbm`, then widen.
- CatBoost is by far the slowest library: on the smoke sample it took about 11× LightGBM's time per trial. Leave it out with `--tune blocking,lgbm,xgb` if time is short; its untuned `P3-cat` pipeline still runs.

Each run's `ledger.json` records wall time and peak memory.

## Analysis figures (in `reports/index.html`)

| area | figures |
|---|---|
| Blocking | recall@K per pass, cost-vs-recall Pareto (Optuna trials + chosen configs), pass combinations, leave-one-pass-out, rank-of-true-match histograms, candidate-count distribution, miss taxonomy, recall by country × source |
| Matcher | PR curves with operating points, score density on hard / singleton negatives, gain importance (per feature and per family), TreeSHAP summary and conflict-feature dependence, per-country / name-frequency bars |
| Decision | reliability diagram + calibrator NLL, macro F0.5 vs threshold, singleton-gate sweep, per-entity F0.5 histogram, score by multiplicity vs ceiling |
| Errors | loss decomposition (singleton FP / blocking miss / matcher FN / matcher FP) per pipeline and as a waterfall, error type × slice, HTML error gallery |
| Comparison | leaderboard with CIs, forest plot of paired Δ vs parent, ablation waterfall, score-vs-time Pareto, slice heatmap, fold stability, per-entity scatter, cross-country transfer |
| Feature space | TreeSHAP-space vs raw-feature embedding coloured TP/FP/FN/TN, what makes FPs look like matches |
| Tuning | TPE vs random history and EDF, parameter importance (fANOVA, PED-ANOVA), slices, contour, parallel coordinates, pruning, top-trial re-scoring, blocking Pareto fronts |
| SAE *(needs `neural`)* | TopK sparse autoencoders on E5 record embeddings: FVU vs k, downstream-AUC fidelity, latent density, match-discriminating latents, ablation, latent dashboard (HTML), decoder map |

## Layout

```
src/
  run.py        entry point (stages above)
  cleaning.py   text normalisation (notebook 01)
  blocking.py   multi-pass sparse TF-IDF retrieval (notebook 02)
  data.py       paths, samples/folds, wide candidates, blocking truncation
  prepare.py    stage prepare
  features.py   pair features (families STR, RARE, NUM, ADDR, LEGAL, TXT)
  models.py     matchers, calibrators, singleton model, set policies, graph refinement
  pipelines.py  pipeline configs, leakage-safe fitting, OOF evaluation, cross-country check
  submit.py     refit, holdout, test inference, outputs, validator, zip
  neural.py     optional frozen bi-encoder
  er_eval/      metric (+ exact threshold optimiser), bootstrap, experiment registry
  er_tune/      Optuna search spaces and studies
  er_viz/       all figures and the HTML report (incl. sae.py)
tests/test_metrics.py
```

Tests: `python tests/test_metrics.py` (or `pytest tests`).

**Fair play:** nothing here looks anything up externally. Every model and statistic is fitted on the provided data only. The only external artefact is the optional MIT-licensed E5 model.
