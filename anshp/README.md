# Anshp: Qwen business entity resolution

The sole learned matcher is **Qwen/Qwen3-Reranker-0.6B**, adapted with LoRA. Country-partitioned lexical retrieval produces candidates; Qwen scores every retained pair independently. Thresholding permits multiple matches and empty results.

The notebook pipeline is implemented and has passed a synthetic integration check using a tiny, randomly initialized Qwen3 model with real LoRA updates. The full data audit has passed. The pretrained 0.6B checkpoint and tokenizer are cached at pinned revision e61197ed45024b0ed8a2d74b80b4d909f1255473. Real tokenizer and GPU forward/backward checks passed, with zero optimizer steps. All 24,229,173 records have been normalized. The training retrieval index is complete with 10,320,219 external records; its first recall pilot is running. Trained-model accuracy is not yet measured.

## Files and execution order

| Notebook | Implementation |
| --- | --- |
| [00_setup_and_data_audit.ipynb](00_setup_and_data_audit.ipynb) | Disk-backed six-file audit; unique IDs; full truth coverage; external ownership; country consistency; deterministic stratified reference splits |
| [01_data_cleaning.ipynb](01_data_cleaning.ipynb) | Raw fields, conservative native-script Qwen text, separate lexical variants; batched versioned Parquet |
| [02_blocking.ipynb](02_blocking.ipynb) | SQLite FTS5 BM25, trigrams, initials, numeric-address anchors, rare-term filtering, reciprocal-rank fusion, provenance and recall reports |
| [03_qwen_training.ipynb](03_qwen_training.ipynb) | Immutable revision pinning; LoRA loading; fit-only SQL supervision; reference-normalized BCE; hard-negative refresh; best adapter saving; resumable pair scoring |
| [04_validation_and_thresholds.ipynb](04_validation_and_thresholds.ipynb) | Exact macro-F0.5 sweep; complete candidate-score verification; threshold policy; country reports; locked holdout evaluation |
| [05_inference_and_submission.ipynb](05_inference_and_submission.ipynb) | Trained-model loading, test scoring, streamed TSV export and strict validation |
| [06_pipeline_checks.ipynb](06_pipeline_checks.ipynb) | Reproducible synthetic integration and negative checks without downloading weights |
| [07_runtime_profiles.ipynb](07_runtime_profiles.ipynb) | Offline, fit-only real-Qwen GPU profile and candidate-count runtime estimates |

All project Python source lives in notebooks. Existing files outside this folder are unchanged. Reusable cells carry the er-definitions tag; stage loading executes definitions without triggering notebook jobs.

## Environment

The verified local environment is Python 3.11.7, PyTorch 2.5.1+cu121, Transformers 5.5.3 and PEFT 0.18.0, with an RTX 4060 Laptop GPU. [requirements.txt](requirements.txt) records direct dependency versions exercised by the check. It is not a complete transitive lock or a GPU-driver installer.

The current anshp/.venv inherits installed system packages. From the workspace root, launch Jupyter with that interpreter:

~~~powershell
anshp/.venv/Scripts/python.exe -m jupyterlab
~~~

Select its Python kernel. If that kernel is not available, register it inside the environment:

~~~powershell
anshp/.venv/Scripts/python.exe -m ipykernel install --sys-prefix --name anshp --display-name "Python (anshp)"
~~~

Run notebook 06 first to verify the environment. It writes reports/pipeline_smoke.json and isolated artifacts under artifacts/smoke-*. These contain synthetic records and are never competition results.

## Running the real-data pilot

1. In notebook 00, set RUN_AUDIT=True. Review source counts, split counts and country checks. Shared external ownership fails closed until connected references are grouped. Tiny strata can leave an evaluation split empty.
2. In notebook 01, set RUN_CLEANING=True. This is the only owner of cleaning rules.
3. In notebook 02, set RUN_BLOCKING=True to build the train index and bounded early-stop candidates. Compare caps 10/20/40, recall, overflow, index size and latency before locking blocking.selected_candidate_cap. A null cap uses 40 only as a pilot hypothesis.
4. In notebook 03, enable PIN_REVISIONS once to resolve immutable Hugging Face commits. RUN_TRAINING=True downloads the pinned checkpoint and runs the configured pilot. The run cell scores a frozen baseline, then trains up to two epochs and saves the best adapter by early-stop macro-F0.5. Inspect the frozen comparison and compute profile before expanding.
5. In notebook 04, enable RUN_VALIDATION to score fixed threshold-tune and holdout samples with the best adapter. The in-memory evaluator rejects more than 100,000 references or 5 million pairs. Larger evaluation requires an external-sort implementation.
6. Build the test index and **all** test candidates using notebook 02's functions, the same cap/channel settings as training, and reference_limit=None. Enable RUN_TEST_INFERENCE in notebook 05.

Example in notebook 02, after locking the candidate cap:

~~~python
cap = CONFIG["blocking"]["selected_candidate_cap"]
if cap is None:
    raise ValueError("Select and evaluate a candidate cap first")
build_retrieval_index(ARTIFACT_DIR/"normalized", ARTIFACT_DIR/"audit",
                      ARTIFACT_DIR/"index_test", "test")
build_candidates(ARTIFACT_DIR/"index_test", ARTIFACT_DIR/"audit",
                 ARTIFACT_DIR/"candidates_test", cap=cap,
                 channel_limit=CONFIG["blocking"]["channel_limit"],
                 reference_limit=None)
~~~

The default supervised pilot uses 10,000 fit references, with up to 8 hard and 2 easier negatives per reference plus every known positive. Non-fit-owned external records are excluded from fitting and mining. Negatives refresh once after the first epoch. Each optimizer step averages the mean pair losses of eight complete reference groups, independently of microbatch boundaries. Training does not resume optimizer state; scoring resumes by persisted pair keys. Cleaning now resumes completed source files using checksum-verified sidecars, and writes progress.json during processing.

## Artifacts and limits

Each stage publishes a completion manifest after successful writes. Input hashes, normalization, retrieval settings, helper source, pinned revisions and checkpoint checksums bind downstream artifacts. Parquet, checkpoints, scores and submissions have SHA-256 checksums. Large intermediate audit/index/candidate databases are checked by size and should be treated as immutable. Interrupted unpublished temporary files may remain for inspection.

Scoring streams from SQLite. Submission validation checks exact saved candidates, valid external IDs, duplicates, full reference coverage and matches being a subset of candidates. Every test reference is retained, including empty sets.

Optional Qwen embedding retrieval is **not implemented**. Country-transfer experiments, full-data recall/cost profiling, production-scale threshold sorting and optimizer checkpoint resume remain future work. The research budget is a planning assumption; no cloud job has been launched.

See [config.json](config.json) for settings and [QWEN_DESIGN.md](QWEN_DESIGN.md) for the broader design. Primary model reference: [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

## Measured local profile

The initial 64-pair profile measured about 42.7 pairs/second for inference at batch 4 and 9.85 pairs/second for forward/backward on the RTX 4060 Laptop GPU, using bfloat16. Batches 16 and 32 were slower on this sample. At that rate, test caps of 10/20/40 imply approximately 113/225/450 GPU-hours before record-lookup I/O, if every reference fills its cap. This is a short fit-only throughput profile, not a quality result or a sustained thermal benchmark. See reports/real_qwen_gpu_profile_initial.json. A repeat during concurrent indexing favored batch 16 at 36.4 pairs/second, showing the need for a longer isolated benchmark before choosing an inference batch size. In that repeat, disabling gradient checkpointing improved forward/backward from 3.69 to 6.43 pairs/second and used 3.60 GB of allocated GPU memory; the training default now disables checkpointing. See reports/real_qwen_gpu_profile.json.
