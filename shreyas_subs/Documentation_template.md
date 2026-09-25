# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Antigravity ER  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an end-to-end, sub-linear Business Entity Resolution pipeline designed for high-precision record linkage across multi-source commercial business datasets (US, India, and France). The pipeline couples a **Dual-Path Inverted Index Blocking Engine** (capturing distinctive brand tokens, address tokens, and numeric plot/house numbers) with a precision-tuned **LightGBM Binary Classifier** operating on a 12-dimensional pairwise similarity feature space. The decision threshold is calibrated to maximize the competition metric ($F_{0.5}$ macro-average), achieving **0.9889 validation $F_{0.5}$** while executing in sub-linear time within 15 GB of system memory.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA) across 26.4 million records, several fundamental data characteristics and noise patterns were discovered:
1. **Deduplicated Reference Source:** Source 1 is deduplicated (2.2M train, 1.73M test) with 0% missing fields. Targets in Source 2 and Source 3 exhibit missing addresses in 3.34% of records.
2. **Singleton Prevalence:** 123,247 Source 1 entities (**5.58%**) have 0 matching records in the ground truth. Under the $F_{0.5}$ metric, false positive merges on singletons award 0.0, whereas predicting an empty match list awards 1.0. High precision is therefore paramount.
3. **Strict Country Boundaries:** Across all 7,638,365 true match pairs in the training ground truth, **0 cross-country matches exist**. Country partitioning serves as a zero-leakage hard blocking boundary.
4. **Multilingual and Indic Scripts:** While Source 1 names are Latin, Source 2 and Source 3 contain Devanagari (3.9%), Tamil (0.6%), Telugu (0.4%), Odia, Bengali, and Kannada transliterations. However, the address field for these entities is Romanized and shares identical numeric plot/house numbers.
5. **Noise Taxonomy:**
   - Name distortions: Domain names (`glypholaspa.com`), prefix additions (`f/k/a`, `aka`, `formerly`, `m/s`), legal suffix permutations (`pvt ltd` vs `private limited`), and typos.
   - Address distortions: Missing PIN codes (present in only 6.7% of addresses), reordered components (`KY, Nicholasville, 602 Maple St`), leading zeros (`00680`), and punctuation noise (`#`, `##`).

### 2.2 Solution Strategy
**Approach Type:** Dual-Path Inverted Index Blocking + Gradient Boosted Decision Trees (LightGBM) + $F_{0.5}$ Macro-Threshold Calibration.  
**Core Innovation:** 
- **Sub-Linear Dual-Path Blocking:** Bypasses the $O(N \times M)$ trillion-pair country cartesian product by indexing distinctive name tokens alongside address house/plot numbers, maintaining 99.93% recall ceiling while running in minutes.
- **Digit Consistency Engine:** Exploits the fact that 94.5% of addresses contain numbers/digits with a 79.9% true match rate, introducing `digit_jaccard` and `digit_mismatch_penalty` to eliminate false merges between distinct businesses on the same street.

---

## 3. Candidate Generation (Blocking)
- **Blocking Keys Used:**
  - Hard partition: `country` (US, India, France).
  - Path A (Brand Tokens): Distinctive name tokens ($\ge 3$ characters, stripped of common entity stopwords like `inc`, `ltd`, `corp`, `services`).
  - Path B (Address & Locality Tokens): Distinctive address words ($\ge 4$ characters) and extracted numeric tokens (house, plot, survey, suite numbers).
- **Candidate Pairs Generated:**
  - Top 40–50 candidates queried per Source 1 entity via token overlap.
  - Permissive candidate filter: $\max(\text{name\_token\_set\_ratio}, \text{addr\_token\_set\_ratio}) \ge 60$.
  - Average candidates per S1 entity: ~12–25 candidates.
- **Ensuring True Matches Are Preserved:**
  - Evaluated on ground truth: normalized `token_set_ratio` achieves 99.93% recall at threshold 60.
  - By indexing both brand name and address digits, entities with trade names (DBA) or non-Latin script names are retrieved via their address and numeric keys.

---

## 4. Matching Model

**Features Used (12-Dimensional Compact Vector):**
- **Name Features:**
  - `name_token_set_ratio`: Set-based token matching (resilient to word order and extra tokens).
  - `name_token_sort_ratio`: Sort-based token matching.
  - `name_wratio`: RapidFuzz weighted heuristic ratio.
  - `name_len_diff_ratio`: Relative character length disparity.
- **Address Features:**
  - `addr_token_set_ratio`: Set-based address token similarity (0 if either address is null).
  - `addr_token_sort_ratio`: Sort-based address token similarity.
  - `is_addr_null`: Binary flag indicating missing address in candidate record.
- **Consistency & Compound Features:**
  - `digit_jaccard`: Jaccard similarity of extracted numeric tokens (house/plot numbers).
  - `digit_mismatch_flag`: Penalty flag set to 1.0 when both records contain digits but share zero common numbers.
  - `max_sim`: $\max(\text{name\_sim}, \text{addr\_sim})$.
  - `min_sim`: $\min(\text{name\_sim}, \text{addr\_sim})$.
  - `domain_match_flag`: Match indicator when a target URL domain slug embeds the reference name.

**Model Type:** LightGBM Binary Classifier (`objective="binary"`, `metric="auc"`, `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`).  
**Threshold Selection Method:** Direct grid search optimization over prediction probabilities $\tau \in [0.30, 0.95]$ evaluated against the exact competition metric: macro-averaged $F_{0.5}$ across all reference entities (including singletons).

---

## 5. Results & Error Analysis
- **Validation Metric:**
  - **Macro-Averaged $F_{0.5}$:** **0.9889** on held-out validation split.
  - **Optimal Threshold $\tau^*$:** 0.850 – 0.900 (reflects the 2× precision penalty in $F_{0.5}$).
- **Key Feature Importances:**
  1. `addr_token_sort_ratio` (1,356 splits)
  2. `name_wratio` (1,306 splits)
  3. `name_len_diff_ratio` (1,169 splits)
  4. `addr_token_set_ratio` (976 splits)
  5. `max_sim` (958 splits)
  6. `digit_jaccard` (573 splits)
- **Error Analysis:**
  - **False Positives (Wrong Merges):** Strongly suppressed by `digit_mismatch_flag` and the high confidence threshold $\tau^* = 0.85+$.
  - **False Negatives (Missed Matches):** Occur primarily when both the brand name is completely substituted (DBA) AND the target address is entirely blank (missing in 3.3% of S2/S3 records).

---

## 6. Conclusion
The proposed pipeline solves multi-source entity resolution at commercial scale by combining sub-linear dual-path inverted indexing with a precision-calibrated gradient boosting classifier. Through systematic text canonicalization, digit consistency checks, and country-isolated streaming, the system achieves **0.9889 validation $F_{0.5}$** while operating comfortably within 15 GB RAM, fulfilling all competition constraints.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
- `src/normalization.py`: Text cleaning, domain stripping, suffix canonicalization.
- `src/blocking.py`: `CountryInvertedIndex` for token-based candidate retrieval.
- `src/features.py`: Pairwise feature extraction.
- `src/model.py`: LightGBM training and macro $F_{0.5}$ optimizer.
- `src/main.py`: End-to-end execution pipeline.
- `requirements.txt`: Pinned dependencies (`rapidfuzz`, `lightgbm`, `polars`, `pandas`, `scipy`).
- `README.md`: Instructions to reproduce all outputs.

### B. Validation Verification
Verified against `utils/validate_submission.py`:
- `matching_results.tsv`: Pass (exact 2 columns, all S1 IDs present, no duplicate IDs, correct TSV format).
- `candidate_pairs.tsv`: Pass (every matched ID is a verified subset of candidates).
- Exit code 0 (`PASS — no blocking issues found. Safe to submit.`).
