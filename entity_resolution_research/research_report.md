# Precision-first multi-source business entity resolution under a $200 SageMaker budget

Technical research report · 25 September 2026 · Pricing basis: us-east-1, USD

## 1. Executive research summary

**Recommendation.** Build a country-partitioned, sparse-first retrieval cascade with explicit numeric/contact conflicts, token rarity, a CPU gradient-boosted matcher, calibrated probabilities, and an out-of-fold singleton/set decision model. Make dense retrieval a measured rescue channel. Restrict cross-encoding to ambiguous cases. Keep the default output as reference-centered match sets; enable graph changes only after an entity-level validation gain.

The task is anchored one-to-many retrieval: 1.73M reference records, roughly 10M external evaluation records, 26.4M records to prepare overall. Exhaustive comparison would require approximately 17.3 trillion reference–candidate pairs. At 40 candidates per reference, the upper planning volume becomes 69.2M pairs. This reduction alone does not ensure precision: retrieval must preserve distinctive evidence, and acceptance must explicitly allow an empty set.

The strongest empirical reasons for this architecture are the large-scale sparse results in [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf), the realistic difficulty axes in [WDC Products](https://www.openproceedings.org/2024/conf/edbt/paper-14.pdf), and the use of blocking plus feature-based matching in [Overture Places](https://docs.overturemaps.org/guides/places/). These establish useful mechanisms, not this dataset's achievable score.

**Expected expenditure:** a conditional planning total of **$198.97**, including $78.97 first-pass research/execution compute, $55 for a complete end-to-end rerun, $20 validation work, $20 ancillary storage/I/O, and $25 emergency reserve. All compute is budgeted at verified SageMaker On-Demand prices; Spot discounts are upside. No cloud jobs were launched for this investigation.

**Highest expected marginal value, a research hypothesis:** (1) reliable normalization plus rare-token retrieval; (2) numeric, phone, URL-path and geographic conflict features; (3) training on actual retrieved negatives and singleton false positives; (4) entity-level threshold selection and singleton modeling. Dense rescue is next if it recovers measured sparse misses. A global GNN, generative LLM matcher, and universal cross-encoder pass have weaker benefit-to-cost justification here.

### Evidence status and assumptions

- **Published evidence** below describes the cited source's own experiments.
- **Proposed** denotes this report's design choices.
- **Estimated** denotes arithmetic from explicitly stated resource/throughput assumptions.
- **NR** means not run. No labels, evaluation data, scorer implementation, or measured throughput were supplied; therefore fold scores, ablation gains, and attained 98–99% recall cannot honestly be reported as results.
- Pending scorer confirmation, empty truth/empty prediction scores 1, and all other zero-overlap cases score 0.
- Pending ontology confirmation, “entity” means a particular physical business branch, not an entire chain or corporate family. Validate this before using branch conflicts or unique-anchor constraints.
- Complete match-set labels, including verified singletons, are required for the main supervised evaluation. Unlisted pairs in incomplete labels are unknown.
- The region default is us-east-1. Reprice before execution elsewhere; credits' eligible services and taxes depend on the account.
- The requested “Conservative $300–400” and “Recommended $600–700” tiers conflict with the hard $200 limit. They are preserved as **optional, unfunded extensions** in Section 11. The $200 plan is the recommendation for the available credits.

### Structural corrections that change the design

1. Ditto is **PVLDB 14(1), 2020**, associated with the VLDB 2021 conference cycle, not EMNLP 2020. [Primary paper](https://arxiv.org/pdf/2004.00584)
2. A top-two margin is not a universal uniqueness test: both candidates can be true matches.
3. Same-source negatives are unsafe when S2/S3 contain duplicates.
4. A reference with \(m_i>40\) true external records has recall at most \(40/m_i\) under a strict 40-record cap. Use an audited overflow path.
5. ANN neighbor recall, semantic candidate recall, and compression retention are different measurements.
6. A correct singleton can lose its entire entity score from one false prediction. If the singleton fraction is \(s\) and fraction \(a\) of those receive false predictions, their contribution to macro-score loss is exactly \(sa\) under the stated convention.
7. A blocker miss is not evidence that the reference is a true singleton.

## 2. Literature matrix

Scale is the source's experimental corpus/table size where verified; it is not a claim about this project. “—” means not applicable or not specified. Author lists use “et al.” where lengthy. Venue versions matter: preprints and final proceedings can report different results.

| Paper / system; authors | Year / venue | Task; dataset scale | Blocking method | Matching model | Hard negatives | Graph method | Key result / limitation | Mechanism adopted |
|---|---|---|---|---|---|---|---|---|
| [A Theory for Record Linkage](https://nhis.ipums.org/nhis/resources/Fellegi69.pdf), Ivan P. Fellegi, Alan B. Sunter | 1969, JASA | Linkage theory; no scale claim | — | Likelihood ratio | — | — | Error-controlled linkage theory | Agreement rarity and abstention |
| [String distance comparison](https://www.cs.utexas.edu/~ai-lab/pubs/ijcai03.pdf), William W. Cohen, Pradeep Ravikumar, Stephen E. Fienberg | 2003, IIWeb workshop at IJCAI | Name matching; small benchmark collections | Given comparisons | String metrics / learned combination | — | — | SoftTF-IDF strong on evaluated tasks | Weighted fuzzy tokens |
| [Correlation Clustering](https://www.cs.cmu.edu/~shuchi/papers/clusteringfull.pdf), Nikhil Bansal, Avrim Blum, Shuchi Chawla | 2004, Machine Learning; FOCS 2002 predecessor | Signed-graph theory | — | — | — | Agreement/disagreement optimization | Complete-graph theory does not solve noisy sparse linkage automatically | Preserve negative evidence |
| [Exact F-measure maximization](https://www.weiweicheng.com/research/papers/cheng-nips11.pdf), Krzysztof Dembczyński, Willem Waegeman, Weiwei Cheng, Eyke Hüllermeier | 2011, NeurIPS/NIPS | Multilabel inference | — | Joint label/count probabilities | — | — | Marginals alone can be insufficient | Empty-set utility and set decisions |
| [GPU similarity search / FAISS](https://arxiv.org/pdf/1702.08734), Jeff Johnson, Matthijs Douze, Hervé Jégou | 2017 preprint; IEEE TBD online 2019 | 95M images; billion-vector search | IVFADC/PQ | — | — | ANN neighborhoods | Batched compressed search at scale | CPU/GPU search choices |
| [HNSW](https://arxiv.org/pdf/1603.09320), Yury A. Malkov, Dmitry A. Yashunin | 2016 preprint; TPAMI 2020, online 2018 | Million-scale; 200M SIFT comparison | Proximity graph search | — | — | ANN graph only | Speed/recall versus memory | Pilot HNSW against IVF |
| [Calibration of modern neural networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger | 2017, ICML | Image/document classifiers | — | Post-hoc calibration | — | — | Temperature scaling useful in tested settings | Cheap held-out calibration |
| [Beta calibration](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf), Meelis Kull, Telmo Silva Filho, Peter Flach | 2017, AISTATS | Binary classifiers | — | Parametric probability mapping | — | — | More flexible than logistic mapping | Tree calibration candidate |
| [Ditto](https://arxiv.org/pdf/2004.00584), Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan | 2020, PVLDB | Pairwise EM; company tables originally 789k / 412k | ZIP / TF-IDF; optional SBERT | Transformer cross-encoder | Similarity-biased labeled pairs | — | Strong pair F1; not exhaustive search | Typed serialization, selective reranking |
| [ZeroER](https://chu-data-lab.github.io/downloads/ZeroER-SIGMOD2020.pdf), Renzhi Wu et al. | 2020, SIGMOD | Five benchmarks; largest table 64,263 | Optional blocks | Regularized mixture model | No labels | Transitivity constraints | Distribution assumptions matter | Probabilistic diagnostic baseline |
| [ScaNN](https://proceedings.mlr.press/v119/guo20h/guo20h.pdf), Ruiqi Guo et al. | 2020, ICML | GloVe 1.2M; Amazon-670k | Anisotropic vector quantization | — | — | — | Preserves important inner products | Compression alternative |
| [DeepBlocker design exploration](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf), Saravanan Thirumuruganathan et al. | 2021, PVLDB | Structured/textual/dirty ER benchmarks | Learned embeddings + nearest neighbors | Multiple encoder designs | Auxiliary/self-supervised objectives | — | Dense design choices matter | Separate encoder and ANN quality |
| [Supervised Contrastive Learning for Product Matching](https://arxiv.org/pdf/2202.02098), Ralph Peeters, Christian Bizer | 2022, WWW Companion | Product matching; ~3k–68k training pairs | Given candidate pairs | RoBERTa SupCon + pair head | Source-aware in-batch | Positive-label components | Assumes internally deduplicated sources | Multi-positive masks |
| [HierGAT](https://doi.org/10.1145/3514221.3517872), Dezhong Yao, Yuhong Gu, Gao Cong, Hai Jin, Xinqiao Lv | 2022, SIGMOD | Magellan/WDC; DI2KG 29,788 cameras / 16,663 monitors | Preblocked; collective top16 | Hierarchical graph attention | Given candidate pairs | Attribute hierarchy; HierGAT+ context | Pairwise graph differs from collective graph | Field and candidate context |
| [Sudowoodo](https://arxiv.org/pdf/2207.04122), Runhui Wang, Yuliang Li, Jin Wang | 2023, ICDE | Standard ER; usual largest table ~64k | Contrastive embeddings | RoBERTa; optional pair head | TF-IDF-clustered batches | — | Useful with few labels; modest evaluated scale | Cheap lexical hard-negative curriculum |
| [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf), Derek Paulsen, Yash Govind, AnHai Doan | 2023, PVLDB | WDC 26M; MusicBrainz 20M; business catalog 2.5M | Lucene sparse top-K | Downstream matcher separate | — | — | Large-scale sparse baseline; field choice can fail | Sparse-first indexing |
| [UniBlocker](https://arxiv.org/pdf/2404.14831), Tianshu Wang et al. | 2024, arXiv | 19 datasets / 9 domains; unlabeled scale to 1.8M vs 2.5M | Universal dense + sparse hybrid | Attribute-aware encoder | In-batch; heuristic positives | — | Mostly ~90% completeness operating targets | Field awareness; cautious rescue |
| [SC-Block](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf), Alexander Brinkmann, Roee Shraga, Christian Bizer | 2024, ESWC | WDC-Block 100k vs 2M; 3,259 shops | SupCon RoBERTa + FAISS | Separate matcher | Source-aware in-batch | Label components | Final paper: ~50% smaller sets, 1.5–4× pipeline speedup | Train retrieval to shrink K |
| [Blocking reproducibility study](https://aclanthology.org/2024.naacl-long.483.pdf), Runhui Wang, Yongfeng Zhang | 2024, NAACL | WDC subsets to 500k | BM25, Sparkly, neural IR blockers | Retrieval comparison | Method-dependent | — | No universal neural winner | Test strong IR baselines |
| [WDC Products](https://www.openproceedings.org/2024/conf/edbt/paper-14.pdf), Ralph Peeters, Reng Chiz Der, Christian Bizer | 2024, EDBT | 11,715 offers / 2,162 products; 27 variants | Controlled benchmark | Symbolic and neural matchers | Explicit hard-case axis | — | Unseen entities alter model rankings | Entity-disjoint stress tests |
| [AttendEM](https://dmas.lab.mcgill.ca/fung/pub/LFX24kbs_preprint.pdf), Jwen Fai Low, Benjamin C. M. Fung, Pulei Xiong | 2024, Knowledge-Based Systems | 13 ER-Magellan datasets | Given pairs | Transformer ensemble | Existing negatives | — | Small gain versus original Ditto | Matched-budget model comparisons |
| [Beyond Scale and Generation](https://arxiv.org/pdf/2607.24688), Zeyu Zhang et al. | July 2026, arXiv preprint | Nine datasets; 1,215 runs, >850 H100 hours | Given pairs | Bi/cross/generative comparison | Benchmark-dependent | — | Scale and initialization confound rankings | Small controlled neural pilot |
| [ER in Practice](https://arxiv.org/pdf/2607.26298), Kaushik Pavani et al. | July 2026, arXiv preprint | Six benchmarks, 864–5M | HNSW + identifiers | Model tournament | Benchmark-dependent | Verified merges | Verification can still reduce F1 | Graph veto ablation only |
| [Splink, MoJ](https://www.gov.uk/government/publications/joined-up-data-in-government-the-future-of-data-linking-methods/splink-mojs-open-source-library-for-probabilistic-record-linkage-at-scale) | 2020, official government engineering report | Applications around 15M | Configurable blocking | Probabilistic comparisons | EM, no required labels | Optional clustering | Scale depends on blocking/compute | Frequency-adjusted evidence |
| [Overture Places](https://docs.overturemaps.org/guides/places/) | 2026, official living documentation | ~81M published places; match test size undisclosed | Spatial + learned blocker | XGBoost | Training details undisclosed | Transitive clustering | Blocking bottleneck; confidence ≠ match probability | Place-specific feature/context design |

Full-text PDFs are archived in **research_papers/** with source URLs and SHA-256 hashes in **manifest.json**. The Faiss library overview and the 2023 embedding analysis are supplemental PDFs beyond the matrix. HierGAT's author-uploaded full text was read, but its publisher PDF could not be downloaded through the available endpoints; that access limitation is recorded rather than claiming a local PDF exists.

## 3. Deep reading: what transfers and what does not

The source-specific summaries in this section are deliberately compact. Proposed modifications are research designs, not attributed paper algorithms.

### 3.1 Transformer and graph matchers

**Ditto — actual details.** Records use [COL] field [VAL] value inside a joint [CLS]/[SEP] pair. A Transformer representation feeds binary classification. Domain preprocessing includes typed spans, numerical rewriting and abbreviation dictionaries; TF-IDF summarization preserves informative tokens. MixDA blends original and augmented representations and needs two training forwards. Augmentations include span/attribute deletion or shuffling and pair swapping. Company labeling samples 10k pairs from each blocker with a similarity bias. There is no fixed negatives-per-positive curriculum or singleton-specific threshold algorithm. Its company pipeline begins with 10.65M candidates. [Full text, §§2–5](https://arxiv.org/pdf/2004.00584)

**Proposed adaptation / complexity.** Preserve exact identifiers outside the encoder, avoid augmenting away branch numbers, and use 128–192 pair tokens only after feature scoring. Dense attention contributes \(O(L^2)\) per pair; the whole model also has substantial projection/feed-forward work. Cross-encoding scales with the number of pairs, not unique records. Test serialization and the feature baseline before paying for additional forwards.

**SupCon product matching — actual details.** RoBERTa receives typed field serialization. An offer and a known positive form two views; different entity labels in the batch are negatives. The paper uses contrastive batch size 1024, temperature .07 and 200 epochs, then BCE on \((u,v,|u-v|,u\odot v)\). Freezing often helps. Source-aware sampling presumes deduplicated sources; training/validation positives are combined in its representation stage. [Full text](https://arxiv.org/pdf/2202.02098)

**Proposed adaptation / limitation.** Use training-fold labels only, mask every known sibling, and treat unknown neighbors as unlabeled. S2/S3 violate the source-deduplication premise. Restrict tuning to one warm-up and one refresh; the published training duration is not the budget default.

**HierGAT — actual details.** Token nodes aggregate into attributes and entities; the pair model uses BCE. Missing attributes use NAN. HierGAT+ adds query-candidate context/alignment; collective evaluation uses TF-IDF top16, query-disjoint splitting, ten epochs and whole-query batches. Iterative hard-negative refresh is not specified. Comparisons mainly use basic Ditto. [Primary publication](https://doi.org/10.1145/3514221.3517872)

**Proposed adaptation / limitation.** Import field comparison, common-token suppression and candidate-context features into the tree. Pairwise hierarchical attention does not justify global connected components. Candidate-context graph cost grows with the retained graph, potentially quadratically within a candidate set; avoid the full neural graph in the funded path.

**ZeroER — actual details.** An EM-fitted match/nonmatch Gaussian mixture uses attribute feature groups, shared correlations and adaptive regularization. Transitivity is constrained during learning/inference; no supervised negative mining is needed. The posterior decision threshold is .5. [Full text](https://chu-data-lab.github.io/downloads/ZeroER-SIGMOD2020.pdf)

**Proposed adaptation / limitation.** A small unlabeled comparator can expose feature/prior problems. Gaussian assumptions are questionable for our binary conflicts and heavy-tailed distances; use trees when complete labels exist. Neither .5 nor transitivity is transferable without entity-level validation.

**AttendEM / WDC Products.** AttendEM combines multiple text arrangements and aggregator outputs, adding inference work; its gain is only .21% against the original Ditto results, versus larger gains against replications. WDC Products varies hard cases, unseen entities and development size; its results show that seen-entity contrastive success need not generalize. These are matcher studies, not 10M retrieval tests. [AttendEM](https://dmas.lab.mcgill.ca/fung/pub/LFX24kbs_preprint.pdf), [WDC Products](https://www.openproceedings.org/2024/conf/edbt/paper-14.pdf)

### 3.2 Blocking algorithms

**Sparkly — actual details.** Manual field concatenation defaults to character trigrams; automatic configuration searches attribute/tokenizer sets using discriminativeness and pruning. Lucene performs sparse top-K probing, with query-side TF/IDF variants to address BM25 asymmetry. MusicBrainz 20M automatic recall is 95%@10, 97%@25, 98%@50. Business-catalog automatic recall is 79/89/94 at those K values, versus manual 99/100/100. WDC26M has no gold recall. [Full text](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)

**Proposed adaptation / limitation.** Compare manual business-aware fields with learned field weights separately by country. Cost follows postings touched, not simply corpus size; common names can create expensive queries. Historical hardware costs are excluded from our SageMaker estimates.

**UniBlocker — actual details.** It concatenates values, adds averaged attribute-name embeddings to token embeddings and restarts positions per field. Mean pooling yields record vectors. Initial backbone: all-mpnet-base-v2, length 256. Augmented/dropout views and in-batch pairs above string similarity .85 become positives. Multi-positive Circle loss uses scale 80, margin .4; batch 128, 10k steps, AdamW 1e-5. Other batch pairs act as negatives; no ANN refresh is specified. [Full text](https://arxiv.org/html/2404.14831v2)

**Proposed adaptation / limitation.** Test compact and typed serialization first. Do not pseudo-label same-brand branches positive based on high string similarity; this could erase precisely the distinctions needed here. Universal pretraining and its English MPNet backbone are not the multilingual budget default.

**SC-Block — actual details.** Typed attributes feed RoBERTa-base; normalized mean-pooled vectors have dimension 768. Known-positive components supply labels. Source-aware pools and duplicated samples guarantee positives. Training uses batch 1024, temperature .07, 20 epochs, learning rate 5e-5; cosine neighbors come from FAISS. Negatives change through batching, not a fixed ANN-mined list. [Final ESWC full text](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf)

**Proposed adaptation / limitation.** Use multi-positive learning with fold-safe labels, but do not assume S2/S3 are deduplicated. Evaluate final compression retention, not just encoder loss. Build/query complexity depends on the chosen ANN index.

**Sudowoodo — actual details.** RoBERTa learning combines contrastive and Barlow Twins losses with text augmentation and embedding cutoff. Cached TF-IDF k-means clusters create difficult batches; cluster/record order is reshuffled each epoch. Defaults include 10k sampled records, three pretraining epochs, batch 64 and learning rate 5e-5; cluster counts 30/60/90/120 are searched. False-negative risk is investigated, not eliminated. [Full text](https://arxiv.org/pdf/2207.04122)

**Proposed adaptation / limitation.** Approximate lexical clusters with country/city/name buckets or cached sparse neighbors. This avoids clustering millions of records solely for mining. Keep easy negatives and mask known positives.

**DeepBlocker / reproducibility study.** DeepBlocker explores alternative aggregation, reconstruction and contrastive/triplet-style representations rather than one mandatory Transformer recipe. The later NAACL study compares IR backbones including Contriever, SPLADE and ColBERT at K=1…20 and up to 500k records; it caps large WDC positive clusters at 20. Its rankings and capped multiplicity do not establish unrestricted business recall. [DeepBlocker](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf), [NAACL study](https://aclanthology.org/2024.naacl-long.483.pdf)

### 3.3 Foundations and current contradictory evidence

Fellegi–Sunter ranks comparison patterns by match/nonmatch likelihood ratio; rare agreements can be stronger evidence than common agreements. Factorized implementations add assumptions about attribute dependence. Cohen et al. combine token weights with approximate token matches; this motivates a feature, not a universally optimal matcher. [Fellegi–Sunter](https://nhis.ipums.org/nhis/resources/Fellegi69.pdf), [string metrics](https://www.cs.utexas.edu/~ai-lab/pubs/ijcai03.pdf)

GFM separates candidate selection at each output cardinality from choosing that cardinality, using joint label/count probabilities plus the probability of an empty truth. The original paper studies F1; Section 6 derives the analogous F0.5 expression. Estimating these distributions may be harder than fitting a robust threshold policy. [GFM](https://www.weiweicheng.com/research/papers/cheng-nips11.pdf)

The 2026 model study uses more than 850 H100 hours and finds architecture/initialization and distribution shift matter alongside size. Its hardware numbers cannot be transferred to a T4. The 2026 self-serve preprint is preliminary: single-seed evaluation, no significance tests, and its own MusicBrainz table drops pair F1 from .540 before merging to .277 after verified merging. Use neither as proof of a cheap guaranteed neural/graph gain. [Model study](https://arxiv.org/pdf/2607.24688), [self-serve study](https://arxiv.org/pdf/2607.26298)

## 4. Representation, retrieval and hard-negative design

### 4.1 Normalization and serialization

**Proposed canonical schema:** record_id, source, country, raw_name, canonical_name, alternate_names, script-preserving tokens, transliteration variants, house_number, unit, street, locality, city, state, postal_code, latitude, longitude, coordinate_quality, phone_set, domain, URL_path, categories, field_missingness, parser/provenance/version fields.

Normalize Unicode and whitespace without destroying originals. Produce accent-folded/transliterated retrieval variants in addition to native-script forms. Expand abbreviation dictionaries by field and locale; “st” in a street and a business name need different treatment. Keep legal suffix removal as an alternate feature, not irreversible normalization. US ZIP+4, French five-digit postcodes and Indian six-digit PINs remain strings; preserve leading zeros. House ranges, bis/ter suffixes, floors, units and sectors remain typed pieces. Parse rather than globally deleting punctuation.

Use [libpostal](https://github.com/openvenues/libpostal) as an optional address parser, validated separately for US/France/India; its reported held-out parsing accuracy is not a guarantee for these listings. Phone parsing uses a pinned [libphonenumber](https://github.com/google/libphonenumber) version, country context, base numbers and separately stored extensions. Use a base-number retrieval channel when extensions differ or are missing. Validation does not prove ownership. Shared call-center numbers and chain domains are frequency-weighted features; branch-specific URL paths can be more useful than the domain.

Compare these encodings on the same training split:

| Representation | Strength | Risk / cost | Proposed use |
|---|---|---|---|
| Typed “[COL] name [VAL] …” | Explicit schema, field missingness | Extra tokens; positional sensitivity | Cross-encoder baseline |
| Compact “name: …; street: …; city: …” | Short, tokenizer-friendly | Less formal field boundaries | Cheap encoder baseline |
| Values only | Compact, robust to renamed schema | Field ambiguity | Retrieval ablation |
| Field-specific encoders with mean pooling | Reuses distinct field signals | More forward passes/storage | Sample-only comparison |
| Weighted field embeddings | Controllable relevance | Weight/normalization choices | Combine normalized name/address vectors if already cached |
| Learned attribute-aware encoder | Can preserve semantics across schemas | Training and integration cost | Optional pilot, not required |

Field order can affect ordinary Transformers through positions and truncation. Use fixed order (name, alternate name, street/house/unit, locality/city/state/postal, category, contact), per-field token budgets, and a controlled order-shuffle ablation. Do not claim permutation invariance merely because shuffling was used during training.

**Numeric disambiguation proposal.** Retain typed tokens such as HOUSE_102, SECTOR_14 and name-number sequences; supply numeric equality/conflict indicators to the tree and optional reranker head. Number masking is an ablation only: it can erase the difference between Hotel21 and Hotel12. Digit embeddings or character modules need specific evidence to justify another model. Numeric sequence edit distance helps detect transposition but does not imply two identifiers are equal. Never round identifiers. Reliability-conditioned conflicts may penalize compression scores; use a hard veto only for validated, semantically impossible combinations.

### 4.2 Independent retrieval channels

Country is the safe hard partition supplied by the task. Other fields are noisy and should generally supply alternative channels rather than mandatory conjunctions.

| Channel | Proposed query / initial quota | Main value | Failure mode / control |
|---|---|---|---|
| Phone | Exact normalized phone; retain rare-key hits | Strong identity clue | Shared/recycled number; bucket frequency and address checks |
| Website | Domain + branch path, or rare domain + name | Contact evidence | Chain domains are hubs |
| Address anchor | House + rare street token + locality/postal variant | Branch discrimination | Parser errors; fallback channels |
| Postal / geo | Postal plus name, or cell plus neighboring cells | Local distractor control | Never postcode/cell alone as identity evidence |
| Rare name token | Top 1–3 informative tokens, corroborated by locality/category | Distinctive names | Misspellings; query character alternatives |
| Sparse name | Character 3/4-gram and token BM25 top40 | Typos / aliases | Frequent names; upper-bound pruning |
| Sparse address | Street/locality weighted retrieval top20–40 | Name changes | Shared buildings |
| Dense rescue | Top20–40 per query after measured gate | Cross-script/semantic aliases | Same-brand branch collapse |
| Optional phonetic | Locale-specific phonetic key plus location | Pronunciation variants | No universal English phonetic rule for India |

Merge channel results by stable external record ID, retain channel masks and original ranks, and apply a cheap compressor. Start with average output 20–30, 95th percentile at most 40; allow a monitored overflow queue for credible high-multiplicity queries. Never collapse distinct external IDs in the final ground-truth count. Byte-identical records can share encoded content only if their full ID expansion is preserved.

Protect strong rare anchors within the compressor, while learning whether their evidence is actually reliable. Score a bounded union of approximately 100–200 proposals using cheap token/contact/number features; expensive fuzzy features apply only after compression. This temporary cap is also a parameter to validate. Oversized postings require secondary keys or a fallback query, not arbitrary truncation silently presented as high recall.

**Candidate recall accounting.** Report pair-weighted recall, macro recall over non-singletons, all-matches coverage, and any-match coverage. Any-match coverage alone is insufficient. Measure every channel union, pre-compression set, final cap and overflow separately. For non-singletons, record the oracle ceiling \(1.25t_i/(0.25m_i+t_i)\); for true singletons the oracle score is1. Here \(t_i=|Y_i\cap C_i|\), even with a perfect downstream classifier.

Sparse query cost is proportional to postings visited with top-K pruning; common-token queries can approach a scan. ANN also lacks a universal worst-case sublinear guarantee. Country sharding, selective terms and quotas make expected work manageable; benchmark tail latency and postings touched.

### 4.3 Retriever comparison and ANN configuration

| Method | Transferable advantage | Limitation at this operating point | Decision |
|---|---|---|---|
| BM25 / inverted tokens | Fast, inspectable rarity evidence | Token mismatch, length/asymmetry effects | Default channel |
| Character TF-IDF | Spelling noise and multilingual surface overlap | Many postings; 4-grams brittle for short names | Default, field-limited |
| MinHash / LSH | Cheap approximate Jaccard | Short records and low lexical overlap; collision tuning | Small ablation only |
| Locality-sensitive/geographic keys | Strong physical locality | Coordinate errors and boundaries | Neighbor cells + fallback |
| Dense bi-encoder | Alias and script generalization | Semantic similarity is not identity | Gated rescue |
| FAISS Flat | Exact vector-neighbor audit | \(O(Nd)\) per query | Small-query audit only |
| HNSW | Strong latency/recall trade-off | FP32 vector/link RAM; build overhead | RAM-rich country pilot |
| IVF / IVFFlat / SQ8 | Tunable search, predictable sizes | Coarse-cell misses | CPU default dense index candidate |
| IVF-PQ | Small index | Quantization can reorder close branches | Retrieve100–200 then exact FP16 rescore |
| ScaNN | Score-aware quantization | Integration and hardware tuning | Substitute only after pilot win |
| SPLADE / ColBERT | Learned sparse / token interaction | Training, postings/vector multiplication | Optional larger-budget research |

FAISS is a library containing several index types; HNSW is a proximity index, not an entity matching graph. [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [ScaNN paper](https://proceedings.mlr.press/v119/guo20h/guo20h.pdf)

**Initial pilot parameters, not tuned values:** normalize vectors for cosine/IP; per-country IVF nlist in {4096, 16384}, nprobe in {16,32,64,128}; train quantizers on a reproducible representative sample, normally 40–100 samples per centroid where affordable. Use PQ48×8 bits for d=384, or SQ8 if compression loss is material. HNSW M=16/32, efConstruction=100/200, efSearch=64/128/256. Measure exact-neighbor recall on stratified queries against the full country's vectors, and semantic gold recall separately. Border cells and missing geodata must have rescue routes.

Do not blindly truncate 384-dimensional vectors to128/256: train/validate a projection or use an encoder explicitly supporting that dimension. Retain an FP16 vector artifact for rescore and audits.

### 4.4 Proposed negative curriculum

Counts below are proposed engineering defaults; the published recipes are in Section 3.

| Strategy | Generation | Retained per positive | Refresh | Label-noise control | Additional cost |
|---|---|---:|---|---|---|
| Easy same-country | Uniform verified different entities | 2 | Batch reshuffle | Complete entity labels | Negligible sampling |
| BM25 / character | Actual retrieved near neighbors | Up to4 | Cached warm-up list | Exclude every known sibling | One sparse retrieval pass |
| Same city/postal | Location-conditioned neighbors | Up to2 within total quota | Static | Location does not establish nonmatch | Indexed lookup |
| Same name / chain | Same normalized name, different verified branch | Up to2 | Static + failures | Require reviewed incompatible identity | Small joins |
| ANN | Embedding neighbors | Up to4 replacing weaker negatives | Once after warm-up | Unknown ≠ negative | Sample queries, no epoch-wide refresh |
| Model false positives | Highest-scoring verified errors | Up to4 +2 diverse | Once or drift-triggered | Train-fold errors only | Cached feature scoring |
| Singleton hard cases | Top retrieved candidates of verified singletons | Up to8 per singleton | Once | Singleton verification required | High expected value per label |

Cap total negatives per anchor (e.g.12), rather than multiplying by every positive when a business has many listings. Keep all known positives; sample very large positive groups with explicit coverage accounting. Contrastive batches use only same-country negative comparisons and multi-positive masks. Cross-country negatives are easy, unavailable at inference, and waste learning capacity.

Mining stores IDs, scores, channel provenance and label status, not copied strings. Use 10k–100k labeled anchors for a first curriculum, write sharded Parquet, and refresh only the sampled pool. Missing-positive labels are the principal risk. Soft heuristic positives based on name overlap are especially unsafe for chains. Calibration and thresholds later use representative complete candidate groups, not this sampled class balance.



## 5. Feature dictionary and model comparison

### 5.1 Feature conventions

All proposed features below are computed from canonical fields and cached token statistics. Define \(E(a,b)=1[a=b]\) only when both fields are present and valid; otherwise NA. Missing does not equal missing. For every field pair retain the categorical availability state {both, left-only, right-only, neither}; trees receive numeric NA plus these states. Missing retrieval-channel scores are NA with a retrieved-channel flag; missing reciprocal ranks are zero. Country/source are categorical context, never raw entity IDs.

Let \(L(a,b)\) be Levenshtein distance; \(DL\) Damerau–Levenshtein distance with the implementation variant pinned; \(l_a\) string length; \(A,B\) token sets; \(W(A)=\sum_{t\in A}w_t\). Clip IDF at a training-selected percentile so corrupted singleton tokens cannot dominate. Compute IDF by field/country over the deployment index; this is label-free transductive information, declared in evaluation.

| # | Feature | Formula / definition | Field | Range | Missing handling | Expected signal |
|---:|---|---|---|---|---|---|
| 1 | Levenshtein similarity | \(1-L/\max(l_a,l_b)\) | Name | [0,1] | NA if either absent | Spelling agreement |
| 2 | Damerau similarity | \(1-DL/\max(l_a,l_b)\) | Name | [0,1] | NA | Transpositions |
| 3 | Jaro | Standard matched-character/transposition score | Name | [0,1] | NA | Short-name similarity |
| 4 | Jaro–Winkler | \(J+\min(prefix,4)\cdot.1(1-J)\), gated at J>.7 | Name | [0,1] | NA | Prefix-supported similarity |
| 5 | Character 3-gram cosine | \(v_3(a)^\top v_3(b)/(\|v_3(a)\|\|v_3(b)\|)\) | Name | [0,1] | NA for zero vector | Robust local overlap |
| 6 | Character 4-gram cosine | Same with 4-grams | Name | [0,1] | NA for zero vector | More specific overlap |
| 7 | Token Jaccard | \(\vert A\cap B\vert /\vert A\cup B\vert \) | Name | [0,1] | NA if absent/empty | Token agreement |
| 8 | Weighted Jaccard | \(W(A\cap B)/W(A\cup B)\) | Name | [0,1] | NA | Rare-token agreement |
| 9 | Symmetric SoftTF-IDF | Average of two directed fuzzy TF-IDF scores | Name | ≥0; implementation-dependent upper bound | NA | Fuzzy rare tokens |
| 10 | Token containment | \(\vert A\cap B\vert /\min(\vert A\vert ,\vert B\vert )\) | Name | [0,1] | NA | Short/long aliases |
| 11 | Acronym equality | Compare acronym of longer name with short form | Name | {0,1} | NA when inapplicable | Initialisms |
| 12 | Phonetic equality | Locale-specific code equality | Name | {0,1} | NA for unsupported locale | Pronunciation variants |
| 13 | Length ratio | \(\min(l_a,l_b)/\max(l_a,l_b)\) | Name | [0,1] | NA | Abbreviation / missing pieces |
| 14 | Prefix ratio | Common-prefix length / min length | Name | [0,1] | NA | Shared start; chain warning |
| 15 | Alternate-name best overlap | Max weighted Jaccard over valid alias pairs | Names | [0,1] | NA if no comparable names | Known aliases |
| 16 | Street similarity | Character cosine or normalized edit score | Street | [0,1] | NA | Same thoroughfare |
| 17 | House equality | \(E(h_a,h_b)\), including suffix/range normalization | House | {0,1} | NA | Same premises |
| 18 | House difference | \(\min(\vert int(h_a)-int(h_b)\vert ,10000)\) | House | [0,10000] | NA for non-simple numbers | Numeric distance, not identity |
| 19 | Address token Jaccard | Token set overlap | Address | [0,1] | NA | General address agreement |
| 20 | Address weighted overlap | Weighted Jaccard | Address | [0,1] | NA | Distinctive street/locality |
| 21 | Rare address overlap | Sum clipped IDF of shared rare tokens | Address | ≥0 | NA if absent | Strong location anchors |
| 22 | Locality similarity | Character 3-gram cosine | Locality | [0,1] | NA | Neighborhood agreement |
| 23 | City equality | \(E(city_a,city_b)\) after alias map | City | {0,1} | NA | Administrative consistency |
| 24 | State equality | Canonical state/region equality | State | {0,1} | NA | Administrative consistency |
| 25 | Postal equality | Exact canonical string equality | Postal | {0,1} | NA | Locality evidence |
| 26 | Postal prefix equality | Country-specific meaningful prefix | Postal | {0,1} | NA | Coarser evidence |
| 27 | Unit equality | Canonical suite/floor/unit agreement | Unit | {0,1} | NA | Co-located businesses |
| 28 | House conflict | Both reliable and unequal normalized house IDs | House | {0,1} | NA if unavailable; separate reliability | Different branches |
| 29 | Postal conflict | Both reliable and unequal postcodes | Postal | {0,1} | NA | Location inconsistency |
| 30 | Phone conflict | Both valid phone sets nonempty and disjoint | Phone | {0,1} | NA | Contact disagreement |
| 31 | Name-number conflict | Distinct typed numeric/alphanumeric sequences | Name | {0,1} | NA if no comparison | Hotel21 vs Hotel12 |
| 32 | Sector/unit numeric conflict | Reliable typed subaddress mismatch | Address | {0,1} | NA | Sector14 vs Sector41 |
| 33 | Numeric-sequence similarity | Normalized edit score on typed sequence | Name/address | [0,1] | NA | Numeric typo vs identity |
| 34 | Weighted conflict score | \(\sum_f\lambda_f r_f I_f\) | Conflicts | ≥0 | Zero unavailable terms plus missing masks | Aggregate contradiction |
| 35 | Phone overlap | Nonempty normalized set intersection | Phone | {0,1} | NA | Shared contact |
| 36 | Phone rarity | Max IDF among shared valid phones | Phone | ≥0 | NA if absent; 0 if no shared phone | Downweight call centers |
| 37 | Domain equality | Exact registrable domain equality | URL | {0,1} | NA | Shared site |
| 38 | Domain rarity | IDF of shared domain; 0 if different | URL | ≥0 | NA if absent | Downweight chain domains |
| 39 | Branch-path similarity | Token overlap on meaningful URL path | URL path | [0,1] | NA if absent | Branch/store identity |
| 40 | Category compatibility | Frozen taxonomy similarity or equality | Category | [0,1] | NA | Supporting business type |
| 41 | Haversine distance | Great-circle distance | Coordinates | [0,~20015] km | NA if invalid | Location separation |
| 42 | Log distance | \(\log(1+d_{km})\) | Coordinates | ≥0 | NA | Stable distance scale |
| 43 | Distance bucket | 0–25m,25–100m,100–500m,.5–2km,>2km | Coordinates | Categorical | Missing category | Nonlinear geographic effects |
| 44 | Same geo cell | Equality at fixed documented resolution | Coordinates | {0,1} | NA | Coarse local support |
| 45 | Same admin region | Trusted region-ID equality | Region | {0,1} | NA | Geographic context |
| 46 | Coordinate quality | Min documented quality; centroid flag categorical | Coordinates | [0,1] / category | Unknown category | Reliability of distance |
| 47 | Name BM25 | Field-specific indexed query score | Retrieval | ≥0 | NA when not retrieved | Sparse evidence |
| 48 | Address BM25 | Address channel score | Retrieval | ≥0 | NA | Address evidence |
| 49 | Dense similarity | Dot product of L2-normalized vectors | Retrieval | [-1,1] | NA when unavailable | Learned similarity |
| 50 | Sparse reciprocal rank | \(1/r_s\) | Retrieval | [0,1] | 0 when absent | Query-relative position |
| 51 | Dense reciprocal rank | \(1/r_d\) | Retrieval | [0,1] | 0 when absent | Query-relative position |
| 52 | Channel agreement | Number of independently defined channels | Retrieval | [0,R] | 0 | Corroboration, not independence proof |
| 53 | RRF | \(\sum_r1/(k_0+rank_r)\) | Retrieval | ≥0 | 0 absent contribution | Robust fusion |
| 54 | Top score | Max first-pass pair score in candidate set | Context | [0,1] | 0 for no candidates + empty flag | Plausibility of any match |
| 55 | Candidate rank | Rank by first-pass score with deterministic ties | Context | [1,K] | No pair row for empty set | Competition |
| 56 | Score minus best | \(s_{ij}-s_{i1}\) | Context | [-1,0] | No pair for empty set | Relative plausibility |
| 57 | Top-two gap | \(s_{i1}-s_{i2}\) | Context | [0,1] | NA for K<2 | Ambiguity, not uniqueness veto |
| 58 | Local score z | \((s_{ij}-\mu_i)/(\sigma_i+\epsilon)\) | Context | Real, clipped [-10,10] | 0 if constant; flag K<2 | Local outlier support |
| 59 | Candidate count | Count before/after compression as two columns | Context | Nonnegative integer | 0 | Density / truncation |
| 60 | High-score count | \(\sum_j1[s_{ij}>u]\), fixed OOF-selected u | Context | [0,K] | 0 | Multiplicity/ambiguity |
| 61 | Name frequency | log1p country and local name DF as two columns | Context | ≥0 | NA for absent name | Chain/common-name risk |
| 62 | Incompatible-rival gap | Score minus best reliably incompatible branch | Context | [-1,1] | NA without rival | Relevant competition |
| 63 | Top-two score ratio | \(s_1/(s_2+\epsilon)\), clipped at100 | Context | [0,100] | NA for K<2 | Secondary margin ablation |
| 64 | Field availability states | {both,left,right,neither} per schema field | Missingness | Categorical | Explicit neither | Avoid missing=agreement |
| 65 | Provenance context | Country, source pair, parse quality, index/model versions in audit | Context | Categorical | Unknown category | Country/source interactions |
| 66 | Overflow / no-candidate flag | Cap reached, overflow used, K=0 | Context | {0,1} each | Always observed | Retrieval failure awareness |

This is a superset dictionary; use feature-family ablations to remove expensive redundant metrics. Jaro, Jaro–Winkler and edit metrics are correlated. SoftTF-IDF or token bipartite alignment can be reserved for ambiguous compressed pairs. Compute first-pass score features out-of-fold for any second-pass/context model; otherwise the context learns training overconfidence.

### 5.2 Three model families

| Family | Proposed training | Calibration / missing fields | Inference and hardware | Research decision |
|---|---|---|---|---|
| LightGBM | Histogram tree classifier, limited depth/leaves, early stopping | Numeric NA plus explicit availability | CPU first; bounded trees, compiled prediction | Primary baseline |
| XGBoost | Histogram objective, same split/features/search budget | Learned missing direction; external-memory option if needed | CPU first; GPU only on profiled large dense training matrix | Direct competitor |
| CatBoost | Shallow/symmetric-tree candidate with source/country categories | Numeric missing handling; encode categorical missing explicitly | CPU comparison; costly categorical statistics can matter | Keep if OOF gain per dollar wins |
| Bi-encoder | Frozen multilingual model, then small contrastive pilot | Missing-field serialization, multi-positive loss | Offline GPU encoding; CPU ANN querying | Retrieval/rescoring signal |
| Cross-encoder | Short pair serialization, BCE; optional numeric side head | Field preservation and supervised calibration | GPU on selected difficult pairs | Conditional increment |

The official [LightGBM GPU guide](https://github.com/lightgbm-org/LightGBM/blob/main/docs/GPU-Performance.rst) says small datasets can be inefficient on GPU because of transfer overhead. [XGBoost external-memory documentation](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) makes I/O bandwidth a central constraint; out-of-core is not free speed. [CatBoost documentation](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing) distinguishes numeric from categorical missing-value processing.

**Proposed tree budget.** Compare one or two configurations per library on the same representative feature sample, then devote the five-fold run to the winner. Start 300–800 trees, 15–63 leaves or analogous depth, strong minimum-leaf support and early stopping; these are search bounds, not fitted settings. Weight references so high-multiplicity anchors do not dominate training. Include actual retrieved negatives and singleton groups. Class weights can aid optimization but distort raw probabilities; fit calibration on unsampled representative groups.

**Contrastive objective.** With \(P(i)\) all known positives in the training batch and \(D(i)\) admissible same-country examples excluding self and unknowns:
\[
L_i=-\frac1{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(z_i^\top z_p/\tau)}
{\sum_{a\in D(i)}\exp(z_i^\top z_a/\tau)}.
\]
Mask known positives from any “negative” designation, while retaining them in the appropriate multi-positive denominator. Gradient accumulation does not enlarge the current microbatch's in-batch negative pool. A queue/cache requires stale-embedding and false-negative controls.

**Frozen encoder comparison, verified dimensions / approximate parameters.** Throughput below is only a sensitivity scenario on one A10G at a 128-token cap, including no guaranteed benchmark. Encoding counts include 11.73M evaluation candidates plus references. Model metadata: [MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [mE5 small](https://huggingface.co/intfloat/multilingual-e5-small), [mE5 base](https://huggingface.co/intfloat/multilingual-e5-base), [BGE-M3](https://huggingface.co/BAAI/bge-m3).

| Backbone | Parameters | Dimension | Assumed records/s range | GPU hours for11.73M | g5.xlarge encoding dollars | FP16 weights / 10M vectors | Expected sparse increment |
|---|---:|---:|---:|---:|---:|---|---|
| multilingual MiniLM L12 | 117.65M | 384 | 500–1500 | 2.17–6.52 | $3.06–9.18 | ~.235GB / 7.68GB | Unknown; paraphrase baseline |
| multilingual E5-small | 117.65M | 384 | 500–1500 | 2.17–6.52 | $3.06–9.18 | ~.235GB / 7.68GB | Unknown; preferred retrieval hypothesis |
| multilingual E5-base | 278.04M | 768 | 200–600 | 5.43–16.29 | $7.65–22.94 | ~.556GB / 15.36GB | Unknown; require small-model miss recovery |
| BGE-M3 | ~568M | 1024 | 60–200 | 16.29–54.31 | $22.94–76.46 | ~1.136GB / 20.48GB | Unknown; long context has limited value for short listings |

Weights alone understate VRAM: activations, optimizer states and batches dominate many runs. Start inference batches128/256 for small encoders,64/128 for base,16/32 for BGE and adjust to measured memory. Fine-tuning microbatches16–32 with mixed precision and accumulation are more conservative. A 24GB A10G permits more batch headroom than a16GB T4; throughput determines economics.

E5 recommends query prefixes on both sides for symmetric tasks; compare that with asymmetric query/passage prompts if needed. General retrieval training is a useful initialization, but business-identity quality remains unmeasured. [E5 author instructions](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md)

**Five-fold budget safeguard.** The headline five-fold protocol can share a label-free frozen encoder/index. If a task-fine-tuned encoder is adopted, each outer fold needs training without its labels and a corresponding index/candidate evaluation. Reusing an encoder trained on held-out identities is leakage. The funded small fine-tuning pilot is exploratory; promote it only if the validation allowance covers the extra fold work or use frozen dense retrieval in the official final comparison.

### 5.3 Cross-encoder scale calculation

The full top5 cascade requires
\[
1.73M\times5=8.65M\ \text{pairs};
\quad \lceil8{,}650{,}000/256\rceil=33{,}790\ \text{batches}.
\]
At batch128 there are67,579 batches. Top10 doubles pair count and approximately doubles compute. All40 would require69.2M pairs.

| End-to-end throughput if measured | Hours for8.65M | g5.xlarge cost at$1.408/h |
|---:|---:|---:|
| 100 pairs/s | 24.03 | $33.83 |
| 200 pairs/s | 12.01 | $16.92 |
| 500 pairs/s | 4.81 | $6.77 |
| 1000 pairs/s | 2.40 | $3.38 |

These are inference arithmetic, excluding training/startup/calibration. Throughput depends on backbone, sequence length, padding, CPU tokenization and batch size. At200 pairs/s, reranking500k pairs costs .694GPU hours or about$0.98. The proposed4h neural allocation covers pilot/adaptation plus a capped difficult subset, not an assumed blanket top5 pass.

Select hard cases using feature-model uncertainty, unreliable contacts, chain ambiguity and disagreement between sparse/dense evidence. Retain accepted high-confidence tree matches outside this subset; calibration must account for whether a pair was selected for reranking. A reranker must not force the whole prediction set to contain only its top5–10 pairs when more real matches exist.

## 6. Mathematical formulation, calibration and decisions

### 6.1 Similarities and retrieval scores

The following formulas define proposed features precisely.

Country/field-specific IDF and weighted Jaccard:
\[
w_t=\min\left(w_{\max},\log\frac{N_c+1}{df_c(t)+1}\right),\qquad
J_w(A,B)=\frac{\sum_{t\in A\cap B}w_t}{\sum_{t\in A\cup B}w_t}.
\]
If the denominator is zero, return NA with a missing/zero-information flag. Do not award agreement for two empty strings.

BM25 with field-specific length normalization:
\[
BM25(q,d)=\sum_{t\in q}
\log\left(1+\frac{N-df(t)+.5}{df(t)+.5}\right)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1(1-b+b|d|/\overline{|d|})}.
\]
Use separate name/address fields and fit their weights on training folds. BM25 is a retrieval score, not a probability; cross-query raw comparisons require context. Lucene's documented default parameters are a starting point, not tuned business settings. [Lucene BM25](https://lucene.apache.org/core/7_6_0/core/org/apache/lucene/search/similarities/BM25Similarity.html)

Let \(v_A(t)\) be L2-normalized TF-IDF and \(m(t)=\arg\max_{u\in B}\mathrm{JW}(t,u)\). A directed SoftTF-IDF feature is
\[
S(A,B)=\sum_{\substack{t\in A\\ \mathrm{JW}(t,m(t))>\theta}}
v_A(t)v_B(m(t))\mathrm{JW}(t,m(t)).
\]
Average both directions for a symmetric feature. Reuse of the same target token means it is not automatically a bounded cosine metric. As a costlier proposed alternative, maximize \(\sum_{(t,u)\in M}\sqrt{w_tw_u}\,\mathrm{sim}(t,u)\) over a one-to-one token matching \(M\), normalized by \(\sqrt{W(A)W(B)}\). Short-token Hungarian alignment costs roughly cubic time in token count. Monge–Elkan averages each token's best counterpart and is cheaper but permits many-to-one alignment. [String-metric study](https://www.cs.utexas.edu/~ai-lab/pubs/ijcai03.pdf)

Dense similarity, fusion and relative scores:
\[
z(x)=f_\theta(x)/\|f_\theta(x)\|_2,\quad s_d(x,y)=z(x)^\top z(y),
\]
\[
RRF(d)=\sum_{r:d\in C_r}\frac1{k_0+\mathrm{rank}_r(d)},\quad
\Delta_i=s_{i1}-s_{i2},\quad r_i=\frac{s_{i1}}{s_{i2}+\epsilon}.
\]
RRF is motivated by [Cormack, Clarke and Buettcher (SIGIR 2009)](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/). Use \(k_0=60\) as an initial fusion setting and validate it. Score ratios are meaningful only on a positive, fixed scale, not arbitrary signed logits.

Reliability-conditioned numeric conflicts:
\[
I_f=1[\text{both valid and incompatible on }f],\qquad
C_{ij}=\lambda_h r_hI_h+\lambda_p r_pI_p+\lambda_{\rm phone}r_{\rm phone}I_{\rm phone}
+\lambda_n r_nI_n.
\]
Supply individual conflicts and reliability to the tree. A compressor may use \(s_{\rm cheap}-C_{ij}\), but tune penalties to candidate recall, not final acceptance precision; penalizing twice can erase true matches.

With coordinates in radians, Earth radius \(R=6371.0088\)km:
\[
d=2R\arcsin\sqrt{\sin^2(\Delta\phi/2)+
\cos\phi_1\cos\phi_2\sin^2(\Delta\lambda/2)}.
\]
Clip the square-root argument to[0,1]. Invalid/centroid coordinates produce missing or quality-conditioned evidence rather than a confident geographic contradiction.

### 6.2 Calibration

Fit all calibrators on predictions made without training on the scored reference groups, using the deployed retrieval/selection distribution.

| Method | Definition | Strength | Limitation / decision |
|---|---|---|---|
| Platt | \(p=\sigma(az+b)\) | Two parameters; can shift prior | Parametric shape can be wrong |
| Temperature | \(p=\sigma(z/T),\,T>0\) | Cheap, preserves ranking | No intercept; cannot repair ranking or arbitrary prior shift |
| Isotonic | Monotone \(g(s)\), fitted by pooled adjacent violators | Flexible | Tail data hunger and ties |
| Beta | \(\sigma(a\log s-b\log(1-s)+c)\), \(a,b\ge0\) | Flexible monotone binary mapping; includes identity | Clip endpoints; regularize small strata |

Assess NLL, Brier score, reliability plots and precision calibration near the actual acceptance threshold. A low average calibration error can hide the extreme-tail errors that harm singletons. Calibration is followed by metric-specific decision tuning; it is not itself F0.5 optimization. [Guo et al.](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [Kull et al.](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf)

Train a small OOF stacker over tree/dense/optional cross-encoder scores if it beats the tree alone. Use a reranked flag and separate supported calibrations for selected/unselected pairs. Never treat a absent cross-encoder score as a negative prediction. Random negative subsampling changes class priors; representative calibration is preferred to a simple prior correction because hard-negative sampling is feature-dependent.

### 6.3 Singleton and set models

Let \(Y_i\) be the complete true external match set, \(C_i\) retrieved candidates, \(p_{ij}\) calibrated pair scores, and \(u_i\) pooled context. Proposed singleton model:
\[
q_i=P(|Y_i|>0\mid u_i),\qquad
P(\text{singleton}\mid u_i)=1-q_i.
\]
Train on one row per reference, including \(C_i=\varnothing\); label it from complete truth, not candidate presence. Context includes maxima, gaps, counts, channel agreement, rarity, conflicts and overflow. Include blocker-miss positives in training and report them separately.

Do not use \(1-\prod_j(1-p_{ij})\) as a calibrated any-match probability: conditional independence is implausible and adding distractors changes it mechanically. Do not use candidate softmax: multiple candidates can all be positive.

Compare:

- A: pair threshold only.
- B: singleton gate \(q_i\ge\tau_q\), then pair thresholds.
- C: jointly tuned set policy using \(q_i\), candidate support and conflicts.

A low-cost proposed policy is
\[
T_i=\mathrm{clip}(T_0+\alpha a_i+\beta c_i,0,1),
\]
\[
\widehat Y_i=
\begin{cases}
\varnothing,&q_i<\tau_q,\\
\{j\in C_i:p_{ij}\ge T_i,\ \mathrm{compatible}(i,j)\},&\text{otherwise}.
\end{cases}
\]
Here \(a_i,c_i\) are defined/calibrated ambiguity and reliable-conflict summaries; fit coefficients and all thresholds inside validation, with shrinkage toward the global rule. Hard compatibility exclusions require validated ontology. A global top1–top2 margin veto is excluded. An optional candidate-specific margin can compare the best reliably incompatible branch, not another plausible duplicate.

**Derived set-utility extension.** For output size \(k>0\), define
\[
A_{jk}=\sum_{m\ge1}\frac{1.25}{0.25m+k}
P(y_{ij}=1,\ |Y_i|=m\mid u_i).
\]
Choose the top \(k\) values in column \(k\), sum them, then compare all \(k\), including empty-set utility \(P(|Y_i|=0\mid u_i)\). The count \(m\) includes true matches outside \(C_i\). This extends the reasoning of [GFM](https://www.weiweicheng.com/research/papers/cheng-nips11.pdf) to F0.5; it is our derivation, not a result reported for business ER. Estimating the joint terms is difficult with sparse labels, so use this as an optional research comparator. Replacing the expectation of a ratio with a ratio of expectations is only an approximation.

### 6.4 Metric and exact global-threshold optimizer

For \(g_i=|Y_i|\), \(k_i=|\widehat Y_i|\), \(t_i=|Y_i\cap\widehat Y_i|\):
\[
F_{0.5}^{(i)}=
\begin{cases}
1,&g_i=k_i=0,\\
\frac{1.25t_i}{0.25g_i+k_i},&0.25g_i+k_i>0.
\end{cases}
\]
This count form avoids undefined precision/recall intermediates and includes false negatives outside retrieval. Equivalently,
\[
F_{0.5}=\frac{1.25TP}{1.25TP+FP+0.25FN}.
\]
Thus FP has four times the coefficient of FN in this denominator; this is not a universal fixed decision-cost ratio for macro F.

The exact threshold-family objective is
\[
t^*=\arg\max_t\,N^{-1}\sum_i F_{0.5}^{(i)}(\{j:p_{ij}\ge t\}).
\]
Sort all candidate events descending by score, start with every predicted set empty, and update only affected references when a score enters. Process ties as a group. Track the sum of entity scores, and evaluate after every distinct score. Complexity is \(O(M\log M)\) sorting plus \(O(M)\) updates for \(M=\sum_i|C_i|\); external sorting supports large M. This finds the optimum **within global threshold policies**, not all possible set predictions.

Tune group thresholds only with enough support; candidate-count, name-frequency, postal-availability, geography and ambiguity groups should be a small preregistered set. Sparse groups shrink to the global threshold. Fit conditional policies inside the outer fold, never on the same OOF labels used to advertise their gain. Include thresholds predicting all empty and all candidates. Do not assume .5 or the micro-F optimum applies.

### 6.5 Conservative graph refinement

The initial output is a set of independent reference-centered stars. Since S1 is deduplicated, a unique-reference constraint on each external record may be appropriate **only if branch-level ontology and labels confirm it**. It does not justify one-to-one S1–S2 or S1–S3 matching, since multiple external records are allowed.

One false bridge between clusters of sizes a and b creates ab implied cross-cluster co-memberships under closure. Shared mall addresses, chain domains, reception phones and common brands can form hubs. Connected components ignore negative evidence; [correlation clustering](https://www.cs.cmu.edu/~shuchi/papers/clusteringfull.pdf) illustrates an objective that uses both signs, but its complete-graph formulation is too large here.

**Proposed optional procedure:** consider only tiny ambiguous components (initial cap20 records), require direct high-confidence support to a reference for every accepted external record, veto reliable country/branch conflicts, and score all cross-pairs needed for a proposed merge. Unknown edges remain unknown. Never merge S1 anchors based only on noisy external bridges. Run one refinement pass and compare singleton false positives before/after. If all-pair verification is too costly, abstain rather than approximate with an unsafe transitive claim. The default funded prediction path skips graph refinement unless its gain survives held-out testing.



## 7. Validation protocol and experiment ledger

### 7.1 Five-fold grouped evaluation with realistic distractors

Use five outer folds grouped by Source1EntityID. Stratify groups approximately by country, singleton status, match multiplicity and attribute completeness without splitting a reference. Keep all external positives of a held-out reference out of supervised fitting, negative mining and task-specific encoder training. Retain the fixed external search corpus as unlabeled retrieval infrastructure when a transductive deployment protocol permits it.

**Leakage sensitivity tests:** use verified real-entity components if bad reference deduplication splits one entity across IDs; additionally evaluate brand/family-held-out, shared-address and repeated-phone strata. Do not put every common-phone/name record into one “true entity” group—shared contact is a risk marker, not proof of identity. External records appearing under multiple references require label/ontology review. Learned alias dictionaries, imputation rules and frequency-based threshold policies must not use validation outcomes.

For each fold, fit model/hyperparameters, calibrators, singleton model and thresholds using only the other four folds, with internal grouped separation. Held-out outer labels are used only to evaluate the locked pipeline. Save individual out-of-fold predictions so all ablations compare exactly the same references.

**Pool progression:** first pilot against at least500k–1M fixed unrelated external distractors, while including the natural evaluated truth set. Then confirm finalists against the full10M pool; small-pool metrics are not the headline result. Sweep pool size0.1M/1M/10M with a fixed reference sample to expose extreme distractors. Compute label-free index IDF over the applicable corpus and document this transductive choice.

The same deployed blocker/compressor must run during validation. Do not artificially inject positives missing from retrieval into the measured candidate set. For training the matcher, such positives may be sampled explicitly, with provenance; evaluation still counts their retrieval misses.

### 7.2 Metrics and fold reporting template

For non-singletons:
\[
R^{macro}_{cand@K}=\frac1{N_+}\sum_{i:g_i>0}
\frac{|Y_i\cap C_i^{(K)}|}{g_i}.
\]
Also report pair-weighted candidate recall, any-match recall and all-match coverage. Overall final macro F is the mean of per-reference F, **not** F computed from macro precision and macro recall.

For descriptive entity precision/recall, define \(P_i=TP_i/k_i\) if k>0, else1; \(R_i=TP_i/g_i\) if g>0, else1 for an empty prediction and0 otherwise. Report this convention prominently; the count-based F formula is authoritative. Pairwise P/R aggregate all emitted pairs and all complete gold pairs, including gold absent from retrieval.

Singleton accuracy \(=\#\{\text{true singletons predicted empty}\}/N_0\).
Two distinct safety measurements:
\[
FP_{1000}=1000\frac{\sum_{i:g_i=0}|\widehat Y_i|}{N_0},\qquad
Affected_{1000}=1000\frac{\sum_{i:g_i=0}1[|\widehat Y_i|>0]}{N_0}.
\]
If no predictions exist, pair precision is reported NA rather than silently awarding a perfect pair classifier.

| Outer fold | Candidate macro R@5 | @10 | @20 | @40 | Mean / median K | Pair P / R | Entity P / R | Macro F0.5 | Singleton accuracy | FP /1000 singletons |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |
| 2 | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |
| 3 | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |
| 4 | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |
| 5 | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |
| Pooled OOF / entity bootstrap95% CI | NR | NR | NR | NR | NR | NR | NR | NR | NR | NR |

Publish country/source pair, name-frequency, missingness, branch density and multiplicity slices. Include p95/p99/max K, overflow rate, zero-candidate rate, compression retention, all-match coverage and singleton affected-rate in the machine-readable metrics artifact.

Bootstrap references (or stronger leakage groups), not independent pairs; use paired resampling for ablation differences. Use a final untouched holdout when repeatedly comparing many policies. An illustrative singleton precision audit: with zero affected singletons among n independently sampled singletons, the approximate95% upper affected-rate is3/n; n=3000 only supports roughly1/1000, not zero risk. Cluster dependence reduces effective sample size. Complete truth quality is a prerequisite, not a modeling detail.

### 7.3 Ablation matrix and cost accounting

Every experiment logs delta versus its stated parent, recall changes, singleton errors, wall time, instance, billed hours and actual dollars. Do not charge shared preprocessing repeatedly when reporting marginal cost; report both incremental and cumulative project spend.

\[
\mathrm{CostEfficiency}=\frac{\Delta F_{0.5}}{\mathrm{USD}},\qquad
\mathrm{CostPer0.001}=\frac{0.001\cdot\mathrm{USD}}{\Delta F_{0.5}}
\quad(\Delta F_{0.5}>0).
\]
If gain is nonpositive, label cost-per-improvement “no improvement,” not a negative attractive cost. If gain is NR, these are NR. Compare full-pipeline gains: cheap pairwise improvement that worsens singleton errors is not a win.

The marginal cost limits below are experiment admission caps within the workstream budget, **not additive guaranteed expenditures or measured runtimes**. Reuse candidate/feature artifacts.

| ID | Experiment / parent | Main question | Instance / expected pilot wall time | Marginal compute cap | ΔF0.5 / cost-per.001 |
|---|---|---|---|---:|---|
| E01 | Exact channels only | Cheap anchors and singleton errors | c5.4xlarge, ≤1h | $.82 | NR / NR |
| E02 | + sparse / E01 | Baseline coverage at K≤40 | r5.4xlarge, ≤2h | $2.42 | NR / NR |
| E03 | Frozen dense only | Lexical-divergence recovery | g5.xlarge, ≤1h sample | $1.41 | NR / NR |
| E04 | Sparse+dense / E02 | Marginal recall after compression | r5.4xlarge, ≤1h cached | $1.21 | NR / NR |
| E05 | Engineered score only | Conventional fuzzy baseline | c5.4xlarge, ≤1h | $.82 | NR / NR |
| E06 | Bi-encoder score only | Retrieval score as classifier | c5.4xlarge, ≤.5h cached | $.41 | NR / NR |
| E07 | Tree ensemble / E05 | Feature interactions | c5.4xlarge, ≤2h | $1.63 | NR / NR |
| E08 | Targeted cross-encoder / E07 | Hard-case incremental precision | g5.xlarge, ≤2h sample | $2.82 | NR / NR |
| E09 | Hard negatives / E07 | Deployment-tail discrimination | c5.4xlarge, ≤1h cached | $.82 | NR / NR |
| E10 | Remove/add IDF / E07 | Rarity benefit | c5.4xlarge, ≤.5h | $.41 | NR / NR |
| E11 | Remove/add numeric conflicts / E07 | Branch discrimination | c5.4xlarge, ≤.5h | $.41 | NR / NR |
| E12 | Remove/add ranks/margins / E07 | Useful candidate context | c5.4xlarge, ≤.5h | $.41 | NR / NR |
| E13 | Singleton B/C vs A | Set modeling versus pair complexity | c5.4xlarge, ≤1h | $.82 | NR / NR |
| E14 | Conservative graph / best locked | Gain without singleton damage | c5.4xlarge, ≤.5h small components | $.41 | NR / NR |

**Channel marginal-recall table to fill from actual retrieval:**

| Added channel | Union R@5/10/20/40 | Δmacro candidate recall@40 | Δall-match coverage | Mean/p95 K | Singleton FP after locked scorer | Cost |
|---|---|---|---|---|---|---|
| Phone/domain anchors | NR | NR | NR | NR | NR | NR |
| Address/rare-token anchors | NR | NR | NR | NR | NR | NR |
| Name BM25/character | NR | NR | NR | NR | NR | NR |
| Address sparse | NR | NR | NR | NR | NR | NR |
| Geo/neighbor cells | NR | NR | NR | NR | NR | NR |
| Dense rescue | NR | NR | NR | NR | NR | NR |
| Compression to20/40 | NR | NR | NR | NR | NR | NR |

Both incremental-addition and leave-one-channel-out ablations are needed: two redundant channels can each look weak in isolation from their shared contribution. Keep fusion/compression retraining rules explicit for each comparison.

### 7.4 Compute gates

1. **Cheap, before model complexity:** audit label completeness, branch identity, scorer empty behavior, file widths and normalization; run exact+sparse retrieval. If macro recall@40 is below95%, inspect misses by failure type. Dense retrieval is not the automatic remedy for parsing or label errors.
2. **Sparse tuning:** manual versus automatic fields; rarity, geography and numeric-preserving compression. Target candidate recall above98%, preferably99%, with most K<40 and small documented overflow. If impossible, report the observed frontier rather than claiming the target.
3. **Dense admission:** on a representative large pool, require a prespecified useful incremental recall (example ≥.5 percentage points after compression), recovered hard-case evidence, and projected embedding/index cost within its envelope. Full10M confirmation is mandatory before a final claim.
4. **Matcher admission:** compare tree families, conflict/IDF/context features and singleton policy first. Expensive neural reranking requires a paired entity-bootstrap lower confidence bound above0 and a useful point gain (example ≥.001 macro F), without exceeding a prespecified singleton-error tolerance.
5. **Graph admission:** the same significance/useful-gain test plus no material singleton deterioration and verified branch constraints. Default remains no graph changes.
6. **Spend admission:** before each job require remaining balance ≥ projected job upper cost + $55 rerun + $25 emergency + remaining necessary storage/finalization. Reserve funds cannot be consumed by optional ablations.

These cutoffs are proposed preregistered choices, not claims that any component will attain them. No quantitative accuracy uplift can be estimated reliably without this project's labels; the report quantifies **minimum useful benefit**, runtime sensitivity and cost instead of fabricating expected deltas.

## 8. End-to-end architecture and implementation pseudocode

### 8.1 Architecture

~~~mermaid
flowchart TD
    RAW["Raw data — S3"] --> CP["Country partition — CPU Processing"]
    CP --> N["Normalize/canonicalize — CPU; Parquet to S3"]
    N --> D["Deterministic blocking — CPU; local postings"]
    N --> S["Sparse retrieval — CPU; local disk/EBS"]
    S --> B["BM25"]
    S --> T["Character TF-IDF"]
    S --> R["Rare-token index"]
    N --> E["Optional frozen/adapted embeddings — GPU batches"]
    E --> A["Dense ANN — CPU build/query; local index"]
    D --> U["Candidate union and ID dedup — CPU"]
    B --> U
    T --> U
    R --> U
    A --> U
    U --> C["Candidate compression + audited overflow — CPU"]
    C --> F["Pair features — CPU; streamed Parquet"]
    F --> G["Fast gradient-boosted matcher — CPU"]
    G --> H["Hard-candidate selection — CPU"]
    H --> X["Optional cross-encoder — GPU; capped pairs"]
    G --> P["OOF probability calibration — CPU"]
    X --> P
    P --> Q["Singleton detection — CPU; include empty groups"]
    Q --> SET["Set-level policy — CPU"]
    SET --> GR["Optional conservative graph refinement — CPU"]
    GR --> OUT["Final match sets + audit — S3"]
    F --> M["Train-fold hard-negative mining — CPU"]
    M -. optional adaptation .-> E
    M -. tree fitting .-> G
~~~

Channel branches are parallel logical inputs, not mandatory sequential filters. A candidate need not satisfy every blocker. GPU stages read sequential shards; random metadata/index access remains local to CPU jobs.

### 8.2 Pseudocode conventions

This is implementation-oriented pseudocode, not an executed production implementation. Helper names correspond to algorithms or schemas defined in Sections4–7. Every artifact has a dataset hash, normalization version, split ID and model/index version. Writes are per-shard and atomic; a success manifest is published only after all expected rows are present.

~~~python
def normalize_record(row, locale_tables, parser, phone_parser):
    r = {"id": stable_id(row.source, row.source_id),
         "source": row.source, "raw": preserve_original(row)}
    r["country"] = canonical_country(row.country)
    # Missing/uncertain country is quarantined or resolved using trusted evidence.
    # Never silently compare two known different countries.
    r["name_native"] = unicode_normalize_casefold(row.name)
    r["name_variants"] = unique_nonempty([
        r["name_native"],
        accent_fold(r["name_native"]),
        transliterate_if_supported(r["name_native"]),
        expand_name_abbreviations(r["name_native"], r["country"])
    ])
    r["aliases"] = normalize_alias_list(row.alternate_name)
    addr = parse_with_fallback(row.address, r["country"], parser)
    for field in ["house", "unit", "street", "locality", "city", "state"]:
        r[field] = canonicalize_typed(addr[field], field, r["country"])
    r["postal"] = canonical_postal_string(row.postal, r["country"])
    r["phones"] = valid_phone_set(row.phone, r["country"], phone_parser,
                                  preserve_extensions=True)
    r["domain"], r["url_path"] = canonical_url(row.url,
                                              preserve_branch_path=True)
    r["lat"], r["lon"], r["geo_quality"] = validate_coordinates(row)
    r["category"] = frozen_category_mapping(row.category)
    r["name_numbers"] = typed_numeric_sequences(r["name_native"])
    r["availability"] = field_availability(r)
    r["tokens"] = cached_tokens_and_ngrams(r)
    r["provenance"] = normalization_versions_and_quality(addr)
    return r

def build_indices(candidate_shards, country):
    # External sort keeps dictionary/posting construction bounded.
    for shard in sequential_parquet(candidate_shards, country=country):
        for r in shard:
            append_metadata_row(r.id, compact_metadata(r))
            for field, token in distinct_index_terms(r):
                external_sort_buffer.emit((field, token, r.id))
            for kind, key in safe_anchor_keys(r):
                anchor_sort_buffer.emit((kind, key, r.id))
    postings = sorted_postings_with_df(external_sort_buffer)
    anchors = sorted_postings_with_df(anchor_sort_buffer)
    sparse = build_fielded_topk_index(postings, stored_raw_text=False)
    return publish_index_manifest(sparse, anchors, metadata_offsets())

def sparse_candidates(q, index, quotas):
    hits = []
    for channel in anchor_channels(q):
        bucket = index.anchor_lookup(channel.key)
        if bucket.df <= channel.safe_size:
            hits.extend(with_provenance(bucket.ids, channel))
        else:
            hits.extend(probe_with_secondary_keys(q, bucket, channel))
    for field in ["name", "address"]:
        hits.extend(index.topk(field, query_variants(q, field),
                               k=quotas[field], pruning="upper_bound"))
    # Neighbor cells and non-geographic fallback both remain possible.
    hits.extend(geo_name_candidates(q, include_neighbor_cells=True))
    return hits

def encode_shards(records, model, out, checkpoint_manifest):
    model.eval().to("cuda")
    for shard in records:
        if checkpoint_manifest.completed(shard.id, model.hash):
            continue
        writer = start_atomic_fp16_shard(out, shard.id)
        for batch in length_bucketed_batches(shard, token_cap=128):
            tokens = tokenize_with_workers(batch)
            tokens = pin_and_async_copy(tokens, device="cuda")
            with inference_mode(), autocast():
                vectors = l2_normalize(model(tokens))
            writer.append(batch.ids, vectors.cpu().to("float16"))
        writer.finish_with_checksum()
        checkpoint_manifest.commit(shard.id, model.hash)

def ann_candidates(q_vectors, country_index, raw_fp16, k_dense=40):
    # Never pretend PQ's approximate ranking is exact.
    approximate_scores, rough_ids = country_index.search(q_vectors, k=200)
    valid = rough_ids >= 0  # FAISS uses -1 when neighbors are unavailable.
    vectors = masked_memmap_gather(raw_fp16, rough_ids, valid)
    exact_scores = batched_dot(q_vectors, vectors.to_float32())
    exact_scores[~valid] = -float("inf")
    # Exact relative to stored FP16 vectors, not pre-quantization FP32.
    return topk_valid_ids(rough_ids, exact_scores, valid, k_dense)

def merge_and_compress(q, channel_hits, compressor, target_k=40):
    candidates = merge_by_external_id(channel_hits)
    for c in candidates:
        c.channel_mask = union_channel_flags(c)
        c.rrf = sum(1.0 / (60 + rank) for rank in c.observed_ranks)
        c.cheap = cheap_pair_features(q, c, include_typed_conflicts=True)
        c.priority = compressor.score(c.cheap, c.rrf, c.channel_mask)
    protected = reliable_rare_anchor_hits(q, candidates)
    chosen = recall_tuned_select(candidates, protected, target_k)
    overflow = validated_multiplicity_overflow(q, candidates, chosen)
    log_counts_and_truncation(q.id, candidates, chosen, overflow)
    return deduplicate_ids(chosen + overflow)

def compute_features(reference_batch, candidate_lists, metadata):
    needed_ids = unique_ids(candidate_lists)
    # Read by shard/offset in batches, not one S3 GET per candidate.
    external_rows = metadata.gather_sorted(needed_ids)
    for q in reference_batch:
        for candidate_id in candidate_lists[q.id]:
            r = external_rows[candidate_id]
            f = compare_name_address_contacts(q, r)
            f.update(numeric_conflicts_with_reliability(q, r))
            f.update(geographic_features_or_missing(q, r))
            f.update(retrieval_provenance_features(q.id, candidate_id))
            f.update(pair_availability(q, r))
            yield (q.id, candidate_id, f)

def mine_negatives(train_refs, candidates, complete_truth, scores, budget):
    for q in train_refs:
        positives = complete_truth[q.id]
        eligible = []
        for j in candidates[q.id]:
            if j in positives:
                continue
            if not complete_truth.knows_pair_status(q.id, j):
                continue  # Unknown is not a negative.
            eligible.append((j, scores[q.id, j], provenance(q.id, j)))
        hard = diverse_top_scored(eligible, max_items=4,
                                   diversity=["source", "channel", "branch"])
        easy = sample_verified_country_negatives(q, max_items=2)
        lexical = diverse_lexical_negatives(eligible, max_items=4)
        yield deduplicate_and_cap(q.id, hard + easy + lexical, total_cap=12)
    # For verified singleton q, use up to8 difficult negatives per reference.
    # Refresh once after warm-up, never with outer-validation labels.

def attach_context(groups, first_pass_model):
    for qid, pairs in groups:
        scores = first_pass_model.predict(pairs.features)
        stats = pool_candidate_statistics(scores, pairs.provenance,
                                          include_empty=True)
        yield add_context_features(pairs, stats)
    # Training context must use first_pass_model excluding these qid groups.

def score_pairs(bundle, feature_groups):
    # Identical scoring path for calibration, set training, tuning and serving.
    contextual = attach_context(feature_groups, bundle.first_pass_model)
    raw = bundle.context_matcher.predict(contextual)
    route = bundle.selector.select(raw, contextual)
    neural = bundle.reranker.predict_selected(contextual, route)
    # Disabled components are explicit identity/no-op implementations.
    return bundle.stacker.combine(raw, neural, route), route

def fit_pair_bundle(fit_data, reference_groups):
    first_pass = fit_feature_tree(fit_data)
    # All training context/stacker inputs exclude the scored reference's labels.
    oof_context = grouped_oof_context_features(fit_data, reference_groups)
    context_matcher = fit_context_tree(fit_data, oof_context)
    selector = fit_selection_rule_inside_training(fit_data, reference_groups)
    reranker = fit_optional_reranker_on_training_only(fit_data, selector)
    oof_routed = grouped_oof_routed_scores(fit_data, reference_groups,
                                          selector, reranker)
    stacker = fit_optional_stacker(oof_routed)
    return ScoreBundle(first_pass, context_matcher, selector, reranker, stacker)

def fit_strict_pipeline(outer_train_refs, corpus, frozen_retriever):
    fit_ids, cal_ids, set_ids, tune_ids = grouped_internal_partition(
        outer_train_refs, stratify=["country", "singleton", "multiplicity"])
    # Retrieving against corpus is permitted; these labels are separated.
    compressor = fit_recall_compressor(
        fit_ids, corpus, frozen_retriever, labels=truth(fit_ids))
    data = retrieve_and_featurize(fit_ids | cal_ids | set_ids | tune_ids,
                                 corpus, frozen_retriever,
                                 compressor=compressor)
    scorer = fit_pair_bundle(data[fit_ids], reference_groups=fit_ids)
    cal_raw, cal_route = score_pairs(scorer, data[cal_ids])
    calibrator = choose_route_aware_calibrator(cal_raw, cal_route,
                                    truth(cal_ids),
                                    criterion="NLL_and_tail_reliability")
    set_raw, set_route = score_pairs(scorer, data[set_ids])
    set_scores = calibrator(set_raw, set_route)
    singleton_model = fit_any_match_model(
        pool_with_zero_candidate_rows(set_scores, set_ids),
        labels={i: bool(truth(i)) for i in set_ids})
    tune_raw, tune_route = score_pairs(scorer, data[tune_ids])
    tune_scores = calibrator(tune_raw, tune_route)
    tune_q = singleton_model.predict(pool_with_zero_candidate_rows(
                                    tune_scores, tune_ids))
    policy = optimize_macro_policy(tune_scores, tune_q, truth(tune_ids))
    return scorer, calibrator, singleton_model, policy, compressor
    # Cross-fitting can reclaim training data, but every upstream score used
    # by a fitted downstream stage must be generated without that row's label.

def five_fold_evaluate(all_labeled_refs, fixed_corpus, frozen_retriever):
    folds = grouped_folds(all_labeled_refs, key="Source1EntityID", n=5)
    oof = []
    for fold in folds:
        train_ids, valid_ids = fold.train, fold.valid
        pipeline = fit_strict_pipeline(train_ids, fixed_corpus,
                                       frozen_retriever)
        candidates = retrieve_all(valid_ids, fixed_corpus, frozen_retriever,
                                  compressor=pipeline[-1])
        prediction = final_predict(valid_ids, candidates, pipeline)
        entity_rows = evaluate_complete_sets_per_reference(
            prediction, truth(valid_ids), candidate_lists=candidates,
            empty_empty_score=1)
        persist_reference_predictions_and_metrics(fold.id, prediction, entity_rows)
        oof.extend(entity_rows)
    return entity_weighted_aggregate(oof), paired_entity_bootstrap(oof)

def entity_f05(gold_count, predicted_count, true_positive_count):
    if gold_count == 0 and predicted_count == 0:
        return 1.0
    return 1.25 * true_positive_count / (0.25 * gold_count + predicted_count)

def exact_macro_threshold(candidate_rows, gold_sets, all_reference_ids):
    # Rows are unique(qid, external_id, calibrated_probability).
    # Include gold outside candidate_rows via the complete gold-set size.
    k = {i: 0 for i in all_reference_ids}
    tp = {i: 0 for i in all_reference_ids}
    g = {i: len(gold_sets[i]) for i in all_reference_ids}
    total = sum(entity_f05(g[i], 0, 0) for i in all_reference_ids)
    best_value, best_threshold = total / len(g), float("inf")
    for score, tied_rows in descending_external_sort_grouped(candidate_rows):
        # Aggregate by reference: one huge isotonic-score tie need not
        # materialize millions of pair rows in RAM.
        changes = {}
        for row in tied_rows:
            dk, dtp = changes.get(row.qid, (0, 0))
            changes[row.qid] = (
                dk + 1, dtp + int(row.external_id in gold_sets[row.qid]))
        for i, (dk, dtp) in changes.items():
            total -= entity_f05(g[i], k[i], tp[i])
            k[i] += dk
            tp[i] += dtp
            total += entity_f05(g[i], k[i], tp[i])
        current = total / len(g)
        if current > best_value:  # Prefer higher threshold on exact ties.
            best_value, best_threshold = current, score
    return best_threshold, best_value

def optimize_macro_policy(pair_scores, q_any, gold_sets):
    # Threshold parameters tuned ONLY on the internal tuning groups.
    best = None
    for singleton_gate in preregistered_gate_grid():
        eligible = filter_groups_by_q(pair_scores, q_any, singleton_gate)
        threshold, score = exact_macro_threshold(
            eligible, gold_sets, all_reference_ids=gold_sets.keys())
        candidate = (score, singleton_gate, threshold)
        best = keep_best_with_simple_policy_tie_break(best, candidate)
    # Conditional ambiguity thresholds: small supported grid, inner validation,
    # shrink sparse groups; compare against best global policy.
    return validated_policy(best)

def final_predict(refs, candidates, pipeline):
    scorer, calibrator, singleton_model, policy, compressor = pipeline
    output = []
    for batch in shard_aligned_batches(refs):
        features = compute_features(batch, candidates, local_metadata())
        raw, route = score_pairs(scorer, group_features_by_reference(features))
        p = calibrator(raw, route)
        groups = pool_with_zero_candidate_rows(p, batch)
        q_any = singleton_model.predict(groups)
        for ref in batch:
            selected = policy.select(ref, p[ref.id], q_any[ref.id])
            selected = optional_verified_graph_refinement(ref, selected)
            output.append((ref.id, sorted(unique_external_ids(selected))))
    assert exactly_one_output_row_per_reference(output, refs)
    assert no_known_cross_country_edges(output)
    return atomic_publish_with_audit(output)
~~~

For large production jobs, process by reference shard and preserve boundaries through pooling/calibration. Avoid one Python object per pair in real code; use Arrow arrays and compiled string kernels. Pure-Python pseudocode loops express logic, not the performance implementation.



## 9. SageMaker systems plan and verified pricing

### 9.1 Price basis and instance selection

Prices were verified25 September2026 from the **official regional SageMaker offer catalog**, published2026-09-25T00:03:13Z, with On-Demand terms effective2026-09-01. These are SageMaker Training/Processing prices, not EC2 prices and not Studio notebook rates. [AWS SageMaker price catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json)

| Instance | vCPU | RAM GiB | GPU / VRAM | Local NVMe | Training $/h | Processing $/h | Proposed role |
|---|---:|---:|---|---|---:|---:|---|
| ml.c5.4xlarge | 16 | 32 | None | None | .816 | .816 | Normalization, compiled features, trees |
| ml.c5.9xlarge | 36 | 72 | None | None | 1.836 | 1.836 | Only if useful CPU scaling is measured |
| ml.m5.4xlarge | 16 | 64 | None | None | .922 | .922 | Balanced fallback for moderate RAM |
| ml.r5.4xlarge | 16 | 128 | None | None | 1.210 | 1.210 | Index construction and random metadata |
| ml.r5.8xlarge | 32 | 256 | None | None | 2.419 | 2.419 | Only if active country index exceeds128GiB |
| ml.g4dn.xlarge | 4 | 16 | T4 /16GB |125GB | .736 | .736 | Budget small-model inference candidate |
| ml.g5.xlarge | 4 | 16 | A10G /24GB |250GB | 1.408 | 1.408 | Neural pilot and inference budget basis |
| ml.g5.2xlarge | 8 | 32 | A10G /24GB |450GB | 1.515 | 1.515 | Tokenization/RAM bottleneck alternative |

At equal16vCPU, r5.4xlarge costs about48% more than c5.4xlarge and provides4× RAM. Prefer m5 when64GiB is enough. A10G must deliver more than1.913× T4 throughput to beat its compute cost per record; benchmark both only if the small pilot expense is justified. g5.2xlarge costs about7.6% more than g5.xlarge and can win when extra CPU workers keep the GPU busy.

GPU histogram boosting is worthwhile only if reduced training time exceeds the GPU/CPU price ratio plus startup/I/O overhead. The small labeled matrices expected here make CPU the initial choice. Do not run CPU string extraction on a GPU instance merely because a later stage uses a Transformer.

### 9.2 Data layout and resource plan

~~~text
s3://project/
  raw/source1|source2|source3/
  normalized/country=US|FR|IN/source=.../shard=.../
  dictionaries/version=.../
  indices/version=.../country=.../{anchors,bm25,tfidf,ann}/
  embeddings/model=.../country=.../source=.../shard=.../
  candidates/retriever=.../split=.../country=.../shard=.../
  features/schema=.../split=.../country=.../shard=.../
  models/{tree,encoder,reranker,calibration,singleton,policy}/version=.../
  predictions/run=.../country=.../shard=.../
  checkpoints/job=.../
  metrics/run=.../
  manifests/
~~~

Parquet with Zstandard is the default normalized/feature format; use Arrow in memory, compact numeric IDs, binary index files, and FP16 NumPy-compatible memory maps. Store lengths/offsets for strings instead of Python dictionaries of10M objects. Use128–512MB objects and32–128MB row groups initially; tune based on actual field widths and reads. Partition by country/source/shard, optionally state when useful, without making state a strict blocking requirement.

| Stage | Working RAM estimate | VRAM | Active local disk estimate | Access / implementation |
|---|---|---|---|---|
| Normalize26.4M |8–24GiB |—|50–200GB | Sequential streaming, bounded parser workers |
| Build sparse indices |32–100GiB |—|100–250GB | External sort, compact postings, one country at a time |
| Sparse query + metadata |32–100GiB |—|100–250GB | Memory-mapped index + RAM cache |
| Embedding small model |8–14GiB host |8–20GB |30–150GB | Sequential batches; avoid full corpus in host RAM |
| ANN construction |16–100GiB depending index |Optional |20–100GB | Sample quantizer training; add chunks |
| Feature extraction |8–24GiB |—|30–150GB | Stream100k–500k pair blocks |
| Tree training |8–28GiB target |Optional |10–100GB | Labeled sample; histogram bins |
| Cross-encoder |8–14GiB host |12–24GB |10–50GB | Length-bucketed128–192-token pairs |
| Calibration/set model |<8GiB |—|<10GB | Per-reference OOF rows |
| Final inference |8–24GiB plus staged index where needed |Only optional reranker |30–150GB | Shard-atomic offline output |

All resource ranges are engineering estimates conditional on actual record widths, vocabulary, postings, duplicate degree and implementation. A128GiB node does not guarantee an arbitrarily large Python-built index fits. At10M records, even80 name trigrams per record creates800M postings; a four-byte ID alone costs3.2GB before frequencies, offsets, vocabulary and additional fields. Country sharding and compression matter.

Nominal69.2M pairs ×66 float32 values equals18.27GB before IDs/metadata; expanded missingness/context columns increase this. Do not load that full feature matrix into a32GiB process. Cache cheap record-level representations, compute batch pair features in compiled kernels, and stream predictions. Training needs the labeled/hard-negative subset, not every unlabeled pair.

**Embedding storage calculations**, decimal GB; GiB=GB/1.073741824:

| Dimensions | FP32,10M | FP16,10M | INT8,10M | FP16 GiB |
|---:|---:|---:|---:|---:|
|128|5.12GB|2.56GB|1.28GB|2.384GiB|
|256|10.24GB|5.12GB|2.56GB|4.768GiB|
|384|15.36GB|7.68GB|3.84GB|7.153GiB|
|768|30.72GB|15.36GB|7.68GB|14.305GiB|

These follow \(N d \times\) bytes per element. Multiply by1.173 for candidates plus references, or2.64 for all26.4M; full384-dimensional FP16 evaluation vectors require9.00864GB. Do not embed unused records by default. INT8 needs quantization scales and quality validation; it is not an exact substitute for floats.

For384-dimensional IVF-PQ48×8,10M codes plus64-bit IDs use roughly \(10^7(48+8)=.56\)GB, excluding centroids, tables and list overhead. IVFFlat uses approximately15.44GB, IVF-SQ8 about3.92GB. HNSWFlat atM32 needs about15.36GB vectors plus2.56GB bottom-level links, with upper layers/metadata/build memory extra. Feeding FP16 vectors to HNSWFlat does not make its internal storage FP16. [Official FAISS index specifications](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

Retain FP16 vectors for exact rescoring and index repair; retain an FP32 audit/training sample. Full FP32 storage is justified only by a measured numerical/reranking benefit or reproducibility requirement. Deleting all uncompressed embeddings after PQ construction can make a cheap reindex require expensive re-encoding.

### 9.3 Avoiding data-movement bottlenecks

Normalize and tokenize canonical fields once; reuse identical serialized content across duplicate records through an ID-expansion map. Keep the model resident on GPU. Use inference mode, mixed precision, length bucketing, pinned host buffers, asynchronous prefetch and batched writes. Profile CPU tokenization separately—four host CPUs can starve an A10G.

Stream Parquet/Arrow sequentially from S3, but stage random-access metadata and binary indices on local disk. A memmap reduces copies and lets the OS page cache help; it does not make unlimited random I/O free. Standard FAISS indices generally occupy RAM; use supported on-disk IVF configurations explicitly if needed, rather than assuming arbitrary indices become out-of-core.

Do not shuttle the full ANN index between GPU and CPU for each query. CPU ANN plus GPU encoding is the initial design. If GPU ANN wins, copy each country index once and process many query batches, accounting for16/24GB VRAM limits.

Training File mode stages data before use; FastFile provides on-demand file access and suits sequential scans. Choose according to access pattern. [SageMaker input modes](https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html)

### 9.4 Managed Spot, checkpoints and execution modes

Use **Processing** for preprocessing, index builds, feature generation, ordinary offline embedding inference and evaluation. Use **Training** for trees and Transformer fitting. No persistent endpoint is needed. Batch Transform is an alternative when a trained model artifact/container already supports bulk inference, but Processing is often simpler for combined feature/model pipelines.

Managed Spot is a **Training** feature; do not mark ordinary Processing or Batch Transform jobs as Managed Spot jobs. A custom checkpointed Training container can run a resumable offline workload, but that is extra implementation, not a Processing Spot flag. [Managed Spot](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html), [Processing API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html), [Transform API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html)

For eligible longer training jobs, checkpoint every5–10minutes: weights, optimizer, scheduler, mixed-precision scaler, RNGs, epoch/step, data cursor, split/normalization/model versions. Persist to /opt/ml/checkpoints and configure its S3 destination. Set max wait greater than max runtime and test resume on a small sample. For encoding, publish completed shard IDs and checksums; retry only incomplete shards. Local NVMe is ephemeral. [SageMaker checkpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)

AWS advertises savings up to90%, but the budget assumes0% guaranteed savings. A hypothetical50–70% discount on eight g5 training hours saves only$5.63–7.88. Complex custom Spot engineering is hard to justify for that saving. Use On-Demand for short jobs, build stages without robust checkpoints, and the first throughput pilot.

No distributed training or multi-node Spark is required initially. Network coordination, replicated loading and idle workers can cost more than single-node sharding. Use local development for scripts, example normalization and report analysis; avoid an always-on Studio GPU. If Studio is required, shut down the application/kernel after use and price it separately.

### 9.5 S3/EBS and operational cost controls

The official S3 catalog verified today lists us-east-1 Standard first-tier storage at$.023/GB-month, PUT/COPY/POST/LIST at$.005/1000, and GET/Tier2 at$.0004/1000. [S3 regional price catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.json)

Current SageMaker gp2 volume prices in the regional catalog are Processing$.10/GB-month and Training$.14/GB-month. Example:200GB Processing volume ×120h/730 ×$.10 = $3.29. Three hundred GB-month of S3 = $6.90. These examples plus requests, short training scratch, images and modest logs support a$20 ancillary allowance only if intermediate retention is bounded. Raw sizes were not provided; measure actual GB-days.

GPU families with fixed NVMe use their local storage rather than an additional configurable gp2 volume under the cited Training/Processing APIs. Do not double-charge that disk as EBS; do not request a volume larger than supported local capacity. [Training ResourceConfig](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html), [ProcessingClusterConfig](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingClusterConfig.html)

Keep buckets/jobs in one region. S3-to-AWS-service transfer in the same region has no S3 data-transfer charge, although requests and other network infrastructure can cost money. Avoid paid NAT paths when S3 endpoints suffice; avoid unnecessary internet egress. [Official S3 pricing](https://aws.amazon.com/s3/pricing/)

Set job runtime limits, tag all jobs, and maintain a ledger using actual billed duration and storage lifetimes. Budget alarms are delayed notifications, not a strict real-time kill switch. Before optional jobs, calculate worst-case projected cost; stop when the protected rerun and emergency allowances would be encroached upon.

## 10. Chronological SageMaker runbook and $200 allocation

The sequence below distinguishes sample research from full-scale execution. One instance per row. Hours are **planning estimates requiring pilot validation**, not promises. Dollar amounts exclude storage, which is separately reserved. Gated jobs can be omitted; unused money remains available rather than being automatically spent.

| Step | Purpose; input → output | Mode / instance / count | RAM target | Hours | Spot suitable? / checkpoint | Compute cost |
|---:|---|---|---|---:|---|---:|
|1|Representative raw/label pilot → normalization/throughput and metric manifest|Processing c5.4xlarge ×1|≤24GiB|6|No; cache pilot artifacts|$4.896|
|2|26.4M raw → normalized Parquet/dictionaries|Processing c5.4xlarge ×1|≤24GiB|8|No native Spot; shard completion|$6.528|
|3|10M candidate metadata → exact/sparse indices|Processing r5.4xlarge ×1|≤100GiB|6|No; completed country indices|$7.260|
|4|Fixed large distractor corpus → sparse ablations and fold-safe cheap compressor|Processing r5.4xlarge ×1|≤100GiB|4|No; cache query results|$4.840|
|5|Gated encoder sample comparison/adaptation → frozen or fitted checkpoint|Training/Processing g5.xlarge ×1|≤14GiB host|4|Training only if resumable; model/optimizer state|$5.632|
|6|Gated11.73M records → FP16 embeddings|Processing g5.xlarge ×1|≤14GiB host|6|On-Demand; completed embedding shards|$8.448|
|7|Gated embeddings → ANN/rescue queries + fold-safe fusion/compressor update|Processing r5.4xlarge ×1|≤100GiB|4|No; index/query shards|$4.840|
|8|1.73M references + selected indices → compressed candidates plus audited overflow|Processing r5.4xlarge ×1|≤100GiB|8|No; reference shards|$9.680|
|9|Labeled/validation candidates → reusable feature shards|Processing c5.4xlarge ×1|≤24GiB|2 of12|No; feature shards|$1.632|
|10|Labeled features → provisional tree for mining|Training c5.4xlarge ×1|≤28GiB|1|Short fits; save model|$.816|
|11|Cached train-fold candidates/scores → hard-negative refresh and features|Processing c5.4xlarge ×1|≤24GiB|4|No; sampled negative IDs|$3.264|
|12|Labeled features → selected tree, five-fold OOF predictions, final fit|Training c5.4xlarge ×1|≤28GiB|7|Possible but short fits; completed fold artifacts|$5.712|
|13|Train-fold hard pairs → fitted reranker; untouched held-out pairs → scores|Training/Processing g5.xlarge ×1|≤14GiB host|2|Training only if resumable; model state|$2.816|
|14|OOF scores → calibration/singleton/policy; fold evaluation|Processing c5.4xlarge ×1|<8GiB|1.5|No; small versioned artifacts|$1.224|
|15|Remaining candidate features → final tree scores|Processing c5.4xlarge ×1|≤24GiB|10 of12|No; feature/prediction shards|$8.160|
|16|Gated final hard subset → cross-encoder scores|Processing g5.xlarge ×1|≤14GiB host|2|No; pair-shard progress|$2.816|
|17|Final scores → calibrated probabilities, match sets and audit|Processing c5.4xlarge ×1|<8GiB|.5|No; atomic final manifest|$.408|
| | **First research + first complete execution** | | | **76** | | **$78.972** |

Rows5–7 are optional dense work. Rows13 and16 separate reranker research from final application. Row14 fits the locked policy before row17 applies it. In implementation, preserve these artifact dependencies and never tune on final prediction outcomes. Startup overhead across split jobs must fit the ledger.

Rows4/7 tune compression using training-fold labels; row8 applies the locked compressor.
The100–200-proposal union entails173–346M cheap proposal evaluations. The8h retrieval/compression envelope therefore also requires approximately6000–12000 cheap evaluations/s, alongside index querying; measure both. If a fixed rule compressor is selected instead, its configuration is still locked inside validation.

The12 feature hours cover reusable labeled shards plus remaining final shards; do not recompute identical features unnecessarily. Candidate volume69.2M corresponds to exactly40 per reference; audited overflow increases actual volume. Use measured M in runtime forecasts.

**Required throughput to fit the nominal envelopes:** normalize26.4M in8h≈917records/s; retrieve1.73M in8h≈60references/s; compute69.2M full feature rows over12h≈1602pairs/s; encode11.73M in6h≈543records/s. Finalization and IO must fit these end-to-end rates. With expensive fuzzy features on all pairs, Python implementations can miss them badly; use compiled kernels, feature-family pruning and cached fields. If a gate fails, remove optional neural experiments or revise the architecture transparently rather than hiding a recall reduction.

### 10.1 Complete rerun reservation (allocation table; dependencies follow the runbook)

The second run rebuilds artifacts from raw data using the selected configuration. It repeats final-model fitting and scoring, not every discarded experiment.

| Rerun job | Instance | Hours | Cost |
|---|---|---:|---:|
| Normalize raw |c5.4xlarge|8|$6.528|
| Rebuild sparse indices |r5.4xlarge|6|$7.260|
| Re-embed required records if dense adopted |g5.xlarge|6|$8.448|
| Rebuild/query ANN if adopted |r5.4xlarge|4|$4.840|
| Retrieve final candidates |r5.4xlarge|8|$9.680|
| Recompute features and score |c5.4xlarge|12|$9.792|
| Refit selected tree |c5.4xlarge|2|$1.632|
| Calibration/set policy/finalization |c5.4xlarge|1|$.816|
| Targeted cross-encoder inference if adopted |g5.xlarge|2|$2.816|
| Selected encoder refit if adopted and measured within1h |g5.xlarge|1|$1.408|
| Selected cross-encoder refit if adopted and measured within1h |g5.xlarge|1|$1.408|
| **Planned rerun compute including these optional refits** | | **51** | **$54.628** |
| **Ring-fenced allowance** | | | **$55.000** |

Rerun job ordering follows model/artifact dependencies as above; feature production can precede training, but model scoring follows fitting. The $51.812 data-pipeline rerun can reuse chosen neural weights; the two additional refit hours raise it to $54.628 within the $55 reservation. Admit those refits only after profiling; if they need longer, transfer money from optional research before launch. Repeating all discarded experiments or all five neural folds is beyond this rerun definition. A full sparse-only rerun is cheaper.

### 10.2 Budget ledger

| Workstream | Planned allocation | Notes |
|---|---:|---|
| Pilot / data and throughput audit |$4.896|Includes local-to-cloud readiness checks|
| Normalization + index construction |$13.788|26.4M normalize;10M sparse index|
| Sparse experiments |$4.840|Reuse retrieval artifacts|
| Bi-encoder comparison + full embeddings + ANN |$18.920|Gated; may remain unspent|
| Hard-negative mining |$3.264|One sampled refresh|
| Tree models |$6.528|Winner-focused five-fold work|
| Cross-encoder experiment/selected scoring |$5.632|Gated capped pool|
| Full candidate retrieval + pair features/scoring |$19.472|Shared labeled/final feature budget|
| Calibration/singleton/set evaluation |$1.632|CPU|
| Full end-to-end rerun |$55.000|Protected before optional work|
| Validation reruns/additional OOF checks |$20.000|Cannot imply free neural fold recomputation|
| S3/EBS/requests/images/logs |$20.000|Measure actual GB-days|
| Emergency reserve |$25.000|12.5% of credits|
| **Total planned ceiling** | **$198.972** | **$1.028 unallocated** |

Maximum first-pass GPU time14h; rerun adds8h for inference plus up to2h selected neural refits, totaling24h before emergency allocation. CPU instance-hours:62 first pass +41 rerun +up to24.5 from a$20 c5 validation allowance≈127.5. These are instance-hours, not vCPU-hours. Sequential compute totals roughly six days including validation; actual elapsed time includes job startup, label review and queueing. No labeling labor or local-compute cost is included in SageMaker credits.

## 11. Infrastructure tiers and incremental research value

The labels below reproduce the requested tier names. Only the $200 tier is funded and recommended under the actual constraint.

| Allocation | “Conservative” optional extension | “Recommended” optional extension | Maximum within available credits |
|---|---:|---:|---:|
| Data/index/sparse research and first retrieval |$100|$110|Included in$78.972 first run|
| Feature/tree research and final scoring |$65|$70|Included|
| Dense/hard-negative research |$20|$130|Included, gated|
| Cross-encoder research |$10|$70|Included, gated|
| Validation / repeatability |$35|$65|$20|
| Complete final rerun |$55|$90|$55|
| Storage/operational allowance |$25|$35|$20|
| Emergency reserve |$40|$80|$25|
| First research/execution subtotal shown separately |—|—|$78.972|
| **Total** | **$350** | **$650** | **$198.972** |

**$350 extension:** concentrates extra spending on sparse field/normalization ablations, stronger labeling audits, repeated seeds and feature-model comparison; neural experiments remain limited. At current rates, the extra money buys research confidence and rerun tolerance rather than an inherently better architecture.

**$650 extension:** funds fold-safe learned dense retrievers, more than one hard-negative refresh, targeted multilingual/branch reranking and robust shift evaluations. It still does not justify global LLM inference, a full10M identity GNN or unvalidated transitive closure. The expected score increment is unknown until lower-tier gates pass.

**$198.97 funded plan:** gives the sparse/feature/singleton system a full execution path, permits a small dense/neural option, and protects a second run. The practical research objective is the best validated macro F0.5 achievable at this measured cost, not spending the entire allowance.

## 12. Reproducibility and acceptance criteria

Deliver the chosen configuration only with:

- A data/schema/version manifest, paper-download manifest, current price snapshot and per-job ledger.
- All five outer-fold predictions, complete metric definitions and retrieval recall at every truncation stage.
- An explicit record of label completeness, singleton prevalence, branch ontology and unresolved adjudications.
- Candidate provenance, numeric/contact/geographic conflict explanations, and final reference-level match sets including empty rows.
- Model/calibrator/policy versions bound to the retriever/index version and selection rules.
- Actual measured throughput and spend replacing every NR/estimate, with at least one complete rerun demonstrated before declaring the empirical objective achieved.

This report supplies the research design and costed execution plan. Achieved recall, macro F0.5, feature importance and statistical ablation gains remain open empirical questions; none is inferred from small benchmark pair-classification results.
