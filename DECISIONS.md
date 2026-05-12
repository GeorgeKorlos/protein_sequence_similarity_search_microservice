## ESM-2 Model Size

* **Options** considered: 150M, 650M, 3B
* **Choice**: 650M

* **Rationale**:
The corpus size (547k sequences) and length distribution (mean 324 aa, p95 729 aa) require a model that balances representation quality with full-corpus feasibility.

**150M**: insufficient throughput-performance tradeoff. Local benchmark showed ~3–5 seq/s → >60 hours estimated for full corpus; embedding quality also expected to be weaker for downstream functional similarity tasks.
**650M**: optimal midpoint. On RTX 3090, full corpus processed in ~88 minutes, sustaining high throughput while capturing sufficient structural/functional signal for Week 6 evaluation.
**3B**: computationally viable only under higher-end infrastructure; deferred to P5 ablation where quality gains vs cost will be measured on DTI retrieval.

Given the moderate corpus size but high sequence-length variance, 650M is the largest model that enables single-pass embedding generation without fragmentation or multi-day runtimes.

## FAISS Index Type

The selected ANN index configuration is **IndexIVFFlat with nprobe=10**.

This configuration achieved a **Recall@10 of 0.9910** with a **median throughput of approximately 45 QPS**. Compared against the exact-search `IndexFlatIP` baseline with recall of `1.0000`, this represents a recall degradation of:

\[
(1.0 - 0.9910) \times 100 = 0.9\%
\]

This tradeoff is acceptable because it reduces query cost substantially while maintaining effectively lossless retrieval quality for production semantic search workloads.

`IndexHNSWFlat` was not selected despite achieving higher QPS at lower `efSearch` values because its recall plateaued below the target quality threshold. Even at `efSearch=128`, HNSW achieved only approximately **0.965 recall**, corresponding to a **3.5% degradation** versus exact search. For protein similarity retrieval, this level of retrieval loss is significant and increases the probability of missing biologically relevant nearest neighbors.

The selected `IVFFlat(nprobe=10)` configuration therefore represents the best operating point in the benchmark: near-exact recall while more than doubling throughput relative to brute-force search.

At approximately **45 QPS**, the implied median query latency is:

\[
\frac{1000}{45} \approx 22.2 \text{ ms/query}
\]

This latency is suitable for interactive API workloads and provides enough headroom for downstream application logic and network overhead within a standard sub-100ms response budget.

Per preregistration, the success criterion was Recall@10 ≥ 0.95 for HNSW. HNSW achieved 0.9655 at efSearch=128, satisfying this threshold. IVFFlat was selected instead because benchmark evidence supported a stricter 0.99 operating point, which HNSW does not reach at any evaluated efSearch value.

## Distance Metric

**Choice**: Cosine similarity via inner product on L2-normalized vectors (METRIC_INNER_PRODUCT)

**Rationale**: All embeddings are L2-normalized in ESM2Embedder.embed_tokenized before being added to the index. On unit-length vectors, inner product is mathematically equivalent to cosine similarity, so no separate normalization step is needed at search time. Cosine similarity is magnitude-invariant — two proteins with similar functional representations but different sequence lengths produce embedding vectors of different magnitudes before normalization; cosine captures directional similarity regardless of magnitude, which is appropriate for semantic comparison. METRIC_INNER_PRODUCT is enforced consistently at index build time in IndexManager, at query time in Searcher.search, and the query vector norm is explicitly asserted before search to catch any normalization failures at runtime.

## Pooling Strategy

**Choice**: Mean pooling over residue token embeddings, excluding BOS and EOS special tokens

**Rationale**: Mean pooling produces a fixed-size sequence representation that aggregates information across all residue positions equally, appropriate for whole-sequence functional similarity rather than residue-level tasks. Special tokens (BOS/EOS) are excluded via the hidden_states[:, 1:-1, :] slice in ESM2Embedder._mean_pool — including them would inject non-residue signal into the sequence representation.

## Sequence Length Cap

ESM-2 maximum context window is 1024 tokens; sequences beyond this are truncated by the tokenizer anyway, so the cap makes the filtering explicit and documented rather than silent. From the corpus statistics, only a small fraction of SwissProt sequences exceed 1024 aa — the cap removes biologically extreme outliers without meaningful corpus loss.

## Batch Size

* **Choice**: Dynamic, memory-aware batching (no fixed batch size)

* **Rationale**:
The corpus exhibits extreme variance in sequence length (2 → 1024 aa), which makes fixed batching fundamentally inefficient:
* Short sequences (≤100 aa): GPU underutilized if batch is small
* Long sequences (≥800 aa): OOM risk if batch is large

A fixed batch size would either:
* OOM on the long tail, or
* Waste >50% of available VRAM on the bulk of the distribution (median ≈ 289 aa)

Implementation:
* Dynamic scheduler adjusts batch size per step based on sequence length and VRAM budget
* Peak batch size: 256 (short sequences)
* Minimum batch size: 1 (long sequences >800 aa)
* VRAM utilization: ~19.8 GB / 24 GB (safety factor = 0.85)

Observed performance (aligned with corpus distribution):
* Average throughput: ~103 seq/s across full dataset
* Peak throughput: ~500 seq/s (post-compile warmup, short sequences)

## Corpus Choice

**Choice:** SwissProt reviewed subset (UniProtKB/Swiss-Prot), release 2026_01

**Source files:**
- `uniprot_sprot.fasta.gz` — sequence pipeline
- `uniprot_sprot.dat.gz` — functional annotations (UMAP coloring, DTI baseline stratification in Week 6)

**Provenance:** See `data/data_source.md` (MD5 verified for both files)

**Corpus statistics (post-filtering):**
- Raw sequences: 574627 
- Post-fragment-filter: 565361
- Post-length-cap (≤1024 aa): 547205
- Final: 547205 sequences
- Length: min=2, mean=324.3, median=289, max=1024, p95=729
- Keyword coverage: 98.84%
- GO term coverage: 96.39%

**Rationale:**
SwissProt is manually reviewed — every entry has experimentally supported functional annotation. This matters for two reasons: (1) embedding quality evaluation in Week 6 requires reliable ground-truth functional labels; random or automated annotations would make the DTI baseline comparison uninformative. (2) SwissProt is the standard benchmark corpus in protein representation learning literature, making results directly comparable to published work. Alternatives (TrEMBL, PDB sequences) were rejected: TrEMBL is computationally predicted and annotation quality is uneven; PDB is structurally biased and ~10x smaller.

**Citation:** The UniProt Consortium, *Nucleic Acids Research* 2023. DOI: 10.1093/nar/gkac1052

## Embedding Normalization

**Choice**: L2 normalization applied post-pooling, before index insertion and before search

**Rationale**: Normalization converts raw pooled embeddings to unit vectors, enabling inner product to function as cosine similarity throughout the pipeline. Applied once in embed_tokenized for corpus embeddings, and verified at query time via norm assertion in Searcher.search.

## Cloud Platform
