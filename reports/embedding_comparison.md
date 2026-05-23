# Embedding Comparison: GO-term Functional Similarity Proxy Task

## Task
This task evaluates whether embedding proximity reflects functional similarity
between proteins using Gene Ontology (GO) term overlap as a relevance signal.
Proteins with similar molecular functions, biological processes, or cellular
roles frequently participate in related biological mechanisms, making GO
similarity a useful proxy for assessing whether an embedding space captures
biologically meaningful structure. Since drug–target interaction (DTI)
prediction often depends on recognizing functionally related proteins, this
serves as an indirect retrieval-based proxy for downstream DTI performance.

## Setup
- Corpus: SwissProt 2026_01, 547,205 sequences filtered to proteins with ≥1 GO annotation
- Queries: 500 randomly sampled proteins (seed=42)
- Top-K retrieval: 10
- Metrics: AUROC (GO-term overlap as relevance signal), Mean Reciprocal Rank (MRR), Hit@10

## Results

| Embedder | Embedding Dim | Mean AUROC | Mean MRR | Notes |
|---|---|---|---|---|
| ESM-2 650M | 1280 | 0.706 | 0.991 | fp16 inference, L2-normalized; AUROC computed on top-k candidates only |
| Physicochemical | 4 | 0.515 | 0.426 | 4 AAindex properties, mean-pooled |
| Random | 1280 | 0.485 | 0.299 | seed=42, L2-normalized |

| Embedder | Hit@10 | Scored Queries | Skipped Queries | Pure-positive Rate |
|---|---|---|---|---|
| ESM-2 | 0.998 | 62 | 438 | 87.4% |
| Physicochemical | 0.700 | 329 | 171 | 4.2% |
| Random | 0.694 | 347 | 153 | 0.0% |

## Interpretation
ESM-2 substantially outperforms both baselines and operates in a qualitatively
different retrieval regime. Its MRR of 0.991 and Hit@10 of 0.998 indicate that
relevant proteins almost always appear at the top of the retrieved set.

Of the 438 skipped queries, 437 had all-positive top-10 neighborhoods. ESM-2
consistently retrieves proteins sharing at least one GO term with the query,
leaving no negative examples for AUROC to discriminate. This is a consequence
of highly homogeneous retrieval neighborhoods rather than poor model behavior.

The apparent gap between MRR (0.991) and AUROC (0.706) is therefore not
contradictory. MRR evaluates whether relevant proteins appear near the top of
the ranking and indicates near-perfect early retrieval performance. AUROC, by
contrast, evaluates separation between positive and negative examples across
the full ranked set. Once most retrieved neighborhoods become entirely positive,
AUROC loses much of its discriminative power and is computed only on the
remaining mixed queries.

The physicochemical baseline performs only slightly above random (AUROC: 0.515
vs 0.485), suggesting that a small set of bulk amino-acid properties contains
weak functional information but does not meaningfully organize proteins by
biological role. Random embeddings behave as expected, producing essentially
chance-level performance.

For P5, these results suggest that ESM-2 embeddings provide a useful retrieval
substrate for retrieval-augmented DTI prediction. Retrieved neighbors are
frequently enriched for shared functional context, increasing the probability
that neighboring proteins contribute biologically relevant information to
downstream prediction tasks.

## Limitations
- GO-term overlap provides a coarse approximation of biological function and
  does not capture all mechanistic relationships
- Only 500 queries were sampled and may underrepresent rare functional categories
- The physicochemical baseline uses only four sequence properties and is
  intentionally simple
- Heavy class imbalance created a large number of pure-positive neighborhoods
  for ESM-2, reducing AUROC interpretability
- Functional similarity does not necessarily imply shared drug-binding behavior;
  this remains an indirect proxy for DTI performance